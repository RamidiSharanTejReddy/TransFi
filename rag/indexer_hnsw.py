# rag/indexer_hnsw.py
import hnswlib
import numpy as np
import json
from pathlib import Path
from typing import List, Dict

class HNSWIndexer:
    def __init__(self, dim: int, path: str = './data/hnsw_index', space: str = 'cosine'):
        self.dim = int(dim) if dim is not None else 0
        self.path = Path(path)
        self.space = space
        self.p = None
        self.metadatas: List[Dict] = []
        self._loaded = False

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """
        embeddings: numpy array shape (n, dim)
        metadatas: list of dicts, length n
        """
        if embeddings is None or embeddings.size == 0:
            return
        embeddings = np.asarray(embeddings, dtype='float32')
        n, dim = embeddings.shape
        # update dim
        self.dim = int(dim)

        # create index and add
        self.p = hnswlib.Index(space=self.space, dim=self.dim)
        # init_index with max_elements >= n
        self.p.init_index(max_elements=max(n, 1000), ef_construction=200, M=16)
        # add items 0..n-1
        self.p.add_items(embeddings, np.arange(n, dtype=np.int32))
        # store metadatas (order must match label ids used)
        # annotate each metadata with embedding_dim for future use (optional)
        for m in metadatas:
            if isinstance(m, dict):
                m.setdefault('embedding_dim', self.dim)
        self.metadatas = list(metadatas)

    def save(self):
        """
        Save index binary + metadata + index info (dim, count)
        """
        self.path.mkdir(parents=True, exist_ok=True)
        idx_file = str(self.path / 'hnsw_index.bin')
        meta_file = str(self.path / 'metadata.jsonl')
        info_file = str(self.path / 'index_info.json')

        if self.p:
            self.p.save_index(idx_file)

        # write metadata lines
        with open(meta_file, 'w', encoding='utf-8') as f:
            for m in self.metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # write index info
        info = {
            "dim": int(self.dim),
            "num_elements": len(self.metadatas)
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """
        Load an HNSW index folder:
          - metadata.jsonl
          - hnsw_index.bin
          - index_info.json
        """
        pth = Path(path)
        meta_file = pth / 'metadata.jsonl'
        idx_file = pth / 'hnsw_index.bin'
        info_file = pth / 'index_info.json'

        metas = []
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        metas.append(json.loads(line.strip()))
                    except Exception:
                        continue

        # read index info first if present
        dim = 0
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                dim = int(info.get('dim', 0))
            except Exception:
                dim = 0
        else:
            # fallback: infer from first metadata entry
            if metas and isinstance(metas[0], dict):
                dim = int(metas[0].get('embedding_dim', 0) or 0)

        inst = cls(dim=dim, path=str(path))
        inst.metadatas = metas

        if idx_file.exists() and inst.dim > 0:
            # load index
            inst.p = hnswlib.Index(space=inst.space, dim=inst.dim)
            inst.p.load_index(str(idx_file))
            inst._loaded = True
            # set ef for search speed/accuracy tradeoff
            try:
                inst.p.set_ef(50)
            except Exception:
                pass
        else:
            inst._loaded = False

        return inst

    def search(self, query_embeddings, k=5):
        """
        query_embeddings: numpy array shape (d,) or (n,d) OR Python list
        returns: list of metadata dicts (top-k) for single query,
                 or list of lists for batch queries
        """
        import numpy as _np  # local import to be safe with module state

        # Coerce to numpy array if necessary
        if query_embeddings is None:
            return [] if not hasattr(query_embeddings, "__len__") or getattr(query_embeddings, "ndim", 1) == 1 else [[] for _ in range(len(query_embeddings))]

        if not hasattr(query_embeddings, "ndim"):
            # likely a Python list: convert
            query_embeddings = _np.asarray(query_embeddings, dtype="float32")

        # Ensure 2D shape for batch and single queries
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if self.p is None:
            # no index loaded; return empty results
            return [] if query_embeddings.shape[0] == 1 else [[] for _ in range(query_embeddings.shape[0])]

        # Perform knn query
        labels, distances = self.p.knn_query(query_embeddings.astype('float32'), k=k)
        results = []
        for lab_list in labels:
            res = []
            for lab in lab_list:
                if lab < 0 or lab >= len(self.metadatas):
                    continue
                res.append(self.metadatas[lab])
            results.append(res)
        return results[0] if len(results) == 1 else results

