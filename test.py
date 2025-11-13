# test_search.py
import numpy as np
from rag.indexer_hnsw import HNSWIndexer
from rag.embedder import embed_texts_async, embed_texts_async, embed_texts_async as embed_async
import asyncio

idx = HNSWIndexer.load("./data/transfi/hnsw_index")
print("Loaded index dim:", idx.dim, "metas:", len(idx.metadatas), "index loaded?", getattr(idx, "_loaded", False))

# synchronous embedding test (use embed_texts_sync if available)
# If you rely on async embedder, run the async wrapper below.
try:
    emb = embed_texts_sync(["What is BizPay?"], model="gemini-embedding-001")[0]
    print("Query emb len (sync):", len(emb))
    res = idx.search(np.array(emb))
    print("Hits:", len(res))
    for r in res[:3]:
        print(r.get("title"), r.get("url"))
except Exception as e:
    print("Sync embed failed, trying async:", e)
    async def run_async():
        v = await embed_async(["What is BizPay?"])
        print("Query emb len (async):", len(v[0]))
        res = idx.search(np.array(v[0]))
        print("Hits:", len(res))
        for r in res[:3]:
            print(r.get("title"), r.get("url"))
    asyncio.run(run_async())
