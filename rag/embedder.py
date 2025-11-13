# rag/embedder.py
"""
Embedder with multi-backend support:
1) Primary remote: Gemini embeddings via google-genai (if configured)
2) Local preferred backend: safetransformer (if installed)
3) Local fallback: sentence-transformers

Behavior:
- embed_texts_async(...) will try Gemini first (if API key present).
- If Gemini fails or is unavailable, it will prefer safetransformer if importable.
- If safetransformer is not available, it will use sentence-transformers.
- All blocking calls run in a ThreadPoolExecutor to keep asyncio responsive.
"""
import os
import asyncio
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

# Try to import Gemini SDK (google-genai)
try:
    from google import genai
    from google.genai.errors import ClientError as GeminiClientError
except Exception:
    genai = None
    GeminiClientError = Exception

# Try to import safetransformer (preferred local)
try:
    import safetransformer as safetf  # hypothetical package name
    _HAS_SAFETRANSFORMER = True
except Exception:
    safetf = None
    _HAS_SAFETRANSFORMER = False

# Fallback: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False

# Threadpool for blocking work
_executor = ThreadPoolExecutor(max_workers=6)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedder")

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
_client = None

def get_gemini_client():
    global _client
    if _client is None:
        if genai is None:
            return None
        if not API_KEY:
            return None
        _client = genai.Client(api_key=API_KEY)
    return _client

# -------------------------
# Helpers for extraction / normalization
# -------------------------
def _extract_vector_from_embedding_obj(obj: Any):
    if isinstance(obj, (list, tuple)):
        return obj
    try:
        if hasattr(obj, "embedding"):
            return obj.embedding
    except Exception:
        pass
    try:
        if hasattr(obj, "dict"):
            d = obj.dict()
        elif hasattr(obj, "to_dict"):
            d = obj.to_dict()
        elif isinstance(obj, dict):
            d = obj
        else:
            d = None
    except Exception:
        d = None
    if isinstance(d, dict):
        for key in ("embedding", "vector", "values", "data"):
            if key in d:
                val = d[key]
                if isinstance(val, dict):
                    for k2 in ("values", "vector", "embedding"):
                        if k2 in val:
                            return val[k2]
                else:
                    return val
    for attr in ("embedding", "vector", "values", "data", "content"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)):
                return val
    return None

def _normalize_embedding_vec(vec):
    if vec is None:
        return None
    try:
        import numpy as np
        if isinstance(vec, np.ndarray):
            return vec.astype("float32").tolist()
    except Exception:
        pass
    if isinstance(vec, (list, tuple)):
        return [float(x) for x in vec]
    return None

# -------------------------
# Local safetransformer backend (preferred local)
# -------------------------
def _init_safetransformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Attempt to initialize a safetransformer-backed embedder.
    The exact API for safetransformer may differ across versions; try common call patterns.
    This function should return a callable embed_fn(texts)->List[List[float]]
    """
    if not _HAS_SAFETRANSFORMER:
        raise RuntimeError("safetransformer not installed")

    # Try hypothetical safetransformer API patterns; handle gracefully
    # NOTE: adjust if your safetransformer API differs.
    try:
        # pattern 1: safetf.Transformer(model_name).encode(...)
        model = safetf.Transformer(model_name)
        def embed_fn(texts):
            embs = model.encode(texts)
            # ensure list of lists
            return [list(map(float, e)) for e in embs]
        return embed_fn
    except Exception:
        pass

    try:
        # pattern 2: safetf.load(model_name).encode(...)
        model = safetf.load(model_name)
        def embed_fn(texts):
            embs = model.encode(texts)
            return [list(map(float, e)) for e in embs]
        return embed_fn
    except Exception:
        pass

    # If we couldn't initialize, raise an error for clarity
    raise RuntimeError("safetransformer installed but unsupported API - please adapt _init_safetransformer_model() to your version")

def embed_safetransformer_sync(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    embed_fn = _init_safetransformer_model(model_name)
    return embed_fn(texts)

async def embed_safetransformer_async(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, embed_safetransformer_sync, texts, model_name)

# -------------------------
# Local sentence-transformers fallback
# -------------------------
_local_st_model = None
def _init_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global _local_st_model
    if _local_st_model is None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed")
        _local_st_model = SentenceTransformer(model_name)
    return _local_st_model

def embed_local_sync(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    model = _init_sentence_transformer(model_name)
    embs = model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    out = []
    for e in embs:
        if hasattr(e, "tolist"):
            arr = e.tolist()
        else:
            arr = list(e)
        out.append([float(x) for x in arr])
    return out

async def embed_local_async(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, embed_local_sync, texts, model_name)

# -------------------------
# Gemini embed call with retries/backoff
# -------------------------
def _call_gemini_embed(client, texts: List[str], model: str):
    if client is None:
        raise RuntimeError("No Gemini client available")
    if hasattr(client, "models") and hasattr(client.models, "embed_content"):
        return client.models.embed_content(model=model, contents=texts)
    if hasattr(client, "embed_content"):
        return client.embed_content(model=model, contents=texts)
    raise AttributeError("Gemini client does not support embed_content API on this SDK/version")

def embed_gemini_sync(texts: List[str], model: str = "gemini-embedding-001", max_retries: int = 4, initial_backoff: float = 1.0):
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client not configured")
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = _call_gemini_embed(client, texts, model=model)
            embeddings_objs = None
            if hasattr(resp, "embeddings"):
                embeddings_objs = resp.embeddings
            elif hasattr(resp, "results"):
                embeddings_objs = resp.results
            elif isinstance(resp, dict):
                embeddings_objs = resp.get("embeddings") or resp.get("results")
            if embeddings_objs is None:
                try:
                    rd = resp.dict() if hasattr(resp, "dict") else (resp.to_dict() if hasattr(resp, "to_dict") else None)
                except Exception:
                    rd = None
                logger.warning("Gemini embed response missing embeddings/results; saved debug.")
                try:
                    import json
                    with open("embed_debug_response.json", "w", encoding="utf-8") as f:
                        if isinstance(rd, dict):
                            json.dump(rd, f, ensure_ascii=False, indent=2)
                        else:
                            f.write(str(resp))
                except Exception:
                    pass
                raise RuntimeError("Gemini embed response missing 'embeddings' or 'results'")
            vectors = []
            for e in embeddings_objs:
                vec = _extract_vector_from_embedding_obj(e)
                vec = _normalize_embedding_vec(vec)
                if vec is None:
                    try:
                        ed = e.dict() if hasattr(e, "dict") else (e.to_dict() if hasattr(e, "to_dict") else None)
                    except Exception:
                        ed = None
                    with open("embed_debug_item.json", "w", encoding="utf-8") as f:
                        import json
                        if isinstance(ed, dict):
                            json.dump(ed, f, ensure_ascii=False, indent=2)
                        else:
                            f.write(str(e))
                    raise RuntimeError("Failed to extract vector for one item (see embed_debug_item.json)")
                vectors.append(vec)
            return vectors
        except Exception as exc:
            is_rate = False
            if isinstance(exc, GeminiClientError):
                try:
                    msg = str(exc).lower()
                    if "quota" in msg or "resource_exhausted" in msg or "too many requests" in msg or "429" in msg:
                        is_rate = True
                except Exception:
                    pass
            if isinstance(exc, AttributeError):
                logger.info("Gemini SDK missing embed method: %s", exc)
                raise
            if is_rate:
                if attempt > max_retries:
                    raise RuntimeError(f"Gemini embedding rate-limited; retries exhausted. Last error: {exc}")
                wait = initial_backoff * (2 ** (attempt - 1))
                wait = wait * (0.7 + 0.6 * random.random())
                wait = min(wait, 60.0)
                logger.warning("Gemini embed rate-limited (attempt %d/%d). backoff %.1fs", attempt, max_retries, wait)
                time.sleep(wait)
                continue
            raise

# -------------------------
# Public async embed function (tries Gemini -> safetransformer -> sentence-transformers)
# -------------------------
async def embed_texts_async(texts: List[str], model: str = "gemini-embedding-001", prefer_local: bool = True):
    """
    Returns list[list[float]]
    - If Gemini configured and available, use it.
    - Else if prefer_local and safetransformer available, use safetransformer.
    - Else fallback to sentence-transformers local model.
    """
    loop = asyncio.get_event_loop()
    client = get_gemini_client()
    # Try remote Gemini first (if available)
    if client is not None:
        try:
            return await loop.run_in_executor(_executor, embed_gemini_sync, texts, model)
        except Exception as e:
            logger.warning("Gemini embedding failed: %s", e)
            # fall through to local
    # Local: try safetransformer if present and preferred
    if prefer_local and _HAS_SAFETRANSFORMER:
        try:
            return await embed_safetransformer_async(texts, model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("safetransformer failed: %s", e)
            # fall back to sentence-transformers
    # Sentence-transformers fallback
    if _HAS_SENTENCE_TRANSFORMERS:
        return await embed_local_async(texts, model_name="all-MiniLM-L6-v2")
    # If nothing available, raise
    raise RuntimeError("No embedding backend available. Install safetransformer or sentence-transformers, or configure Gemini API key.")

# -------------------------
# Backwards-compatible alias used elsewhere
# -------------------------
def get_client():
    return get_gemini_client()
