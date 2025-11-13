# rag/embedder.py - Nomic embeddings + Gemini LLM
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

try:
    from google import genai
except Exception:
    genai = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_executor = ThreadPoolExecutor(max_workers=4)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedder")

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
_client = None
_nomic_model = None

def get_gemini_client():
    global _client
    if _client is None:
        if genai is None:
            raise RuntimeError("google-genai not installed")
        if not API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=API_KEY)
    return _client

def get_nomic_model():
    global _nomic_model
    if _nomic_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        _nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    return _nomic_model

def embed_nomic_sync(texts: List[str]):
    model = get_nomic_model()
    embeddings = model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    return [[float(x) for x in emb] for emb in embeddings]

async def embed_texts_async(texts: List[str], model: str = "nomic-embed-text", prefer_local: bool = True):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, embed_nomic_sync, texts)

def get_client():
    return get_gemini_client()
