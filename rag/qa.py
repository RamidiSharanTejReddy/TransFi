# rag/qa.py (uses HNSWIndexer)
import time, asyncio, os
from typing import List, Dict
from .embedder import embed_texts_async, get_gemini_client as get_genai_client
from .indexer_hnsw import HNSWIndexer
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

async def retrieve_and_rank(index: HNSWIndexer, query: str, top_k=8, embed_model='gemini-embedding-001'):
    t0 = time.time()
    emb = await embed_texts_async([query], model=embed_model)
    query_emb = emb[0]
    docs = index.search(query_emb, k=top_k)
    return docs, time.time() - t0

def build_prompt(question: str, docs: List[Dict]):
    p = "You are an assistant answering questions about TransFi products/solutions. Use the provided excerpts and cite URLs in square brackets.\n\n"
    p += f"Question: {question}\n\n"
    p += "Context excerpts:\n"
    for i, d in enumerate(docs):
        snippet = (d.get('chunk') or '')[:800].replace('\n',' ')
        url = d.get('url')
        p += f"[{i+1}] {url}\n{snippet}\n\n"
    p += "Answer concisely and cite sources like [1], [2]. If you cannot answer, say 'I don't know.'\n\nAnswer:"
    return p

def generate_answer_sync(prompt: str, model: str = 'gemini-1.5-flash'):
    client = get_genai_client()
    resp = client.models.generate_content(model=model, contents=prompt)
    if hasattr(resp, 'text'):
        return resp.text
    if hasattr(resp, 'candidates') and resp.candidates:
        return resp.candidates[0].content
    return str(resp)

async def generate_answer_async(prompt: str, model: str = 'gemini-2.5-pro'):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, generate_answer_sync, prompt, model)

async def answer_question(index: HNSWIndexer, question: str, top_k=8, embed_model='gemini-embedding-001', gen_model='gemini-2.5-pro'):
    t0 = time.time()
    docs, retrieval_time = await retrieve_and_rank(index, question, top_k, embed_model)
    # DEBUG: show top-5 retrieved docs
    print("\n=== Retrieved Documents (Top 5) ===")
    for i, d in enumerate(docs[:5]):
        print(f"[{i+1}]")
        print("URL:", d.get("url"))
        print("Title:", d.get("title"))
        snippet = d.get("chunk", "")[:200].replace("\n", " ")
        print("Snippet:", snippet)
        print("-------------------------------")

    prompt = build_prompt(question, docs)
    start = time.time()
    answer = await generate_answer_async(prompt, model=gen_model)
    llm_time = time.time() - start
    total_time = time.time() - t0
    return {'question': question, 'answer': answer, 'sources': docs, 'metrics': {'total_latency': total_time, 'retrieval_time': retrieval_time, 'llm_time': llm_time, 'docs_retrieved': len(docs)}}
