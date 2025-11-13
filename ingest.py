#!/usr/bin/env python3
"""ingest.py using hnswlib for local vector index
"""
import argparse, asyncio, time, os, json
from pathlib import Path
from rag.extractor import fetch, clean_html, extract_title, find_internal_links
from rag.chunking import char_chunk
from rag.embedder import embed_texts_async
from rag.indexer_hnsw import HNSWIndexer
import aiohttp
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import xml.etree.ElementTree as ET
import aiohttp
from yarl import URL

async def fetch_sitemap(seed_url: str, session: aiohttp.ClientSession, timeout_sec: int = 10):
    """
    Try to download /sitemap.xml or discover sitemap from robots.txt.
    Returns list of absolute URLs (may be empty).
    """
    site = URL(seed_url).origin()
    candidates = [f"{site}/sitemap.xml", f"{site}/sitemap_index.xml"]
    # try robots.txt for sitemap: https://site/robots.txt
    candidates.append(f"{site}/robots.txt")

    found = []
    for c in candidates:
        try:
            async with session.get(c, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    continue
                text = await resp.text()
                # If robots.txt contains sitemap links, extract them
                if c.endswith("robots.txt"):
                    for line in text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            u = line.split(":", 1)[1].strip()
                            if u:
                                candidates.append(u)
                    continue
                # We assume this is an XML sitemap
                try:
                    root = ET.fromstring(text)
                    # common sitemap urlset -> url -> loc
                    for url_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                        found.append(url_el.text)
                    # fallback: any <loc> tags
                    if not found:
                        for loc in root.findall(".//loc"):
                            if loc.text:
                                found.append(loc.text)
                except Exception:
                    # not xml; skip
                    continue
        except Exception:
            continue
    # return unique absolute urls
    return list(dict.fromkeys(found))

async def gather_pages(seed_url: str,
                       concurrency: int,
                       session: aiohttp.ClientSession,
                       max_pages: int = 200,
                       max_depth: int = 4,
                       allowed_host: str = None):
    """
    BFS-style crawler that:
     - starts from multiple seeds (homepage, /products, /solutions, sitemap)
     - follows internal links up to max_depth
     - respects max_pages
    Returns (pages, failures) where pages is list of (url, html)
    """
    sem = asyncio.Semaphore(concurrency)

    async def safe_fetch(u):
        async with sem:
            return await fetch(session, u)

    base = URL(seed_url)
    allowed_host = allowed_host or base.host

    # Seed queue: homepage, /products, /solutions, sitemap discovered pages
    seeds = [str(base), str(base.join(URL("/products"))), str(base.join(URL("/solutions")))]
    # Add sitemap-discovered urls (best-effort)
    try:
        s = await fetch_sitemap(seed_url, session)
        if s:
            # limit sitemap seeds to first 200 to avoid massive sites
            seeds.extend(s[:200])
    except Exception:
        pass

    # normalize and dedupe seeds
    queue = []
    seen = set()
    for s in seeds:
        if not s:
            continue
        u = str(URL(s).with_fragment(None))
        if u not in seen:
            queue.append((u, 0))  # (url, depth)
            seen.add(u)

    pages = []
    failed = []
    # BFS loop
    while queue and len(pages) + len(failed) < max_pages:
        url, depth = queue.pop(0)
        if depth > max_depth:
            continue
        u, status, content = await safe_fetch(url)
        if status != 200:
            failed.append((u, status))
            continue
        pages.append((u, content))

        # find internal links and enqueue (only when depth < max_depth)
        try:
            links = find_internal_links(content, seed_url, allowed_prefixes=None)
        except Exception:
            links = []
        for l in links:
            # normalize
            try:
                L = str(URL(l).with_fragment(None))
            except Exception:
                continue
            # restrict to same host
            try:
                if URL(L).host != allowed_host:
                    continue
            except Exception:
                continue
            if L not in seen:
                # optionally restrict to desired path segments (but relaxed here to get more pages)
                seen.add(L)
                queue.append((L, depth + 1))

        # small safety: if queue grows huge, trim older entries if we're near max_pages
        if len(queue) > max_pages * 3:
            queue = queue[:max_pages * 2]

    return pages, failed

async def run_ingest(url, out_dir, chunk_size, overlap, concurrency, embed_model):
    t0 = time.time()
    out = Path(out_dir)
    raw_dir = out / 'raw_html'
    clean_dir = out / 'clean_text'
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pages, failed = await gather_pages(url, concurrency, session)
    docs = []
    for u, html in pages:
        name = u.replace('https://','').replace('http://','').replace('/','_')
        with open(raw_dir / (name + '.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        text = clean_html(html)
        title = extract_title(html)
        with open(clean_dir / (name + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        docs.append({'url': u, 'title': title, 'text': text})
    chunks = []
    for d in docs:
        cs = char_chunk(d['text'], chunk_size=chunk_size, overlap=overlap)
        for i,c in enumerate(cs):
            chunks.append({'url': d['url'], 'title': d['title'], 'chunk': c, 'chunk_id': f"{d['url']}__{i}"})
    total_chunks = len(chunks)
    batch = 4
    embeddings = []
    metadatas = []
    emb_start = time.time()
    for i in range(0, total_chunks, batch):
        batch_chunks = chunks[i:i+batch]
        texts = [c['chunk'] for c in batch_chunks]
        embs = await embed_texts_async(texts, model=embed_model)
        for j, e in enumerate(embs):
            embeddings.append(e)
            metadatas.append({'url': batch_chunks[j]['url'], 'title': batch_chunks[j]['title'], 'chunk': batch_chunks[j]['chunk'], 'chunk_id': batch_chunks[j]['chunk_id']})
    emb_time = time.time() - emb_start
    embeddings = np.vstack(embeddings) if embeddings else np.zeros((0,0))
    dim = embeddings.shape[1] if embeddings.size else 0
    index = HNSWIndexer(dim=dim, path=str(out / 'hnsw_index'))
    if embeddings.size:
        index.add(embeddings, metadatas)
        index.save()
    total_time = time.time() - t0
    metrics = {
        'total_time_s': round(total_time,3),
        'pages_scraped': len(pages),
        'pages_failed': len(failed),
        'total_chunks': total_chunks,
        'embedding_time_s': round(emb_time,3),
        'index_path': str(out / 'hnsw_index'),
        'errors': failed
    }
    print('\n=== Ingestion Metrics ===')
    for k,v in metrics.items():
        print(f"{k}: {v}")
    with open(out / 'ingest_metrics.json','w',encoding='utf-8') as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--out-dir', default='./data/transfi')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--embed-model', default='gemini-embedding-001')
    args = parser.parse_args()
    asyncio.run(run_ingest(args.url, args.out_dir, args.chunk_size, args.overlap, args.concurrency, args.embed_model))

if __name__ == '__main__':
    main()
