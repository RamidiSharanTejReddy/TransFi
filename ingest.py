#!/usr/bin/env python3
"""ingest.py - RAG ingestion for TransFi products and solutions"""
import argparse, asyncio, time, os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from rag.extractor import fetch, clean_html, extract_title, find_internal_links
from rag.chunking import char_chunk, semantic_chunk
from rag.embedder import embed_texts_async
from rag.indexer_hnsw import HNSWIndexer
import aiohttp
import numpy as np

async def gather_products_solutions(base_url: str, session: aiohttp.ClientSession, max_pages: int = 100):
    """Scrape Products and Solutions from homepage navigation"""
    sem = asyncio.Semaphore(5)
    
    async def safe_fetch(u):
        async with sem:
            return await fetch(session, u)
    
    pages = []
    failed = []
    seen = set()
    
    # First scrape homepage to find all product/solution links
    print(f"🔍 Discovering links from homepage: {base_url}")
    u, status, content = await safe_fetch(base_url)
    
    if status != 200:
        print(f"❌ Failed to fetch homepage: {status}")
        return pages, failed
    
    # Extract all links from homepage
    try:
        all_links = find_internal_links(content, base_url)
        product_solution_links = []
        
        for link in all_links:
            # Filter for product and solution pages
            if ('/products/' in link or '/solutions/' in link):
                # Skip generic pages, only specific products/solutions
                if not link.endswith('/products') and not link.endswith('/solutions'):
                    product_solution_links.append(link)
                    
        print(f"🎯 Found {len(product_solution_links)} product/solution links")
        
        # Scrape each discovered page
        for url in product_solution_links:
            if url in seen or len(pages) >= max_pages:
                continue
            seen.add(url)
            
            u, status, content = await safe_fetch(url)
            if status == 200:
                pages.append((u, content))
                print(f"✓ Scraped: {u}")
                
                # Look for additional internal links on each page
                try:
                    sub_links = find_internal_links(content, base_url)
                    for sub_link in sub_links:
                        if (('/products/' in sub_link or '/solutions/' in sub_link) and 
                            sub_link not in seen and 
                            not sub_link.endswith('/products') and 
                            not sub_link.endswith('/solutions')):
                            product_solution_links.append(sub_link)
                except Exception:
                    pass
            else:
                failed.append((u, status))
                print(f"✗ Failed: {u} ({status})")
                
    except Exception as e:
        print(f"❌ Error extracting links from homepage: {e}")
        # Fallback to hardcoded URLs if link extraction fails
        fallback_urls = [
            f"{base_url}/products/bizpay",
            f"{base_url}/products/checkouts", 
            f"{base_url}/products/collections",
            f"{base_url}/products/payouts",
            f"{base_url}/products/ramp",
            f"{base_url}/products/single-api",
            f"{base_url}/products/wallet",
            f"{base_url}/solutions/enterprises",
            f"{base_url}/solutions/startups",
            f"{base_url}/solutions/payment-gateway",
            f"{base_url}/solutions/payment-service-provider",
            f"{base_url}/solutions/payroll"
        ]
        
        for url in fallback_urls:
            if len(pages) >= max_pages:
                break
            u, status, content = await safe_fetch(url)
            if status == 200:
                pages.append((u, content))
                print(f"✓ Fallback scraped: {u}")
            else:
                failed.append((u, status))
    
    print(f"\n🎯 Total pages scraped: {len(pages)}")
    return pages, failed

def extract_structured_data(url: str, html: str):
    """Extract structured data as per assignment requirements"""
    title = extract_title(html)
    text = clean_html(html)
    
    # Extract short description (first 200 chars)
    short_description = text[:200].replace('\n', ' ').strip()
    
    return {
        'title': title,
        'url': url,
        'short_description': short_description,
        'long_description': text,  # Full text
        'text': text  # For compatibility
    }

async def run_ingest(url, chunk_size=1000, overlap=200, concurrency=5):
    """Main ingestion flow"""
    print(f"🚀 Starting ingestion from: {url}")
    t0 = time.time()
    
    # Create output directory
    out_dir = Path("./data/transfi")
    raw_dir = out_dir / 'raw_html'
    clean_dir = out_dir / 'clean_text'
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Scraping phase
    scrape_start = time.time()
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pages, failed = await gather_products_solutions(url, session)
    scrape_time = time.time() - scrape_start
    
    # Data extraction phase
    extract_start = time.time()
    docs = []
    for page_url, html in pages:
        # Save raw HTML
        filename = page_url.replace('https://', '').replace('http://', '').replace('/', '_')
        with open(raw_dir / f"{filename}.html", 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Extract structured data
        doc = extract_structured_data(page_url, html)
        docs.append(doc)
        
        # Save structured data in clean_text format
        structured_content = f"""TITLE: {doc['title']}
URL: {doc['url']}
SHORT_DESCRIPTION: {doc['short_description']}
LONG_DESCRIPTION: {doc['long_description']}"""
        
        with open(clean_dir / f"{filename}.txt", 'w', encoding='utf-8') as f:
            f.write(structured_content)
        
        print(f"📄 Extracted: {doc['title']}")

    
    # Chunking phase (use semantic chunking)
    chunk_start = time.time()
    chunks = []
    total_tokens = 0
    for doc in docs:
        # Use semantic chunking instead of character chunking
        doc_chunks = semantic_chunk(doc['text'], chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                'url': doc['url'],
                'title': doc['title'], 
                'chunk': chunk_text,
                'chunk_id': f"{doc['url']}__{i}"
            })
            total_tokens += len(chunk_text.split())
    chunk_time = time.time() - chunk_start

    
    # Embedding phase
    embed_start = time.time()
    embeddings = []
    metadatas = []
    batch_size = 4
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c['chunk'] for c in batch]
        embs = await embed_texts_async(texts)
        
        for j, emb in enumerate(embs):
            embeddings.append(emb)
            metadatas.append(batch[j])
        
        print(f"📊 Embedded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    embed_time = time.time() - embed_start
    
    # Indexing phase
    index_start = time.time()
    embeddings_array = np.vstack(embeddings) if embeddings else np.zeros((0,0))
    dim = embeddings_array.shape[1] if embeddings_array.size else 0
    
    index = HNSWIndexer(dim=dim, path=str(out_dir / 'hnsw_index'))
    if embeddings_array.size:
        index.add(embeddings_array, metadatas)
        index.save()
    index_time = time.time() - index_start
    
    total_time = time.time() - t0
    
    # Print comprehensive metrics
    print("\n" + "="*50)
    print("🎯 INGESTION METRICS")
    print("="*50)
    print(f"Total Time: {total_time:.1f}s")
    print(f"Pages Scraped: {len(pages)}")
    print(f"Pages Failed: {len(failed)}")
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Total Tokens Processed: {total_tokens:,}")
    print(f"Embedding Generation Time: {embed_time:.1f}s")
    print(f"Indexing Time: {index_time:.1f}s")
    print(f"Average Scraping Time per Page: {scrape_time/max(len(pages),1):.1f}s")
    if failed:
        print(f"Errors: {[f'{url} ({status})' for url, status in failed]}")
    print("="*50)
    
    # Save metrics
    metrics = {
        'total_time_s': round(total_time, 1),
        'pages_scraped': len(pages),
        'pages_failed': len(failed),
        'total_chunks': len(chunks),
        'total_tokens': total_tokens,
        'embedding_time_s': round(embed_time, 1),
        'indexing_time_s': round(index_time, 1),
        'avg_scraping_time_per_page': round(scrape_time/max(len(pages),1), 1),
        'errors': [f'{url} ({status})' for url, status in failed]
    }
    
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Ingest TransFi products and solutions')
    parser.add_argument('--url', default='https://www.transfi.com', help='Base URL to scrape')
    args = parser.parse_args()
    
    asyncio.run(run_ingest(args.url))

if __name__ == '__main__':
    main()
