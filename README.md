# TransFi RAG — Phase 1 (Gemini + local hnswlib)

This project implements Phase 1 of the RAG assignment without FAISS. Instead it uses **hnswlib** (local, on-disk, approximate nearest neighbor) for vector search. Everything is stored locally.

## Highlights
- Async scraping (aiohttp)
- Gemini embeddings & generation (google-genai)
- Local hnswlib index + metadata (no FAISS, no external DB)
- CLI: ingest.py and query.py
- Metrics printing

## Run
1. Create venv & install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Set Gemini API key:
```bash
export GOOGLE_API_KEY='YOUR_KEY'
```
3. Ingest:
```bash
python ingest.py --url "https://www.transfi.com" --out-dir ./data/transfi
```
4. Query:
```bash
python query.py --question "What is BizPay?" --index ./data/transfi
```
