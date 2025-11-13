#!/usr/bin/env python3
"""query.py - RAG query interface"""
import argparse, asyncio, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from rag.indexer_hnsw import HNSWIndexer
from rag.qa import answer_question

async def run_single(question):
    """Process single question"""
    # Auto-detect index path
    index_path = "./data/transfi/hnsw_index"
    if not Path(index_path).exists():
        print(f"❌ Index not found at {index_path}")
        print("💡 Run: python ingest.py --url 'https://www.transfi.com' first")
        return
    
    print(f"🔍 Loading index from: {index_path}")
    index = HNSWIndexer.load(index_path)
    
    print(f"❓ Question: {question}")
    res = await answer_question(index, question)
    
    print(f"\n✅ Answer:\n{res['answer']}")
    print(f"\n📚 Sources:")
    for i, s in enumerate(res['sources'][:3]):
        print(f"  {i+1}. {s.get('title')} - {s.get('url')}")
        print(f"     Snippet: {s.get('chunk')[:150].replace('\n',' ')}...")
    
    print(f"\n📊 Metrics:")
    for k, v in res['metrics'].items():
        print(f"  {k}: {v}")

async def run_batch(questions_file, concurrent=False):
    """Process multiple questions"""
    index_path = "./data/transfi/hnsw_index"
    index = HNSWIndexer.load(index_path)
    
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"🔄 Processing {len(questions)} questions {'concurrently' if concurrent else 'sequentially'}")
    
    if concurrent:
        tasks = [answer_question(index, q) for q in questions]
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for q in questions:
            results.append(await answer_question(index, q))
    
    for i, res in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Question {i+1}: {res['question']}")
        print(f"Answer: {res['answer'][:400]}...")
        print(f"Metrics: {res['metrics']}")

def main():
    parser = argparse.ArgumentParser(description='Query TransFi RAG system')
    parser.add_argument('--question', type=str, help='Single question to ask')
    parser.add_argument('--questions', type=str, help='File with multiple questions')
    parser.add_argument('--concurrent', action='store_true', help='Process questions concurrently')
    
    args = parser.parse_args()
    
    if args.question:
        asyncio.run(run_single(args.question))
    elif args.questions:
        asyncio.run(run_batch(args.questions, args.concurrent))
    else:
        print('❌ Provide --question "text" or --questions file.txt')

if __name__ == '__main__':
    main()
