#!/usr/bin/env python3
"""query.py using HNSW index
"""
import argparse, asyncio, time, json
from dotenv import load_dotenv
load_dotenv()
from rag.indexer_hnsw import HNSWIndexer
from rag.qa import answer_question

async def run_single(index_dir, question):
    index = HNSWIndexer.load(index_dir)
    res = await answer_question(index, question)
    print('\nQuestion:', question)
    print('\nAnswer:\n', res['answer'])
    print('\nSources:')
    for i,s in enumerate(res['sources'][:6]):
        print(f" {i+1}. {s.get('title')} - {s.get('url')}")
        print('    Snippet:', s.get('chunk')[:200].replace('\n',' '), '...')
    print('\nMetrics:', res['metrics'])

async def run_batch(index_dir, qfile, concurrent=False):
    index = HNSWIndexer.load(index_dir)
    with open(qfile, 'r', encoding='utf-8') as f:
        qs = [l.strip() for l in f if l.strip()]
    tasks = []
    if concurrent:
        tasks = [answer_question(index, q) for q in qs]
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for q in qs:
            results.append(await answer_question(index, q))
    for r in results:
        print('\n----\nQ:', r['question'])
        print('A:', r['answer'][:800])
        print('Metrics:', r['metrics'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str)
    parser.add_argument('--questions', type=str)
    parser.add_argument('--index', required=True, help='index directory, e.g. ./data/transfi/hnsw_index')
    parser.add_argument('--concurrent', action='store_true')
    args = parser.parse_args()
    if args.question:
        asyncio.run(run_single(args.index, args.question))
    elif args.questions:
        asyncio.run(run_batch(args.index, args.questions, concurrent=args.concurrent))
    else:
        print('Provide --question or --questions')

if __name__ == '__main__':
    main()
