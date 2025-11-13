# api.py - Add this at the very top
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# test_client.py - Fixed test client
import asyncio
import aiohttp
import json
import time

async def test_api():
    """Test all API endpoints"""
    
    print("🧪 Testing TransFi RAG API")
    
    # Test ingestion (async - returns immediately)
    print("\n1. Testing ingestion endpoint...")
    ingest_data = {
        "urls": ["https://www.transfi.com"],
        "callback_url": "http://localhost:8001/webhook"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/ingest", json=ingest_data) as resp:
            result = await resp.json()
            print(f"✅ Ingest response: {result}")
    
    print("📡 Ingestion started in background. Check webhook receiver for completion.")
    print("🔔 Once you see the webhook notification, you can test queries.")
    
    # Test query with existing index (if available)
    print("\n2. Testing single query (using existing index if available)...")
    query_data = {"question": "What is BizPay?"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/query", json=query_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ Query response: {result['answer'][:200]}...")
                print(f"📊 Metrics: {result['metrics']}")
            else:
                error = await resp.json()
                print(f"❌ Query failed: {error['detail']}")
    
    # Test batch query
    print("\n3. Testing batch query...")
    batch_data = {
        "questions": [
            "What is BizPay?",
            "How does TransFi Checkout work?",
            "What are TransFi's solutions for startups?"
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/query/batch", json=batch_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ Batch response: {len(result['results'])} answers received")
                print(f"📊 Aggregate metrics: {result['aggregate_metrics']}")
            else:
                error = await resp.json()
                print(f"❌ Batch query failed: {error['detail']}")

async def test_queries_only():
    """Test only queries (assuming index exists)"""
    print("🔍 Testing queries with existing index...")
    
    # Test single query
    query_data = {"question": "What is BizPay and its features?"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/query", json=query_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ Answer: {result['answer']}")
                print(f"📚 Sources: {len(result['sources'])} documents")
                print(f"📊 Metrics: {result['metrics']}")
            else:
                error = await resp.json()
                print(f"❌ Query failed: {error}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        asyncio.run(test_queries_only())
    else:
        asyncio.run(test_api())
