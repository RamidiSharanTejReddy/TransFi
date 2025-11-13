# api.py - Add this at the very top
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# api.py - FastAPI server with async ingestion and query endpoints
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import aiohttp
import time
import json
from pathlib import Path
import logging

# Import our existing RAG components
from rag.indexer_hnsw import HNSWIndexer
from rag.qa import answer_question
from ingest import run_ingest

app = FastAPI(title="TransFi RAG API", version="1.0.0")

# Pydantic models
class IngestRequest(BaseModel):
    urls: List[str]
    callback_url: str

class QueryRequest(BaseModel):
    question: str

class BatchQueryRequest(BaseModel):
    questions: List[str]
    callback_url: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    metrics: dict

# Background task for ingestion
async def background_ingest(urls: List[str], callback_url: str):
    """Run ingestion in background and send metrics to callback"""
    start_time = time.time()
    
    try:
        # Run ingestion for each URL
        for url in urls:
            await run_ingest(url)
        
        # Collect metrics
        metrics_file = Path("./data/transfi/metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {"error": "Metrics file not found"}
        
        # Add completion info
        metrics["status"] = "completed"
        metrics["total_ingestion_time"] = round(time.time() - start_time, 1)
        
        # Send to callback URL
        await send_webhook(callback_url, {"metrics": metrics})
        
    except Exception as e:
        error_metrics = {
            "status": "failed",
            "error": str(e),
            "total_ingestion_time": round(time.time() - start_time, 1)
        }
        await send_webhook(callback_url, {"metrics": error_metrics})

async def send_webhook(callback_url: str, data: dict):
    """Send webhook notification"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(callback_url, json=data) as response:
                logging.info(f"Webhook sent to {callback_url}, status: {response.status}")
    except Exception as e:
        logging.error(f"Failed to send webhook to {callback_url}: {e}")

@app.post("/api/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Start ingestion process in background"""
    # Add background task
    background_tasks.add_task(background_ingest, request.urls, request.callback_url)
    
    return {"message": "Ingestion started"}

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process single query"""
    try:
        # Load index
        index_path = "./data/transfi/hnsw_index"
        if not Path(index_path).exists():
            raise HTTPException(status_code=404, detail="Index not found. Run ingestion first.")
        
        index = HNSWIndexer.load(index_path)
        
        # Process query
        result = await answer_question(index, request.question)
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            metrics=result['metrics']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/batch")
async def batch_query_endpoint(request: BatchQueryRequest):
    """Process multiple queries concurrently"""
    try:
        # Load index
        index_path = "./data/transfi/hnsw_index"
        if not Path(index_path).exists():
            raise HTTPException(status_code=404, detail="Index not found. Run ingestion first.")
        
        index = HNSWIndexer.load(index_path)
        
        # Process queries concurrently
        tasks = [answer_question(index, q) for q in request.questions]
        results = await asyncio.gather(*tasks)
        
        # Convert to response format
        responses = [
            QueryResponse(
                question=result['question'],
                answer=result['answer'],
                sources=result['sources'],
                metrics=result['metrics']
            ) for result in results
        ]
        
        # Calculate aggregate metrics
        total_time = sum(r.metrics.get('total_latency', 0) for r in responses)
        total_docs = sum(r.metrics.get('docs_retrieved', 0) for r in responses)
        
        aggregate_metrics = {
            "total_questions": len(request.questions),
            "total_processing_time": round(total_time, 2),
            "average_time_per_question": round(total_time / len(request.questions), 2),
            "total_documents_retrieved": total_docs
        }
        
        response_data = {
            "results": responses,
            "aggregate_metrics": aggregate_metrics
        }
        
        # If callback URL provided, send async
        if request.callback_url:
            await send_webhook(request.callback_url, response_data)
            return {"message": "Batch processing completed, results sent to callback"}
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
