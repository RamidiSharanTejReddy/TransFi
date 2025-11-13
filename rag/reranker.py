# rag/reranker.py - Fixed semantic re-ranking
import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder

class SemanticReranker:
    def __init__(self):
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Re-rank documents using cross-encoder for better relevance"""
        if not documents:
            return documents
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            chunk = doc.get('chunk', '')[:500]  # Limit for cross-encoder
            pairs.append([query, chunk])
        
        # Get relevance scores
        scores = self.rerank_model.predict(pairs)
        
        # Add scores to documents and sort
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Sort by rerank score and return top_k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]

# Global reranker instance
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = SemanticReranker()
    return _reranker
