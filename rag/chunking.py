# rag/chunking.py - Advanced semantic chunking
import re
from typing import List

def semantic_chunk(text: str, chunk_size=1000, overlap=200):
    """Advanced chunking that respects sentence and paragraph boundaries"""
    if not text:
        return []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = get_overlap_text(current_chunk, overlap) + "\n\n" + paragraph
            else:
                # Paragraph itself is too long, split by sentences
                sentences = split_sentences(paragraph)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = get_overlap_text(current_chunk, overlap) + " " + sentence
                        else:
                            # Single sentence too long, force split
                            chunks.extend(char_chunk(sentence, chunk_size, overlap))
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_overlap_text(text: str, overlap_chars: int) -> str:
    """Get last overlap_chars characters, preferring sentence boundaries"""
    if len(text) <= overlap_chars:
        return text
    
    # Try to find sentence boundary within overlap region
    overlap_text = text[-overlap_chars:]
    sentences = split_sentences(overlap_text)
    
    if len(sentences) > 1:
        # Return last complete sentence(s)
        return '. '.join(sentences[-2:]) + '.'
    
    return overlap_text

def char_chunk(text: str, chunk_size=1000, overlap=200):
    """Fallback character-based chunking"""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks
