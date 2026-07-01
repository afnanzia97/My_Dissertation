import numpy as np
import faiss

from config import TOP_K
from src.retrieval.embedder import embed_query
from src.retrieval.vector_store import load_index

_index = None
_chunks = None


def _get_index_and_chunks():
    """Lazy-load the FAISS index and chunks once, then cache in memory."""
    global _index, _chunks
    if _index is None or _chunks is None:
        _index, _chunks = load_index()
    return _index, _chunks


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Embed query, search FAISS index, return top_k matching chunks.
    Each result is a dict: {id, text, source, page, score}.
    """
    index, chunks = _get_index_and_chunks()

    query_vec = embed_query(query).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    return results
