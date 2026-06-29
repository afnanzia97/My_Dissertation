import os
import json

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, CHUNKS_PATH
from src.retrieval.embedder import embed_texts


def build_index():
    """Embed all chunks and save a FAISS index to disk."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. Run ingestion first (--mode ingest)."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("[VectorStore] No chunks to index.")
        return

    print(f"[VectorStore] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts).astype(np.float32)

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[VectorStore] Saved index with {index.ntotal} vectors → {FAISS_INDEX_PATH}")


def load_index():
    """Load the FAISS index and chunks from disk. Returns (index, chunks)."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run --mode ingest first."
        )
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. Run --mode ingest first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks
