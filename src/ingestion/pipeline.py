import os
import json

from config import DATA_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_BASE_DIR
from src.ingestion.loader import load_documents


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into overlapping chunks of chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def run_ingestion():
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    print(f"[Ingestion] Loading documents from: {DATA_DIR}")
    documents = load_documents(DATA_DIR)

    if not documents:
        print("[Ingestion] No documents found. Add PDF or .txt files to data/documents/ and re-run.")
        return

    print(f"[Ingestion] Loaded {len(documents)} page(s) across all documents")

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        text_chunks = _chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in text_chunks:
            if chunk.strip():
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk.strip(),
                    "source": doc["source"],
                    "page": doc["page"],
                })
                chunk_id += 1

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[Ingestion] Saved {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    run_ingestion()
