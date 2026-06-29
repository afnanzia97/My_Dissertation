import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    return get_model().encode(texts, show_progress_bar=True, convert_to_numpy=True)


def embed_query(query: str) -> np.ndarray:
    return get_model().encode([query], convert_to_numpy=True)[0]
