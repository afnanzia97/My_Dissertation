import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- LLM ---
LLM_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 5

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "documents")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
FAISS_INDEX_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "chunks.json")
