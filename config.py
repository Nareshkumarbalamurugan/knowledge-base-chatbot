"""Central configuration for the knowledge base chatbot."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "knowledge_base"

# --- Embedding model (runs locally via sentence-transformers) ---
# all-MiniLM-L6-v2 is small, fast, and good enough for a personal KB.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- LLM (served locally via Ollama) ---
# Make sure you've run: ollama pull llama3.2
OLLAMA_MODEL = "llama3.2"

# --- Chunking ---
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 4              # number of chunks retrieved per question

# --- Supported file extensions ---
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
