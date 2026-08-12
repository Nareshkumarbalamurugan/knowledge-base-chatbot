"""Central configuration for the personal knowledge-base chatbot."""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "knowledge_base"

# --- Embedding model (runs locally via sentence-transformers) ---
# all-MiniLM-L6-v2 is small, fast, and good enough for a personal KB.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- LLM provider ---
# "ollama"    -> fully local, free, needs the Ollama app running
# "anthropic" -> hosted, needs ANTHROPIC_API_KEY set as an environment variable
LLM_PROVIDER = os.environ.get("KB_LLM_PROVIDER", "ollama")

# Make sure you've run: ollama pull llama3.2
OLLAMA_MODEL = os.environ.get("KB_OLLAMA_MODEL", "llama3.2")

# Any current Claude model string works here (see docs.claude.com for the
# latest list). Requires ANTHROPIC_API_KEY to be set in your environment.
ANTHROPIC_MODEL = os.environ.get("KB_ANTHROPIC_MODEL", "claude-sonnet-5")

# --- Chunking ---
# Chunking is now token-aware (falls back to a word-based estimate if
# tiktoken isn't installed) and tries to break on sentence/paragraph
# boundaries instead of mid-word.
CHUNK_SIZE_TOKENS = 300      # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 50    # overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 4                    # number of chunks retrieved per question
MAX_RELEVANT_DISTANCE = 1.1  # above this, chunks are treated as "not relevant"

# --- Conversation memory ---
MAX_HISTORY_TURNS = 6        # how many prior user/assistant turns to keep in context

# --- General-chat fallback ---
# When nothing relevant is found in the knowledge base, let the assistant
# answer as a normal helpful day-to-day assistant instead of refusing.
ALLOW_GENERAL_CHAT_FALLBACK = True

# --- Supported file extensions ---
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".pptx", ".html", ".htm", ".csv", ".json"}

# --- Quick notes ---
# File that the "quick note" feature in the UI appends to.
QUICK_NOTES_FILE = DOCS_DIR / "quick_notes.md"
