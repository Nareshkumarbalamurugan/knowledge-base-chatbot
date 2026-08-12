"""
Reads every supported file in data/documents/, splits it into overlapping
chunks, embeds those chunks locally, and stores them in a ChromaDB
collection on disk.

Run this whenever you add or change documents:

    python ingest.py
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm

import config


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path: Path) -> str:
    # Keep markdown as plain text; headings/lists still read fine as text.
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def load_documents() -> list[dict]:
    """Return a list of {text, source} dicts for every supported file."""
    docs = []
    if not config.DOCS_DIR.exists():
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(config.DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        reader = READERS[path.suffix.lower()]
        try:
            text = reader(path)
        except Exception as e:
            print(f"  [skip] {path.name}: could not read ({e})")
            continue
        if text.strip():
            docs.append({"text": text, "source": path.name})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character-based chunker."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_vectorstore():
    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = load_documents()

    if not documents:
        print(
            "No documents found. Add .pdf, .md, .txt, or .docx files to "
            f"{config.DOCS_DIR} and re-run this script."
        )
        return

    print(f"Loaded {len(documents)} document(s). Chunking...")

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_idx}-{chunk_idx}")
            all_metadatas.append({"source": doc["source"], "chunk": chunk_idx})

    print(f"Created {len(all_chunks)} chunk(s). Embedding and storing (this may take a moment)...")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    # Start fresh each time ingest.py runs, so deleted/edited docs don't leave stale chunks.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME, embedding_function=embed_fn
    )

    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        collection.add(
            documents=all_chunks[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
            metadatas=all_metadatas[i : i + batch_size],
        )

    print(f"Done. Knowledge base stored in {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
