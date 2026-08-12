"""
Reads every supported file in data/documents/, splits it into overlapping
token-aware chunks, embeds those chunks locally, and stores them in a
ChromaDB collection on disk.

Run this whenever you add or change documents:

    python ingest.py

You can also ingest a single file incrementally (used by the "quick note"
feature in the UI) via `add_single_file()` without rebuilding everything.
"""

import csv
import io
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm

import config

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:
    _ENC = None

    def count_tokens(text: str) -> int:
        # Rough fallback: ~1.3 tokens per word.
        return max(1, int(len(text.split()) * 1.3))


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs)
                    if line.strip():
                        texts.append(line)
        if texts:
            parts.append(f"Slide {i}: " + " ".join(texts))
    return "\n".join(parts)


def read_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excess blank lines.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_csv(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
        lines.append(pairs)
    return "\n".join(lines)


def read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return json.dumps(data, indent=2, ensure_ascii=False)


READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".pptx": read_pptx,
    ".html": read_html,
    ".htm": read_html,
    ".csv": read_csv,
    ".json": read_json,
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


# --------------------------------------------------------------------------
# Chunking (token-aware, sentence/paragraph boundary friendly)
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(s for s in _SENTENCE_SPLIT.split(para) if s)
    return sentences or [text]


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack sentences into chunks of ~chunk_size_tokens, with a
    sentence-level overlap between consecutive chunks."""
    sentences = _split_into_sentences(text)
    chunks = []
    current, current_tokens = [], 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence)
        if current and current_tokens + s_tokens > chunk_size_tokens:
            chunks.append(" ".join(current))
            # Build overlap from the tail of the current chunk.
            overlap, overlap_tok = [], 0
            for s in reversed(current):
                t = count_tokens(s)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_tok += t
            current, current_tokens = overlap, overlap_tok

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


# --------------------------------------------------------------------------
# Vector store helpers
# --------------------------------------------------------------------------

def _get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def _get_client():
    return chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))


def build_vectorstore():
    """Full rebuild: wipes and re-creates the collection from every file."""
    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = load_documents()

    if not documents:
        print(
            "No documents found. Add .pdf, .md, .txt, .docx, .pptx, .html, "
            f".csv, or .json files to {config.DOCS_DIR} and re-run this script."
        )
        return

    print(f"Loaded {len(documents)} document(s). Chunking...")

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(
            doc["text"], config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
        )
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_idx}-{chunk_idx}")
            all_metadatas.append({"source": doc["source"], "chunk": chunk_idx})

    print(f"Created {len(all_chunks)} chunk(s). Embedding and storing (this may take a moment)...")

    embed_fn = _get_embed_fn()
    client = _get_client()
    # Start fresh each time a full rebuild runs, so deleted/edited docs
    # don't leave stale chunks behind.
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


def add_single_file(path: Path):
    """Incrementally embed and add/replace one file's chunks, without
    rebuilding the whole collection. Used by the UI's quick-note feature."""
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    text = READERS[path.suffix.lower()](path)
    if not text.strip():
        return 0

    embed_fn = _get_embed_fn()
    client = _get_client()
    try:
        collection = client.get_collection(config.COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        collection = client.create_collection(config.COLLECTION_NAME, embedding_function=embed_fn)

    # Remove any existing chunks for this source so re-ingesting a changed
    # file doesn't leave duplicates behind.
    try:
        collection.delete(where={"source": path.name})
    except Exception:
        pass

    chunks = chunk_text(text, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


if __name__ == "__main__":
    build_vectorstore()
