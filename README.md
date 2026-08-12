# Personal Knowledge Base Chatbot

A local, privacy-friendly chatbot that answers questions using **your own documents**
(PDF, Markdown, TXT, DOCX). No API keys, no cloud calls — everything runs on your
machine using [Ollama](https://ollama.com) for the LLM and
[sentence-transformers](https://www.sbert.net/) + [ChromaDB](https://www.trychroma.com/)
for retrieval.

This is a Retrieval-Augmented Generation (RAG) app: your documents are chunked,
embedded, and stored in a local vector database. When you ask a question, the
most relevant chunks are retrieved and passed to a local LLM as context, so
answers are grounded in your own notes instead of the model's general knowledge.

## Features

- Ingests **PDF, Markdown, TXT, and DOCX** files from a folder
- Chunks and embeds documents locally (no data leaves your machine)
- Semantic search over your knowledge base with ChromaDB
- Chat interface built with Streamlit
- Shows **source citations** (filename + chunk) for every answer
- Runs entirely on a free local LLM via Ollama

## Architecture

```
 documents/  --->  ingest.py  --->  ChromaDB (local vector store)
                                         |
                                         v
   user question --> app.py --> retrieve top-k chunks --> Ollama LLM --> answer + sources
```

## Setup

### 1. Install Ollama and pull a model

Download Ollama from https://ollama.com, then pull a small model:

```bash
ollama pull llama3.2
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your documents

Drop your files (`.pdf`, `.md`, `.txt`, `.docx`) into `data/documents/`.

### 4. Build the knowledge base

```bash
python ingest.py
```

This reads every file in `data/documents/`, splits it into chunks, embeds them,
and stores them in `vectorstore/`.

### 5. Run the chatbot

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) and start asking
questions about your documents.

## Project structure

```
knowledge-base-chatbot/
├── app.py              # Streamlit chat UI
├── chatbot.py           # RAG logic: retrieval + prompting the LLM
├── ingest.py            # Document loading, chunking, embedding
├── config.py             # Central settings (model names, chunk size, etc.)
├── requirements.txt
├── data/documents/      # Put your source files here
└── vectorstore/         # Auto-generated local ChromaDB store
```

## Configuration

Edit `config.py` to change:
- `OLLAMA_MODEL` — which local model to use (default `llama3.2`)
- `EMBEDDING_MODEL` — sentence-transformers model for embeddings
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — how documents are split
- `TOP_K` — how many chunks are retrieved per question

## Why local instead of an API?

This project is built to run entirely offline once set up — no API key, no
per-request cost, and your documents never leave your machine. That makes it
a good fit for notes, journals, or anything you'd rather not send to a third
party.

## Possible extensions

- Swap Ollama for OpenAI/Anthropic by changing `chatbot.py`
- Add conversation memory (multi-turn context)
- Support more file types (e.g. `.pptx`, `.html`)
- Deploy the Streamlit app to Streamlit Community Cloud (note: local Ollama
  won't be reachable from the cloud, so you'd need to swap in an API-based
  model for a hosted deployment)

## License

MIT
