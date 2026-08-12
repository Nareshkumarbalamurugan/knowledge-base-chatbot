# Personal Assistant / Knowledge Base Chatbot

A local-first personal chatbot that doubles as a day-to-day assistant and a
Retrieval-Augmented Generation (RAG) app over **your own documents**
(PDF, Markdown, TXT, DOCX, PPTX, HTML, CSV, JSON). Runs fully offline by
default using [Ollama](https://ollama.com) for the LLM and
[sentence-transformers](https://www.sbert.net/) + [ChromaDB](https://www.trychroma.com/)
for retrieval — or switch to the Anthropic API when you want a stronger
model or don't want Ollama running locally.

## What's new in this version

- **More file types**: PDF, Markdown, TXT, DOCX, **PPTX, HTML, CSV, JSON**
- **Smarter chunking**: token-aware, sentence/paragraph-boundary splitting instead of raw character slicing
- **Conversation memory**: multi-turn chat, not just one-shot Q&A
- **General-chat fallback**: if nothing relevant is in your knowledge base, it answers like a normal helpful assistant instead of refusing — so it works for everyday questions, not just document lookup
- **Quick notes**: jot something down from the sidebar and it's searchable immediately, no need to re-run `ingest.py`
- **Swappable LLM provider**: local Ollama *or* the Anthropic API, switchable from the sidebar
- **New chat button** to reset conversation memory

## Architecture

```
 documents/  --->  ingest.py  --->  ChromaDB (local vector store)
                                         |
                                         v
   user question --> app.py --> retrieve relevant chunks --> LLM (Ollama or Anthropic) --> answer + sources
                                         ^
                              (falls back to general chat
                               if nothing relevant is found)
```

## Setup

### 1. Choose your LLM provider

**Option A — local (default, free, private):** install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
```

**Option B — Anthropic API:** copy `.env.example` to `.env`, fill in
`ANTHROPIC_API_KEY`, and set `KB_LLM_PROVIDER=anthropic` (or just pick
"anthropic" in the sidebar at runtime).

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your documents

Drop files (`.pdf`, `.md`, `.txt`, `.docx`, `.pptx`, `.html`, `.csv`, `.json`)
into `data/documents/`.

### 4. Build the knowledge base

```bash
python ingest.py
```

### 5. Run the chatbot

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Ask about
your documents, or just chat — it'll pull in your notes when relevant and
behave like a normal assistant otherwise. Use the sidebar to save quick
notes, switch providers, or start a new chat.

## Project structure

```
knowledge-base-chatbot/
├── app.py              # Streamlit chat UI (memory, quick notes, provider switch)
├── chatbot.py           # RAG + general-chat logic, LLM provider abstraction
├── ingest.py             # Document loading, token-aware chunking, embedding
├── config.py              # Central settings (provider, models, chunking, memory)
├── requirements.txt
├── .env.example          # Template for ANTHROPIC_API_KEY
├── data/documents/      # Put your source files here
└── vectorstore/         # Auto-generated local ChromaDB store
```

## Configuration

Edit `config.py` (or set the matching env vars) to change:
- `LLM_PROVIDER` / `KB_LLM_PROVIDER` — `"ollama"` or `"anthropic"`
- `OLLAMA_MODEL`, `ANTHROPIC_MODEL` — which model to use per provider
- `EMBEDDING_MODEL` — sentence-transformers model for embeddings
- `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` — how documents are split
- `TOP_K` / `MAX_RELEVANT_DISTANCE` — retrieval count and relevance cutoff
- `MAX_HISTORY_TURNS` — how much conversation memory is kept
- `ALLOW_GENERAL_CHAT_FALLBACK` — set `False` to go back to strict "only answer from documents"

## Why this is a good day-to-day assistant, not just a document Q&A tool

Unlike a strict RAG bot that refuses to answer anything outside your
documents, this version treats the knowledge base as *optional context*:
if your question matches your notes, it grounds the answer and cites
sources; if not, it just answers normally, the way any helpful assistant
would. Combined with the quick-notes sidebar, you can build up a personal
knowledge base organically while using it as your everyday chatbot.

## Possible extensions

- Add reminders/tasks with due dates stored alongside notes
- Voice input/output
- Deploy the Streamlit app to Streamlit Community Cloud (use the Anthropic
  provider for hosted deployments, since local Ollama won't be reachable)
- Per-document tags/folders for better organization

## License

MIT
