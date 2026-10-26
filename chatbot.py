"""
Core chat logic: given a user question and recent conversation history,
retrieve the most relevant chunks from ChromaDB and ask an LLM (local
Ollama or the Anthropic API) to answer using that context. Falls back to
normal helpful-assistant behavior when nothing relevant is in the
knowledge base, so this also works as a day-to-day personal chatbot.
"""

import os
import threading

import chromadb
from chromadb.utils import embedding_functions

import config

_lock = threading.Lock()
_client = None
_embed_fn = None


def get_chroma_client():
    global _client
    with _lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        return _client


def get_embed_fn():
    global _embed_fn
    with _lock:
        if _embed_fn is None:
            _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL
            )
        return _embed_fn


RAG_SYSTEM_PROMPT = (
    "You are the user's personal assistant. You have access to a knowledge "
    "base built from the user's own documents. When relevant context from "
    "that knowledge base is provided below, ground your answer in it and "
    "mention which source(s) you used. If the provided context doesn't "
    "actually answer the question, say so and then answer normally from "
    "your own general knowledge, clearly distinguishing the two."
)

GENERAL_SYSTEM_PROMPT = (
    "You are the user's helpful personal assistant for day-to-day life "
    "(planning, advice, quick questions, writing help, and so on). Be "
    "concise, warm, and practical."
)


class LLMProvider:
    """Thin wrapper so app code doesn't care whether we're talking to a
    local Ollama model or the Anthropic API."""

    def __init__(self, provider: str = None):
        self.provider = provider or config.LLM_PROVIDER

    def chat(self, system: str, messages: list[dict]) -> str:
        if self.provider == "anthropic":
            return self._chat_anthropic(system, messages)
        return self._chat_ollama(system, messages)

    def _chat_ollama(self, system: str, messages: list[dict]) -> str:
        import ollama

        full_messages = [{"role": "system", "content": system}] + messages
        response = ollama.chat(model=config.OLLAMA_MODEL, messages=full_messages)
        return response["message"]["content"]

    def _chat_anthropic(self, system: str, messages: list[dict]) -> str:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your environment "
                "to use the Anthropic provider, or switch KB_LLM_PROVIDER "
                "back to 'ollama' in config.py."
            )
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")


class KnowledgeBaseChatbot:
    def __init__(self, provider: str = None):
        self.llm = LLMProvider(provider)
        embed_fn = get_embed_fn()
        client = get_chroma_client()
        try:
            self.collection = client.get_collection(
                name=config.COLLECTION_NAME, embedding_function=embed_fn
            )
        except Exception as e:
            raise RuntimeError(
                "No knowledge base found. Run `python ingest.py` first "
                "after adding documents to data/documents/."
            ) from e

    def retrieve(self, question: str, top_k: int = None):
        """Return the top-k most relevant chunks for a question, filtered
        to ones under the relevance-distance threshold."""
        top_k = top_k or config.TOP_K
        results = self.collection.query(query_texts=[question], n_results=top_k)

        chunks = []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []
        for doc, meta, distance in zip(docs, metas, dists):
            chunks.append({"text": doc, "source": meta["source"], "distance": distance})

        relevant = [c for c in chunks if c["distance"] <= config.MAX_RELEVANT_DISTANCE]
        return relevant if relevant else (chunks if not config.ALLOW_GENERAL_CHAT_FALLBACK else [])

    def build_user_turn(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return question
        context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
        return f"Context from knowledge base:\n{context}\n\nQuestion: {question}"

    def ask(self, question: str, history: list[dict] = None, top_k: int = None):
        """Retrieve relevant chunks, build conversation context, and
        generate an answer.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        from earlier turns (already trimmed to config.MAX_HISTORY_TURNS by
        the caller/UI). Returns (answer, chunks_used).
        """
        history = history or []
        chunks = self.retrieve(question, top_k)

        system = RAG_SYSTEM_PROMPT if chunks else GENERAL_SYSTEM_PROMPT
        user_turn = self.build_user_turn(question, chunks)

        messages = history + [{"role": "user", "content": user_turn}]
        answer = self.llm.chat(system, messages)
        return answer, chunks
