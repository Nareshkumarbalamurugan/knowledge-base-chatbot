"""
Core RAG logic: given a user question, retrieve the most relevant chunks
from ChromaDB and ask a local Ollama model to answer using only that
context.
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama

import config


class KnowledgeBaseChatbot:
    def __init__(self):
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
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
        """Return the top-k most relevant chunks for a question."""
        top_k = top_k or config.TOP_K
        results = self.collection.query(query_texts=[question], n_results=top_k)

        chunks = []
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"text": doc, "source": meta["source"], "distance": distance})
        return chunks

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        return (
            "You are a helpful assistant answering questions using only the "
            "context below, which comes from the user's personal documents. "
            "If the answer isn't in the context, say you don't have that "
            "information in the knowledge base rather than guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly and concisely, and mention which source(s) you used."
        )

    def ask(self, question: str, top_k: int = None):
        """Retrieve relevant chunks and generate an answer. Returns (answer, chunks)."""
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return "I don't have anything relevant in the knowledge base for that.", []

        prompt = self.build_prompt(question, chunks)
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
        return answer, chunks
