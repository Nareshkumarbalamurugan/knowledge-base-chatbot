"""
Streamlit UI for the Personal Knowledge Base Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st

import config
from chatbot import KnowledgeBaseChatbot

st.set_page_config(page_title="Personal Knowledge Base Chatbot", page_icon="📚")

st.title("📚 Personal Knowledge Base Chatbot")
st.caption(
    "Ask questions about your own documents. Runs fully locally — "
    f"embeddings via `{config.EMBEDDING_MODEL}`, generation via Ollama's `{config.OLLAMA_MODEL}`."
)


@st.cache_resource
def load_bot():
    return KnowledgeBaseChatbot()


try:
    bot = load_bot()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (chunk {s.get('chunk', '?')})")

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, chunks = bot.ask(question)
        st.markdown(answer)
        if chunks:
            with st.expander("Sources"):
                for c in chunks:
                    st.markdown(f"- **{c['source']}**")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [{"source": c["source"]} for c in chunks],
        }
    )

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers questions using retrieval-augmented generation "
        "(RAG) over documents you add to `data/documents/`."
    )
    st.markdown("**To update the knowledge base:**")
    st.code("python ingest.py", language="bash")
