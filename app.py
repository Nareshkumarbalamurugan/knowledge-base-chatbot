"""
Streamlit UI for the Personal Knowledge Base Chatbot.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

import config
import ingest
from chatbot import KnowledgeBaseChatbot

st.set_page_config(page_title="Personal Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Personal Assistant")
st.caption(
    "Ask about your own documents, or just chat about your day — it falls back "
    "to a normal helpful assistant when there's nothing relevant in your notes."
)

# --------------------------------------------------------------------------
# Sidebar: settings, knowledge base management, quick notes
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    provider = st.radio(
        "LLM provider",
        options=["ollama", "anthropic"],
        index=0 if config.LLM_PROVIDER == "ollama" else 1,
        help="Ollama runs fully locally and free. Anthropic needs ANTHROPIC_API_KEY set.",
    )
    model_label = config.OLLAMA_MODEL if provider == "ollama" else config.ANTHROPIC_MODEL
    st.caption(f"Model: `{model_label}`")

    if st.button("🗑️ New chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Quick note")
    st.caption("Jot something down and it's searchable right away — no need to re-run ingest.py.")
    note = st.text_area("Note", label_visibility="collapsed", placeholder="Remember to...")
    if st.button("💾 Save note", disabled=not note.strip()):
        config.QUICK_NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.QUICK_NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n{note.strip()}")
        try:
            n_chunks = ingest.add_single_file(config.QUICK_NOTES_FILE)
            st.success(f"Saved and indexed ({n_chunks} chunk(s)).")
        except Exception as e:
            st.warning(f"Saved to file, but indexing failed: {e}")

    st.divider()
    st.subheader("Knowledge base")
    st.write(f"Documents folder: `{config.DOCS_DIR}`")
    st.markdown("**To rebuild fully after adding files:**")
    st.code("python ingest.py", language="bash")


@st.cache_resource(show_spinner=False)
def load_bot(provider_choice: str):
    return KnowledgeBaseChatbot(provider=provider_choice)


try:
    bot = load_bot(provider)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**")

question = st.chat_input("Ask something, or just say what's on your mind...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build trimmed conversation history for the LLM (plain role/content,
    # no source metadata) so multi-turn context carries over.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ][-(config.MAX_HISTORY_TURNS * 2):]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, chunks = bot.ask(question, history=history)
            except Exception as e:
                answer, chunks = f"Something went wrong: {e}", []
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
