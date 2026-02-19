import streamlit as st
import asyncio
import sys
from pathlib import Path

# Add project root to path (needed when running via streamlit)
sys.path.append(str(Path(__file__).resolve().parent))

from agent.agent import agent
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# Sidebar — workspace selector
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()
    st.caption("**Model →** Groq · openai/gpt-oss-120b")
    st.caption("**Embeddings →** paraphrase-multilingual-MiniLM-L12-v2")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# Main — header
# ─────────────────────────────────────────────
st.title("🤖 Agentic RAG")
st.markdown(
    "Ask any question about the ingested documents. "
    "The agent will search the **Postgres vector store** and answer using retrieved context."
)
st.divider()

# ─────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Render chat history
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# Helper: stream agent response
# ─────────────────────────────────────────────
async def stream_agent(query: str):
    """Stream agent response using agent.astream with stream_mode='messages'."""
    input_data = {
        "messages": [HumanMessage(content=query)],
    }
    
    # Use LangGraph's astream with messages mode for token-level streaming
    async for msg, metadata in agent.astream(input_data, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            yield msg.content

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the documents…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant response with streaming
    with st.chat_message("assistant"):
        def sync_generator():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            gen = stream_agent(prompt)
            try:
                while True:
                    try:
                        yield loop.run_until_complete(gen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        full_response = st.write_stream(sync_generator())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
