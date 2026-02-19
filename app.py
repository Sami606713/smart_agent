import streamlit as st
import asyncio
import sys
from pathlib import Path

# Add project root to path (needed when running via streamlit)
sys.path.append(str(Path(__file__).resolve().parent))

from agent.agent import agent

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
# Helper: run async agent synchronously
# ─────────────────────────────────────────────
def run_agent(query: str) -> str:
    """Run the async LangChain agent inside a new event loop."""
    async def _invoke():
        input_data = {
            "messages": [{"role": "user", "content": query}],
        }
        response = await agent.ainvoke(input_data)
        return response["messages"][-1].content

    # Streamlit runs its own event loop; create a new one for the agent
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_invoke())
    finally:
        loop.close()

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the documents…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response…"):
            try:
                answer = run_agent(prompt)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
