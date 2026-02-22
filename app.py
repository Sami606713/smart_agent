import streamlit as st
import asyncio
import nest_asyncio
import sys
from pathlib import Path

# Apply nest_asyncio for safe asyncio in Streamlit
nest_asyncio.apply()

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from agent.agent import agent
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# Sidebar
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
# Main header
# ─────────────────────────────────────────────
st.title("🤖 Agentic RAG")
st.markdown(
    "Ask any question about the ingested documents. "
    "**Only final AI responses shown** - tools work silently in background."
)
st.divider()

# ─────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Render chat history (ONLY final responses)
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# FIXED: Get ONLY final AI response (simplest & most reliable)
# ─────────────────────────────────────────────
async def get_final_ai_only(query: str):
    """Run agent and return ONLY the final AI response content."""
    input_data = {
        "messages": [HumanMessage(content=query)],
    }
    
    # Use ainvoke for complete final result (no streaming complexity)
    result = await agent.ainvoke(input_data)
    
    # Extract final AIMessage (last message after all tools)
    final_messages = result["messages"]
    final_msg = final_messages[-1]
    
    if isinstance(final_msg, AIMessage) and final_msg.content:
        return final_msg.content
    
    return "No response generated."

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the documents…"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("**Thinking...**")

        # Run async agent safely
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            full_response = loop.run_until_complete(get_final_ai_only(prompt))
        except Exception as e:
            full_response = f"Error: {str(e)}"
            st.error(f"Agent error: {e}")
        finally:
            loop.close()

        # Display final response
        message_placeholder.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Auto-scroll to bottom
    st.rerun()