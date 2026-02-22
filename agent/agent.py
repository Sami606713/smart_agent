from langchain.agents import create_agent
from tools.arabic.arabic_tool import get_arabic_knowledge_tools
from tools.eng.eng_tools import get_eng_knowledge_tools
from model.load_model import get_model
from middleware.dynamic_prompt import DynamicPromptMiddleware
from middleware.language_mw import LanguageMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit


# Combine all tools - LanguageMiddleware will filter them dynamically
all_tools = get_arabic_knowledge_tools() + get_eng_knowledge_tools()

agent = create_agent(
    get_model(), 
    tools=all_tools,
    debug=False,
    # Initial system prompt is replaced by LanguageMiddleware
    system_prompt="Initializing agent...",
    middleware=[
        TodoListMiddleware(),
        LanguageMiddleware(verbose=True),
        DynamicPromptMiddleware(verbose=True),
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
    ]
)

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    class MockRuntime:
        def __init__(self, state):
            self.state = state

    async def test_agent():
        print("\n=== Testing Arabic Query ===")
        ar_query = "ما هو دور المعلم؟"
        print(f"Query: {ar_query}")
        ar_input = {"messages": [HumanMessage(content=ar_query)]}
        ar_response = await agent.ainvoke(ar_input)
        print(f"Arabic Response: {ar_response['messages'][-1].content}")

        print("\n=== Testing English Query ===")
        en_query = "What is the role of the teacher?"
        print(f"Query: {en_query}")
        en_input = {"messages": [HumanMessage(content=en_query)]}
        en_response = await agent.ainvoke(en_input)
        print(f"English Response: {en_response['messages'][-1].content}")

    from langchain_core.messages import HumanMessage
    asyncio.run(test_agent())