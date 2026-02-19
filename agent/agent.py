from langchain.agents import create_agent
from tools.qa_tool import get_knowledge_tools
from model.load_model import get_model

agent = create_agent(get_model(), tools=get_knowledge_tools(),debug=True)

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
        print("--- Testing agent ---")
        
        # Simulate state that the agent would provide
        mock_state = {
            "user_preferences": {
                "workspace_id": "arabic",
                "agent_id": None
            }
        }
        
        query = "ما هو دور المعلم؟" # "What is the role of the teacher?"
        print(f"Query: {query}")

        input_data = {
                "messages": [{"role": "user", "content": query}],
            }
        
        print("--- Invoking Agent ---")
        response = await agent.ainvoke(input_data)
        
        print("\nAgent Result:")
        print(response["messages"][-1].content)

    asyncio.run(test_agent())