from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()   

def get_model():
    return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
if __name__ == "__main__":
    print(get_model())