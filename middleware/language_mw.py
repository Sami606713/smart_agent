from typing import Callable, List, Optional
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langdetect import detect, DetectorFactory
from rich.console import Console
from rich.panel import Panel

# Ensure consistent language detection
DetectorFactory.seed = 0
console = Console()

ARABIC_SYSTEM_PROMPT = """أنت مساعد ذكي ونظام RAG متعدد اللغات.
لديك وصول إلى أدوات البحث والاسترجاع باللغة العربية.

**القواعد:**
1. استخدم دائمًا أدوات البحث المتوفرة قبل الإجابة.
2. اعتمد في إجابتك فقط على الوثائق المسترجعة.
3. باللغة العربية دائمًا إذا كان سؤال المستخدم بالعربية.
4. اذكر معرف الوثيقة (Document ID) المستخدمة في إجابتك.
"""

ENGLISH_SYSTEM_PROMPT = """You are an intelligent assistant and a multilingual RAG system.
You have access to English search and retrieval tools.

**Rules:**
1. ALWAYS use the search tools before answering.
2. Base your answer ONLY on the retrieved documents.
3. Respond in English since the user's query is in English.
4. Cite the Document ID used in your answer.
"""

class LanguageMiddleware(AgentMiddleware):
    """
    Middleware to detect user language, filter relevant tools, 
    and adapt the system prompt accordingly.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _detect_language(self, messages: List) -> str:
        """Detect language from the first human message."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if not content.strip():
                    continue
                try:
                    lang = detect(content)
                    return "ar" if lang == "ar" else "en"
                except Exception:
                    # Fallback to English if detection fails
                    return "en"
        return "en"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools and adapt prompt based on detected language."""
        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        # 1. Detect language (or get from state if already detected)
        language = state.get("language")
        if not language:
            language = self._detect_language(messages)
            if isinstance(state, dict):
                state["language"] = language

        # 2. Filter tools
        all_tools = request.tools or []
        filtered_tools = []
        
        if language == "ar":
            filtered_tools = [t for t in all_tools if "arabic" in t.name.lower()]
        else:
            filtered_tools = [t for t in all_tools if "english" in t.name.lower()]
            
        if self.verbose:
            console.print(f"[bold blue][LANGUAGE_MW][/bold blue] Detected language: [cyan]{language}[/cyan]")
            console.print(f"[bold blue][LANGUAGE_MW][/bold blue] Active tools: [cyan]{', '.join([t.name for t in filtered_tools])}[/cyan]")

        # 3. Localize System Prompt if it's the base prompt
        # We replace the system message if it matches a generic one or if we want to enforce localization
        updated_messages = []
        system_msg_index = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_msg_index = i
                break
        
        localized_prompt = ARABIC_SYSTEM_PROMPT if language == "ar" else ENGLISH_SYSTEM_PROMPT
        
        if system_msg_index != -1:
            # Replace existing system message
            new_messages = list(messages)
            new_messages[system_msg_index] = SystemMessage(content=localized_prompt)
        else:
            # Insert new system message
            new_messages = [SystemMessage(content=localized_prompt)] + list(messages)

        # 4. Invoke the next handler with overridden tools and state
        new_state = dict(state) if isinstance(state, dict) else state
        new_state["messages"] = new_messages
        
        modified_request = request.override(
            tools=filtered_tools,
            state=new_state
        )
        
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Async version of tool filtering and prompt adaptation."""
        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        language = state.get("language")
        if not language:
            language = self._detect_language(messages)
            if isinstance(state, dict):
                state["language"] = language

        all_tools = request.tools or []
        filtered_tools = []
        
        if language == "ar":
            filtered_tools = [t for t in all_tools if "arabic" in t.name.lower()]
        else:
            filtered_tools = [t for t in all_tools if "english" in t.name.lower()]
            
        if self.verbose:
            console.print(f"[bold blue][LANGUAGE_MW] (Async)[/bold blue] Detected language: [cyan]{language}[/cyan]")
            console.print(f"[bold blue][LANGUAGE_MW] (Async)[/bold blue] Active tools: [cyan]{', '.join([t.name for t in filtered_tools])}[/cyan]")

        system_msg_index = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_msg_index = i
                break
        
        localized_prompt = ARABIC_SYSTEM_PROMPT if language == "ar" else ENGLISH_SYSTEM_PROMPT
        
        if system_msg_index != -1:
            new_messages = list(messages)
            new_messages[system_msg_index] = SystemMessage(content=localized_prompt)
        else:
            new_messages = [SystemMessage(content=localized_prompt)] + list(messages)

        new_state = dict(state) if isinstance(state, dict) else state
        new_state["messages"] = new_messages
        
        modified_request = request.override(
            tools=filtered_tools,
            state=new_state
        )
        
        return await handler(modified_request)
