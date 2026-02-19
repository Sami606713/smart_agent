"""
Dynamic Prompt Middleware

Dynamically updates system prompts based on tool results using LLM generation.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

console = Console()


# ========== Meta Prompt Template ==========

META_PROMPT_TEMPLATE = """You are a meta-prompt generator for an AI research assistant.

Your task is to generate a NEW SYSTEM PROMPT for the research assistant based on the current state of the research workflow.
The assistant is evaluated on its ability to complete the user’s task efficiently and accurately.

Success results in a positive score and a bonus reward.

---

## Current Context
- Last Tool Executed: {tool_name}
- Tool Result Preview: {tool_result_preview}
- User’s Original Query: {user_query}
- Available Tools: {available_tools}
---

## Your Objective
Generate a concise, actionable SYSTEM PROMPT that guides the assistant on what to do next.

The generated system prompt MUST:
1. Acknowledge the current workflow state (what tool just ran and what it returned)
2. Decide the next best action: search, refine, synthesize, or answer
3. Provide tool-specific guidance based on the last executed tool
4. Include clear stopping conditions to avoid unnecessary loops

Generate ONLY the system prompt text. Do not include explanations.

If the tool result already sufficiently answers the user query, instruct the assistant to stop and provide the final answer.

---

## Tool Usage Decision Policy (MANDATORY)
The system prompt MUST guide the assistant using the following logic:
- If confidence in answering is high → synthesize and answer immediately
- If key information is missing → determine the most effective tool to fill the gap
- Never repeat a search unless the query is explicitly refined

---
"""

META_PROMPT_TEMPLATE_AR = """أنت محيا (Meta-Prompt Generator) لمساعد أبحاث ذكاء اصطناعي.

مهمتك هي إنشاء "موجه نظام" (SYSTEM PROMPT) جديد لمساعد البحث بناءً على الحالة الحالية لسير العمل.

---

## السياق الحالي
- آخر أداة تم تنفيذها: {tool_name}
- معاينة نتيجة الأداة: {tool_result_preview}
- استعلام المستخدم الأصلي: {user_query}
- الأدوات المتاحة: {available_tools}
---

## هدفك
أنشئ موجه نظام (SYSTEM PROMPT) موجزًا وقابلًا للتنفيذ يوجه المساعد بشأن ما يجب فعله بعد ذلك.

يجب أن يقوم موجه النظام بـ:
1. الإقرار بحالة سير العمل الحالية (ما هي الأداة التي عملت وماذا أرجعت).
2. اتخاذ القرار بشأن الإجراء التالي: بحث، تحسين، تركيب، أو إجابة.
3. تقديم إرشادات خاصة بالأداة بناءً على آخر أداة تم تنفيذها.
4. تضمين شروط إيقاف واضحة لتجنب الحلقات غير الضرورية.

أنشئ نص موجه النظام فقط باللغة العربية. لا تدرج أي تفسيرات.

---
"""


# ========== Context Schema ==========

@dataclass
class ToolContext:
    """Context schema to track tool execution history."""
    last_tool_name: str = ""
    last_tool_result: str = ""
    tool_history: list = field(default_factory=list)


# ========== Helper Functions ==========

def _extract_user_query(messages: list) -> str:
    """Extract the original user query from message history."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content[:500]  # Truncate for prompt
    return "Unknown query"


def _get_available_tools(request: ModelRequest) -> str:
    """Get list of available tools from request."""
    try:
        tools = getattr(request, 'tools', [])
        if tools:
            tool_names = [getattr(t, 'name', str(t)) for t in tools]
            return ", ".join(tool_names[:10])  # Limit to 10 tools
    except Exception:
        pass
    return "search tools, content extraction, research tools"


def _extract_model_from_request(request: ModelRequest) -> Optional[BaseChatModel]:
    """Extract the base model from the ModelRequest."""
    try:
        # Try to get model from request attributes
        if hasattr(request, 'model') and request.model is not None:
            return request.model

        # Try to get from runtime
        if hasattr(request, 'runtime') and request.runtime is not None:
            if hasattr(request.runtime, 'model'):
                return request.runtime.model

        # Try to get from state
        model = request.state.get("model")
        if model is not None:
            return model

    except Exception:
        pass

    return None


def _get_default_initial_prompt() -> str:
    """Minimal default prompt when no tool has been called yet."""
    return """You are an intelligent research assistant. Analyze the user's query and use appropriate tools to find accurate information.

**IMPORTANT - Google Dork Search Technique:**
When using search tools, ALWAYS format your queries using Google Dork operators for efficient searching:
- `site:domain.com` - Search within specific domains (e.g., site:stackoverflow.com)
- `filetype:pdf` - Filter by file type (pdf, doc, xls)
- `intitle:keyword` - Keywords in page title
- `inurl:keyword` - Keywords in URL
- `"exact phrase"` - Exact phrase matching with quotes
- `OR` - Match either term
- `-term` - Exclude terms (e.g., -pinterest -youtube)
- `after:YYYY-MM-DD` - Recent content only
"""


def _get_fallback_prompt(tool_name: str, tool_result: str) -> str:
    """Fallback prompt when model generation fails."""
    return f"""You are an intelligent research assistant.

The tool '{tool_name}' just completed. Analyze the results and decide your next action.
If you have sufficient information, provide a comprehensive answer. Otherwise, continue researching.

**Google Dork Search Reminder:**
When searching, use Google Dork operators for better results:
- `site:domain.com` for specific domains
- `filetype:pdf` for documents
- `intitle:keyword` for title matches
- `"exact phrase"` for precise matching
- `-exclude` to filter out unwanted results"""


def _generate_prompt_with_model(
    model: BaseChatModel,
    tool_name: str,
    tool_result: str,
    user_query: str,
    available_tools: str,
    current_phase: str = "unknown",
    language: str = "en"
) -> str:
    """Use the model to generate a dynamic system prompt."""
    # Truncate tool result to avoid token limits
    tool_result_preview = tool_result[:500] if tool_result else "No result"

    # Select template based on language
    template = META_PROMPT_TEMPLATE_AR if language == "ar" else META_PROMPT_TEMPLATE

    # Format the meta-prompt
    meta_prompt = template.format(
        tool_name=tool_name,
        tool_result_preview=tool_result_preview,
        user_query=user_query,
        available_tools=available_tools,
        current_phase=current_phase
    )

    if True: # verbose equivalent
        console.print(Panel(f"[bold magenta]META_PROMPT[/bold magenta] Generating dynamic system prompt with LLM\n"
                      f"[bold cyan]TOOL:[/bold cyan] {tool_name}\n"
                      f"[bold cyan]PHASE:[/bold cyan] {current_phase}",
                      title="Dynamic Prompt Generation", border_style="magenta"))

    # Call the model to generate the prompt
    response = model.invoke([HumanMessage(content=meta_prompt)])

    # Extract the generated prompt
    generated_prompt = response.content if hasattr(response, 'content') else str(response)

    return generated_prompt.strip()


async def _agenerate_prompt_with_model(
    model: BaseChatModel,
    tool_name: str,
    tool_result: str,
    user_query: str,
    available_tools: str,
    current_phase: str = "unknown",
    language: str = "en"
) -> str:
    """Use the model to generate a dynamic system prompt asynchronously."""
    # Truncate tool result to avoid token limits
    tool_result_preview = tool_result[:500] if tool_result else "No result"

    # Select template based on language
    template = META_PROMPT_TEMPLATE_AR if language == "ar" else META_PROMPT_TEMPLATE

    # Format the meta-prompt
    meta_prompt = template.format(
        tool_name=tool_name,
        tool_result_preview=tool_result_preview,
        user_query=user_query,
        available_tools=available_tools,
        current_phase=current_phase
    )

    if True: # verbose equivalent
        console.print(Panel(f"[bold magenta]META_PROMPT[/bold magenta] Generating dynamic system prompt with LLM (Async)\n"
                      f"[bold cyan]TOOL:[/bold cyan] {tool_name}\n"
                      f"[bold cyan]PHASE:[/bold cyan] {current_phase}",
                      title="Dynamic Prompt Generation (Async)", border_style="magenta"))

    # Call the model to generate the prompt asynchronously
    response = await model.ainvoke([HumanMessage(content=meta_prompt)])

    # Extract the generated prompt
    generated_prompt = response.content if hasattr(response, 'content') else str(response)

    return generated_prompt.strip()


# ========== Middleware Class ==========

class DynamicPromptMiddleware(AgentMiddleware):
    """Dynamically update system prompt based on tool responses using LLM generation."""

    def __init__(
        self,
        verbose: bool = True
    ):
        """
        Initialize dynamic prompt middleware.

        Args:
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self.base_system_prompt = None

    def _get_tool_context(self, request: ModelRequest) -> tuple[str, str]:
        """Extract tool name and result from message history."""
        tool_name = ""
        tool_result = ""

        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                tool_result = msg.content
                break

        return tool_name, tool_result

    def _generate_dynamic_prompt(
        self,
        request: ModelRequest,
        tool_name: str,
        tool_result: str
    ) -> str:
        """Generate a dynamic prompt using the model from request."""
        model = _extract_model_from_request(request)

        if model is None:
            if self.verbose:
                console.print("[yellow][⚠️ WARNING] No model found in request, using fallback prompt[/yellow]")
            return _get_fallback_prompt(tool_name, tool_result)

        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        user_query = _extract_user_query(messages)
        available_tools = _get_available_tools(request)
        current_phase = state.get("current_phase", "unknown") if isinstance(state, dict) else getattr(state, "current_phase", "unknown")
        language = state.get("language", "en") if isinstance(state, dict) else getattr(state, "language", "en")
        try:
            new_prompt = _generate_prompt_with_model(
                model=model,
                tool_name=tool_name,
                tool_result=tool_result,
                user_query=user_query,
                available_tools=available_tools,
                current_phase=current_phase,
                language=language
            )
            
            if self.verbose:
                console.print(Panel(new_prompt, title="[bold green]✨ GENERATED PROMPT[/bold green]", border_style="green"))
            
            return new_prompt
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red][⚠️ ERROR] Prompt generation failed:[/bold red] {e}")
            return _get_fallback_prompt(tool_name, tool_result)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Wrap model call to dynamically update system prompt based on tool responses."""
        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        # Extract existing system prompt or use default
        system_prompt = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
                break

        if not system_prompt:
            system_prompt = _get_default_initial_prompt()
            if self.verbose:
                console.print("[bold blue][SYSTEM_PROMPT] Using default initial prompt[/bold blue]")

        self.base_system_prompt = system_prompt

        # Check if there are tool responses
        tool_name, tool_result = self._get_tool_context(request)

        if tool_name and tool_result:
            # Generate new system prompt dynamically
            new_prompt = self._generate_dynamic_prompt(request, tool_name, tool_result)

            # Update messages with new system prompt
            updated_messages = self._update_system_message(messages, new_prompt)

            # Merge messages into the existing state
            new_state = dict(request.state)
            new_state["messages"] = updated_messages
            
            # Override the request with updated state
            modified_request = request.override(state=new_state)
            return handler(modified_request)
        else:
            # First call or no tool responses yet
            if self.verbose:
                console.print("[bold cyan][FIRST_CALL] No tool responses yet, using base system prompt[/bold cyan]")
            
            # Ensure system message is in messages
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
                modified_request = request.override(state={"messages": messages})
                return handler(modified_request)

            return handler(request)

    async def _agenerate_dynamic_prompt(
        self,
        request: ModelRequest,
        tool_name: str,
        tool_result: str
    ) -> str:
        """Generate a dynamic prompt using the model from request asynchronously."""
        model = _extract_model_from_request(request)

        if model is None:
            if self.verbose:
                console.print("[yellow][⚠️ WARNING] No model found in request, using fallback prompt[/yellow]")
            return _get_fallback_prompt(tool_name, tool_result)

        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        user_query = _extract_user_query(messages)
        available_tools = _get_available_tools(request)
        current_phase = state.get("current_phase", "unknown") if isinstance(state, dict) else getattr(state, "current_phase", "unknown")
        language = state.get("language", "en") if isinstance(state, dict) else getattr(state, "language", "en")
        try:
            new_prompt = await _agenerate_prompt_with_model(
                model=model,
                tool_name=tool_name,
                tool_result=tool_result,
                user_query=user_query,
                available_tools=available_tools,
                current_phase=current_phase,
                language=language
            )
            
            if self.verbose:
                console.print(Panel(new_prompt, title="[bold green]✨ GENERATED PROMPT (Async)[/bold green]", border_style="green"))
            
            return new_prompt
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red][⚠️ ERROR] Prompt generation failed:[/bold red] {e}")
            return _get_fallback_prompt(tool_name, tool_result)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Wrap model call asynchronously to dynamically update system prompt based on tool responses."""
        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        # Extract existing system prompt or use default
        system_prompt = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
                break

        if not system_prompt:
            system_prompt = _get_default_initial_prompt()
            if self.verbose:
                console.print("[bold blue][SYSTEM_PROMPT] Using default initial prompt[/bold blue]")

        self.base_system_prompt = system_prompt

        # Check if there are tool responses
        tool_name, tool_result = self._get_tool_context(request)

        if tool_name and tool_result:
            # Generate new system prompt dynamically asynchronously
            new_prompt = await self._agenerate_dynamic_prompt(request, tool_name, tool_result)

            # Update messages with new system prompt
            updated_messages = self._update_system_message(messages, new_prompt)

            # Merge messages into the existing state
            new_state = dict(request.state)
            new_state["messages"] = updated_messages
            
            # Override the request with updated state
            modified_request = request.override(state=new_state)
            return await handler(modified_request)
        else:
            # First call or no tool responses yet
            if self.verbose:
                console.print("[bold cyan][FIRST_CALL] No tool responses yet, using base system prompt[/bold cyan]")
            
            # Ensure system message is in messages
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
                modified_request = request.override(state={"messages": messages})
                return await handler(modified_request)

            return await handler(request)

    def _update_system_message(self, messages: List, new_prompt: str) -> List:
        """Update or insert system message with new prompt."""
        updated_messages = []
        system_found = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                updated_messages.append(SystemMessage(content=new_prompt))
                system_found = True
            else:
                updated_messages.append(msg)

        if not system_found:
            updated_messages.insert(0, SystemMessage(content=new_prompt))

        return updated_messages
