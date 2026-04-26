"""
AI Orchestrator Agent (LangGraph-based ReAct, compatible with langchain v1.x)
- LangGraph create_react_agent (replaces deprecated langchain AgentExecutor)
- Browser + Terminal tools
- In-memory conversation history
- Retry logic + safety filters
"""
import logging
import time
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from agent.llm_abstraction import get_llm
from tools.browser_tool import run_browser_search
from tools.terminal_tool import run_command

logger = logging.getLogger(__name__)


# ── LangChain Tools ──────────────────────────────────────────────────────────

@tool
def browser(query: str) -> str:
    """
    Open a web browser and search Google or visit a URL.
    Input: a search query string or a full URL (must start with http).
    Returns: page title, current URL, and a text snippet.
    """
    result = run_browser_search(query)
    if result.get("success"):
        return (
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Snippet: {result['snippet']}"
        )
    return f"Browser error: {result.get('error', 'unknown')}"


@tool
def terminal(command: str) -> str:
    """
    Run a safe shell command on the local machine.
    Input: a shell command string (e.g. 'echo hello', 'ls -la', 'python3 --version').
    Returns: stdout, stderr, and exit code.
    """
    result = run_command(command)
    if not result.get("success") and "error" in result:
        return f"Error: {result['error']}"
    return (
        f"stdout: {result.get('stdout', '')}\n"
        f"stderr: {result.get('stderr', '')}\n"
        f"exit code: {result.get('returncode', '?')}"
    )


TOOLS = [browser, terminal]


# ── Orchestrator ─────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Stateful agent with memory, retry logic, and safety filters."""

    SAFETY_BLOCKLIST = ["drop table", "delete from", "format c:", "shutdown -h", "rm -rf /"]

    def __init__(self, provider: str = "anthropic", max_retries: int = 3):
        self.provider = provider
        self.max_retries = max_retries
        self.history: list = []   # list of LangChain message objects
        self._init_agent()

    def _init_agent(self):
        llm = get_llm(self.provider)
        self.graph = create_react_agent(llm, TOOLS)
        logger.info(f"[Orchestrator] Agent ready (provider={self.provider})")

    def _is_safe(self, prompt: str) -> Optional[str]:
        for blocked in self.SAFETY_BLOCKLIST:
            if blocked.lower() in prompt.lower():
                return f"Prompt blocked by safety filter: '{blocked}'"
        return None

    def run(self, prompt: str) -> dict:
        """Execute the agent with retry and exponential back-off."""
        blocked_reason = self._is_safe(prompt)
        if blocked_reason:
            return {"success": False, "output": blocked_reason}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[Orchestrator] Attempt {attempt}/{self.max_retries}: {prompt[:80]}")
                messages = self.history + [HumanMessage(content=prompt)]
                result = self.graph.invoke({"messages": messages})

                all_messages = result["messages"]
                # Last AI message is the final answer
                ai_msgs = [m for m in all_messages if isinstance(m, AIMessage)]
                output = ai_msgs[-1].content if ai_msgs else "No response generated."
                tool_steps = sum(1 for m in all_messages if hasattr(m, "type") and m.type == "tool")

                # Update memory (keep last 20 message pairs)
                self.history = all_messages[-40:]

                logger.info(f"[Orchestrator] Done. Steps: {tool_steps}")
                return {
                    "success": True,
                    "output": output,
                    "steps": tool_steps,
                    "attempt": attempt,
                }
            except Exception as e:
                last_error = str(e)
                logger.error(f"[Orchestrator] Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        return {
            "success": False,
            "output": f"All {self.max_retries} attempts failed. Last error: {last_error}",
        }

    def get_history(self) -> list:
        return [
            {"role": m.__class__.__name__, "content": m.content}
            for m in self.history
        ]

    def clear_history(self):
        self.history = []
