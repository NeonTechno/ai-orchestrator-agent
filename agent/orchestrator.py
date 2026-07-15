url: https://raw.githubusercontent.com/NeonTechno/ai-orchestrator-agent/main/agent/orchestrator.py

"""
AI Orchestrator Agent — LangGraph ReAct agent.

Fixes vs v1.0:
- AIMessage.content guarded for list payloads (tool-use content blocks)
- Tool step count uses isinstance(m, ToolMessage) instead of fragile .type check
- Safety blocklist uses compiled regex for robust, case-insensitive matching
- History window documented and enforced
"""
import logging
import re
import time
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent.llm_abstraction import get_llm
from tools.browser_tool import run_browser_search
from tools.terminal_tool import run_command

logger = logging.getLogger(__name__)


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def browser(query: str) -> str:
    """
    Open a web browser and search the web or visit a URL.
    Pass a plain search query (e.g. "AI agents 2025") or a full URL starting with http/https.
    Returns: page title, current URL, and a text snippet from the page.
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
    Examples: 'echo hello', 'ls -la', 'python3 --version', 'cat /etc/os-release'.
    Returns: stdout, stderr, and exit code.
    Dangerous commands (rm -rf /, mkfs, fork-bomb, etc.) are automatically blocked.
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

# Compiled once at module load — matched against normalised (lowercase) prompt
_SAFETY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"drop\s+table",
    r"delete\s+from",
    r"format\s+c:",
    r"shutdown\s+-[hrp]",
    r"rm\s+-rf\s+/",
    r"\bmkfs\b",
]]

# History window: keep at most this many messages to bound memory usage
_HISTORY_WINDOW = 40


def _extract_text(content) -> str:
    """
    Safely extract a plain-text string from an AIMessage content value.
    content may be: str | list[str | dict]  (LangChain content-block format)
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return str(content)


class OrchestratorAgent:
    """Stateful ReAct agent with conversation memory, retry, and safety filters."""

    def __init__(self, provider: str = "anthropic", max_retries: int = 3):
        self.provider = provider
        self.max_retries = max_retries
        self.history: list = []  # LangChain message objects, capped at _HISTORY_WINDOW
        self._init_agent()

    def _init_agent(self):
        llm = get_llm(self.provider)
        self.graph = create_react_agent(llm, TOOLS)
        logger.info(f"[Orchestrator] Agent ready (provider={self.provider!r})")

    def _is_safe(self, prompt: str) -> Optional[str]:
        """Return a block-reason string if the prompt matches a safety pattern, else None."""
        for pattern in _SAFETY_PATTERNS:
            if pattern.search(prompt):
                return f"Prompt blocked by safety filter (matched: {pattern.pattern!r})"
        return None

    def run(self, prompt: str) -> dict:
        """
        Run the agent with e
xponential back-off retry.

        Returns:
            dict with keys: success (bool), output (str), steps (int), attempt (int)
        """
        blocked = self._is_safe(prompt)
        if blocked:
            return {"success": False, "output": blocked, "steps": 0, "attempt": 0}

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[Orchestrator] Attempt {attempt}/{self.max_retries}: {prompt[:80]!r}"
                )
                messages = self.history + [HumanMessage(content=prompt)]
                result = self.graph.invoke({"messages": messages})

                all_messages = result["messages"]

                # Extract final AI response — guard against list content blocks
                ai_msgs = [m for m in all_messages if isinstance(m, AIMessage)]
                raw_output = ai_msgs[-1].content if ai_msgs else ""
                output = _extract_text(raw_output) or "No response generated."

                # Count tool invocations using ToolMessage (robust across LangGraph versions)
                tool_steps = sum(1 for m in all_messages if isinstance(m, ToolMessage))

                # Trim history to keep memory bounded
                self.history = all_messages[-_HISTORY_WINDOW:]

                logger.info(f"[Orchestrator] Done — {tool_steps} tool call(s)")
                return {"success": True, "output": output, "steps": tool_steps, "attempt": attempt}

            except Exception as exc:
                last_error = str(exc)
                logger.error(f"[Orchestrator] Attempt {attempt} failed: {exc}")
                if attempt < self.max_retries:
                    sleep_s = 2 ** attempt
                    logger.info(f"[Orchestrator] Retrying in {sleep_s}s...")
                    time.sleep(sleep_s)

        return {
            "success": False,
            "output": f"All {self.max_retries} attempts failed. Last e
rror: {last_error}",
            "steps": 0,
            "attempt": self.max_retries,
        }

    def get_history(self) -> list:
        """Return conversation history as serialisable dicts."""
        result = []
        for m in self.history:
            content = _extract_text(m.content)
            result.append({"role": m.__class__.__name__, "content": content})
        return result

    def clear_history(self) -> None:
        self.history = []
        logger.info("[Orchestrator] History cleared.")
