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


# LangChain Tools

@tool
def browser(query: str) -> str:
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
    result = run_command(command)
    if not result.get("success") and "error" in result:
        return f"Error: {result['error']}"
    return (
        f"stdout: {result.get('stdout', '')}\n"
        f"stderr: {result.get('stderr', '')}\n"
        f"exit code: {result.get('returncode', '?')}"
    )