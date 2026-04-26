"""LLM Abstraction Layer - supports OpenAI and Anthropic via LangChain"""
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

def get_llm(
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> BaseChatModel:
    """Factory: returns a LangChain chat model for the given provider."""
    if provider == "openai":
        model = model or "gpt-4o-mini"
        logger.info(f"[LLM] Using OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "anthropic":
        model = model or "claude-3-5-haiku-20241022"
        logger.info(f"[LLM] Using Anthropic model: {model}")
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'anthropic'.")
