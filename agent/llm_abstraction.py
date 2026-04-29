"""LLM Abstraction Layer — factory for Anthropic and OpenAI chat models."""
import logging
import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Map provider name → (env-var name, default model)
_PROVIDER_CONFIG: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-haiku-20241022"),
    "openai":    ("OPENAI_API_KEY",    "gpt-4o-mini"),
}


def get_llm(
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.

    Raises:
        ValueError: unknown provider name.
        EnvironmentError: required API key env-var is not set.
    """
    if provider not in _PROVIDER_CONFIG:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Valid choices: {list(_PROVIDER_CONFIG)}"
        )

    env_var, default_model = _PROVIDER_CONFIG[provider]
    if not os.environ.get(env_var):
        raise EnvironmentError(
            f"Provider {provider!r} requires the {env_var!r} environment variable to be set."
        )

    chosen_model = model or default_model
    logger.info(f"[LLM] provider={provider!r}  model={chosen_model!r}")

    if provider == "openai":
        return ChatOpenAI(model=chosen_model, temperature=temperature, max_tokens=max_tokens)

    # provider == "anthropic"
    return ChatAnthropic(model=chosen_model, temperature=temperature, max_tokens=max_tokens)
