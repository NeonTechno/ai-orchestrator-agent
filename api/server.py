"""FastAPI server for the AI Orchestrator Agent."""
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, field_validator

from agent.orchestrator import OrchestratorAgent

# Keep logs local and avoid logging full prompts or credentials.
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("logs/orchestrator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
API_KEY = os.getenv("ORCHESTRATOR_API_KEY", "").strip()
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
VALID_PROVIDERS = ("anthropic", "openai")

app = FastAPI(
    title="AI Orchestrator Agent",
    description="LangGraph ReAct agent with browser + terminal tools (Anthropic / OpenAI)",
    version="1.2.0",
)

# Wildcard origins + credentials are intentionally avoided.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_agent_cache: dict[str, OrchestratorAgent] = {}


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Require a constant-time API-key comparison; fail closed in production."""
    if not API_KEY:
        if APP_ENV == "production":
            logger.error("ORCHESTRATOR_API_KEY is not configured in production")
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _validate_provider(provider: str) -> str:
    if provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=422, detail="provider must be 'anthropic' or 'openai'")
    return provider


def get_agent(provider: str = "anthropic") -> OrchestratorAgent:
    """Return a cached OrchestratorAgent for a supported provider."""
    _validate_provider(provider)
    if provider not in _agent_cache:
        logger.info("[Server] Initialising agent for provider=%s", provider)
        _agent_cache[provider] = OrchestratorAgent(provider=provider)
    return _agent_cache[provider]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    provider: Optional[str] = "anthropic"

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        if len(value) > int(os.getenv("MAX_PROMPT_LENGTH", "12000")):
            raise ValueError("prompt exceeds the configured maximum length")
        return value

    @field_validator("provider")
    @classmethod
    def provider_must_be_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VALID_PROVIDERS:
            raise ValueError("provider must be 'anthropic' or 'openai'")
        return value


class RunResponse(BaseModel):
    success: bool
    output: str
    steps: int = 0
    attempt: int = 1
    elapsed_seconds: float = 0.0
    provider: str = "anthropic"


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI Orchestrator Agent",
        "version": "1.2.0",
        "endpoints": ["/run", "/health", "/history", "/history (DELETE)"],
        "authentication": "api-key-required" if API_KEY or APP_ENV == "production" else "development-mode",
    }


@app.get("/health")
def health():
    # Deliberately minimal: do not expose provider configuration or secrets.
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/run", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def run_agent(req: RunRequest):
    provider = req.provider or "anthropic"
    logger.info("[API] /run provider=%s prompt_length=%d", provider, len(req.prompt))
    started = time.monotonic()
    try:
        result = get_agent(provider).run(req.prompt)
    except HTTPException:
        raise
    except Exception:
        logger.exception("[API] Agent execution failed")
        raise HTTPException(status_code=500, detail="Agent execution failed; check server logs")
    return RunResponse(
        success=result["success"],
        output=result["output"],
        steps=result.get("steps", 0),
        attempt=result.get("attempt", 1),
        elapsed_seconds=round(time.monotonic() - started, 2),
        provider=provider,
    )


@app.get("/history", dependencies=[Depends(require_api_key)])
def get_history(provider: str = "anthropic"):
    provider = _validate_provider(provider)
    return {"provider": provider, "history": get_agent(provider).get_history()}


@app.delete("/history", dependencies=[Depends(require_api_key)])
def clear_history(provider: str = "anthropic"):
    provider = _validate_provider(provider)
    get_agent(provider).clear_history()
    logger.info("[API] History cleared for provider=%s", provider)
    return {"message": f"Conversation history cleared for provider={provider}"}
