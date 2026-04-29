"""FastAPI server for the AI Orchestrator Agent."""
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from agent.orchestrator import OrchestratorAgent

# ── Ensure logs directory exists BEFORE configuring file handler ──────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Orchestrator Agent",
    description="LangGraph ReAct agent with browser + terminal tools (Anthropic / OpenAI)",
    version="1.1.0",
)

# CORS — allow all origins so browser clients can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agent cache (one instance per provider, initialised lazily) ───────────────
_agent_cache: dict[str, OrchestratorAgent] = {}


def get_agent(provider: str = "anthropic") -> OrchestratorAgent:
    """Return a cached OrchestratorAgent for the given provider."""
    if provider not in _agent_cache:
        logger.info(f"[Server] Initialising agent for provider={provider!r}")
        _agent_cache[provider] = OrchestratorAgent(provider=provider)
    return _agent_cache[provider]


# ── Pydantic models ───────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "anthropic"

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v.strip()

    @field_validator("provider")
    @classmethod
    def provider_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("anthropic", "openai"):
            raise ValueError("provider must be 'anthropic' or 'openai'")
        return v


class RunResponse(BaseModel):
    success: bool
    output: str
    steps: int = 0
    attempt: int = 1
    elapsed_seconds: float = 0.0
    provider: str = "anthropic"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI Orchestrator Agent",
        "version": "1.1.0",
        "endpoints": ["/run", "/health", "/history", "/history (DELETE)"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/run", response_model=RunResponse)
def run_agent(req: RunRequest):
    provider = req.provider or "anthropic"
    logger.info(f"[API] /run  provider={provider}  prompt={req.prompt[:120]!r}")
    t0 = time.time()
    try:
        agent = get_agent(provider)
        result = agent.run(req.prompt)
    except Exception as exc:
        logger.error(f"[API] /run error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed = round(time.time() - t0, 2)
    return RunResponse(
        success=result["success"],
        output=result["output"],
        steps=result.get("steps", 0),
        attempt=result.get("attempt", 1),
        elapsed_seconds=elapsed,
        provider=provider,
    )


@app.get("/history")
def get_history(provider: str = "anthropic"):
    agent = get_agent(provider)
    return {"provider": provider, "history": agent.get_history()}


@app.delete("/history")
def clear_history(provider: str = "anthropic"):
    agent = get_agent(provider)
    agent.clear_history()
    logger.info(f"[API] History cleared for provider={provider!r}")
    return {"message": f"Conversation history cleared for provider={provider!r}"}
