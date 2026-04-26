"""FastAPI server for the AI Orchestrator Agent"""
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from agent.orchestrator import OrchestratorAgent

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── App + Agent ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Orchestrator Agent",
    description="LangChain agent with browser + terminal tools (OpenAI / Anthropic)",
    version="1.0.0",
)

# Lazy-init agent
_agent: Optional[OrchestratorAgent] = None

def get_agent() -> OrchestratorAgent:
    global _agent
    if _agent is None:
        provider = "anthropic"
        try:
            _agent = OrchestratorAgent(provider=provider)
        except Exception as e:
            logger.error(f"Anthropic init failed ({e}), falling back to OpenAI")
            _agent = OrchestratorAgent(provider="openai")
    return _agent


# ── Request / Response Models ─────────────────────────────────────────────────
class RunRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None  # override provider per-request

class RunResponse(BaseModel):
    success: bool
    output: str
    steps: int = 0
    attempt: int = 1
    elapsed_seconds: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "service": "AI Orchestrator Agent", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/run", response_model=RunResponse)
def run_agent(req: RunRequest):
    logger.info(f"[API] /run called: {req.prompt[:100]}")
    t0 = time.time()
    agent = get_agent()
    # Per-request provider override
    if req.provider and req.provider != agent.provider:
        try:
            tmp = OrchestratorAgent(provider=req.provider)
            result = tmp.run(req.prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        result = agent.run(req.prompt)
    elapsed = round(time.time() - t0, 2)
    return RunResponse(
        success=result["success"],
        output=result["output"],
        steps=result.get("steps", 0),
        attempt=result.get("attempt", 1),
        elapsed_seconds=elapsed,
    )

@app.get("/history")
def get_history():
    agent = get_agent()
    return {"history": agent.get_history()}

@app.delete("/history")
def clear_history():
    agent = get_agent()
    agent.clear_history()
    return {"message": "Conversation history cleared"}
