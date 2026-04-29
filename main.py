"""Entry point — starts the AI Orchestrator Agent FastAPI server.

Environment variables (all optional):
  HOST      bind address   (default: 0.0.0.0)
  PORT      listen port    (default: 8000)
  RELOAD    auto-reload    (default: false)
  LOG_LEVEL uvicorn level  (default: info)
"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
