# 🤖 AI Orchestrator Agent

A modular AI Orchestrator Agent built with **LangChain**, **LangGraph**, **Playwright**, and **FastAPI**, supporting Anthropic Claude and OpenAI backends.

## Features

- Dual LLM support through a provider abstraction
- Playwright browser automation
- Terminal command tool with timeout and output limits
- LangGraph ReAct agent with bounded in-memory conversation history
- FastAPI endpoints: `/run`, `/health`, and `/history`
- Retry handling and logging

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Add the provider API key to your environment or .env loader.
python main.py
```

The application currently reads environment variables directly; the `.env.example` file documents expected values but does not automatically load them. Export variables in your shell or configure your deployment environment.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Minimal health check |
| POST | `/run` | Run an agent prompt |
| GET | `/history` | Read provider conversation history |
| DELETE | `/history` | Clear provider conversation history |

Example:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ORCHESTRATOR_API_KEY" \
  -d '{"prompt":"Search for recent AI agent research","provider":"anthropic"}'
```

## Security configuration

- Set `APP_ENV=production` and a strong random `ORCHESTRATOR_API_KEY` before exposing the service. In production, the service fails closed if the key is missing.
- Send the key using the `X-API-Key` header to `/run` and `/history`.
- Configure `CORS_ALLOWED_ORIGINS` as a comma-separated list of exact trusted browser origins. The default only permits localhost development origins.
- `/health` is intentionally public and returns only a minimal status and timestamp.
- `MAX_PROMPT_LENGTH` limits prompt size (default 12,000 characters).
- Do not log or commit provider keys. The `.env.example` file contains placeholders only.
- API-key authentication does **not** sandbox the terminal tool. Do not expose this service to untrusted networks until terminal execution and browser access have been independently reviewed and isolated. A command blocklist alone is not a security boundary.

For local development, `APP_ENV` defaults to `development`, where API-key authentication is optional. This mode is not suitable for a publicly accessible deployment.

## Tests

```bash
pytest -q
```

The repository includes the original `test_agent.py` suite and focused API security regression tests under `tests/`. Run the suite in a configured development environment; tests have not been executed as part of this repository edit.

## Technology

- Python
- LangChain and LangGraph
- FastAPI and Uvicorn
- Playwright / Chromium
- Anthropic and OpenAI SDK integrations

## License

MIT — NeonTechno / Decentralized Rights Protocol.
