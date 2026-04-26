# 🤖 AI Orchestrator Agent

A production-ready, modular AI Orchestrator Agent built with **LangChain**, **LangGraph**, **Playwright**, and **FastAPI** — supporting both **Anthropic Claude** and **OpenAI GPT** backends.

Built autonomously using [Composio](https://composio.dev) tools.

---

## ✨ Features

- **Dual LLM Support** — Anthropic Claude (default) or OpenAI GPT via a clean abstraction layer
- **Browser Automation** — Playwright headless Chromium for real web search & navigation
- **Terminal Tool** — Safe subprocess execution with timeout and blocklist
- **LangGraph ReAct Agent** — Modern agentic loop with tool use and memory
- **FastAPI Server** — REST API with `/run`, `/health`, `/history` endpoints
- **Conversation Memory** — Maintains context across multiple requests
- **Retry Logic** — Exponential back-off, up to 3 attempts per request
- **Safety Filters** — Blocks dangerous shell commands and prompt injections
- **Full Logging** — Structured logs to file and stdout

---

## 📁 Project Structure

```
ai-orchestrator-agent/
├── agent/
│   ├── llm_abstraction.py    # OpenAI + Anthropic LLM factory
│   └── orchestrator.py       # LangGraph ReAct agent with memory + retry
├── tools/
│   ├── browser_tool.py       # Playwright headless browser
│   └── terminal_tool.py      # Safe shell command executor
├── api/
│   └── server.py             # FastAPI REST server
├── main.py                   # Entry point
├── test_agent.py             # 20-test suite (all passing)
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."       # optional, OpenAI fallback
```

### 3. Start the server

```bash
python main.py
# or
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 4. Run the agent

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Open Google and search for AI agents"}'
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/run` | Run agent with a prompt |
| `GET` | `/history` | Get conversation history |
| `DELETE` | `/history` | Clear conversation history |

### Example `/run` request

```json
{
  "prompt": "Search for the latest news on AI agents and summarize",
  "provider": "anthropic"
}
```

### Example response

```json
{
  "success": true,
  "output": "Here are the latest developments in AI agents...",
  "steps": 3,
  "attempt": 1,
  "elapsed_seconds": 8.4
}
```

---

## 🛡️ Safety

- Dangerous shell commands (`rm -rf /`, etc.) are blocked by the terminal tool
- Malicious prompt patterns blocked at orchestrator level
- All commands run with configurable timeouts

---

## 🧪 Tests

```bash
python test_agent.py
```

**20/20 tests passing** — covers terminal tool, browser tool, API endpoints, LLM abstraction, and safety filters.

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Backend | Anthropic Claude / OpenAI GPT |
| Agent Framework | LangChain + LangGraph |
| Browser Automation | Playwright (Chromium) |
| API Server | FastAPI + Uvicorn |
| Orchestration | Composio |

---

## 📄 License

MIT — built by [NeonTechno](https://github.com/NeonTechno) / Decentralized Rights Protocol
