"""
AI Orchestrator Agent — pytest test suite.

Run with:
    pytest test_agent.py -v

Tests are split into marks:
  - unit        : no network, no server needed
  - integration : hits live internet (browser tests)
  - server      : requires a running FastAPI server on localhost:8000

Skip slow/external tests:
    pytest test_agent.py -v -m "not integration and not server"
"""
import json
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Terminal Tool — unit tests (no network, no server)
# ---------------------------------------------------------------------------

class TestTerminalTool:
    def setup_method(self):
        # Import here so pytest collection doesn't fail if deps are missing
        from tools.terminal_tool import run_command, is_safe_command
        self.run_command = run_command
        self.is_safe_command = is_safe_command

    def test_basic_execution(self):
        r = self.run_command("echo hello")
        assert r["success"], f"Expected success, got: {r}"
        assert "hello" in r["stdout"]

    def test_stdout_and_version(self):
        r = self.run_command("python3 --version")
        assert r["success"]
        assert "Python 3" in r["stdout"] or "Python 3" in r["stderr"]

    def test_safety_blocks_rm_rf(self):
        r = self.run_command("rm -rf /")
        assert not r["success"]
        assert "blocked" in r.get("error", "").lower()

    def test_safety_blocks_uppercase(self):
        """Blocklist must be case-insensitive."""
        safe, reason = self.is_safe_command("RM -RF /")
        assert not safe

    def test_safety_blocks_extra_spaces(self):
        """Blocklist must survive collapsed whitespace."""
        safe, reason = self.is_safe_command("rm  -rf  /")
        assert not safe

    def test_timeout_respected(self):
        r = self.run_command("sleep 100", timeout=2)
        assert not r["success"]
        assert "timed out" in r.get("error", "").lower()

    def test_nonzero_exit_code(self):
        r = self.run_command("false")   # always exits 1
        assert not r["success"]
        assert r["returncode"] == 1

    def test_invalid_command(self):
        r = self.run_command("this_command_does_not_exist_xyz")
        assert not r["success"]

    def test_stdout_limit(self):
        """stdout should be capped at OUTPUT_LIMIT chars."""
        from tools.terminal_tool import OUTPUT_LIMIT
        r = self.run_command(f"python3 -c \"print('x' * {OUTPUT_LIMIT * 2})\"")
        assert len(r.get("stdout", "")) <= OUTPUT_LIMIT


# ---------------------------------------------------------------------------
# Browser Tool — integration tests (live internet)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBrowserTool:
    def setup_method(self):
        from tools.browser_tool import run_browser_search
        self.run_browser_search = run_browser_search

    def test_loads_direct_url(self):
        r = self.run_browser_search("https://example.com")
        assert r["success"], f"Browser failed: {r.get('error')}"
        assert "Example" in r.get("title", "")
        assert len(r.get("snippet", "")) > 10

    def test_search_query(self):
        r = self.run_browser_search("AI agents Wikipedia")
        assert r["success"], f"Search failed: {r.get('error')}"
        assert r.get("snippet"), "Expected non-empty snippet"

    def test_loads_wikipedia(self):
        r = self.run_browser_search("https://en.wikipedia.org/wiki/Artificial_intelligence")
        assert r["success"]
        snippet_lower = r.get("snippet", "").lower()
        assert "intelligence" in snippet_lower or "artificial" in snippet_lower


# ---------------------------------------------------------------------------
# LLM Abstraction — unit tests (no API calls made)
# ---------------------------------------------------------------------------

class TestLLMAbstraction:
    def setup_method(self):
        from agent.llm_abstraction import get_llm
        self.get_llm = get_llm

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            self.get_llm("badprovider")

    def test_missing_anthropic_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            self.get_llm("anthropic")

    def test_missing_openai_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            self.get_llm("openai")

    def test_anthropic_object_created(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from langchain_anthropic import ChatAnthropic
        llm = self.get_llm("anthropic")
        assert isinstance(llm, ChatAnthropic)

    def test_openai_object_created(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from langchain_openai import ChatOpenAI
        llm = self.get_llm("openai")
        assert isinstance(llm, ChatOpenAI)


# ---------------------------------------------------------------------------
# Orchestrator Safety Filter — unit tests (no LLM initialised)
# ---------------------------------------------------------------------------

class TestOrchestratorSafety:
    """Tests _is_safe() directly without spinning up an LLM."""

    def _make_agent(self):
        from agent.orchestrator import OrchestratorAgent, _SAFETY_PATTERNS
        class _Mock(OrchestratorAgent):
            def _init_agent(self): pass  # skip LLM init
        obj = _Mock.__new__(_Mock)
        obj.history = []
        obj.provider = "mock"
        obj.max_retries = 1
        return obj

    def test_blocks_drop_table(self):
        agent = self._make_agent()
        assert agent._is_safe("DROP TABLE users") is not None

    def test_blocks_delete_from(self):
        agent = self._make_agent()
        assert agent._is_safe("DELETE FROM accounts") is not None

    def test_blocks_shutdown(self):
        agent = self._make_agent()
        assert agent._is_safe("shutdown -h now") is not None

    def test_blocks_rm_rf(self):
        agent = self._make_agent()
        assert agent._is_safe("please run rm -rf /") is not None

    def test_allows_normal_prompt(self):
        agent = self._make_agent()
        assert agent._is_safe("Search for the latest AI news") is None

    def test_allows_code_question(self):
        agent = self._make_agent()
        assert agent._is_safe("write a python function to sort a list") is None


# ---------------------------------------------------------------------------
# FastAPI Server — integration tests (requires running server)
# ---------------------------------------------------------------------------

def _server_running() -> bool:
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "2", "http://localhost:8000/health"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(result.stdout)
        return data.get("status") == "healthy"
    except Exception:
        return False


@pytest.mark.server
@pytest.mark.skipif(not _server_running(), reason="FastAPI server not running on localhost:8000")
class TestAPIServer:
    def _get(self, path: str) -> dict:
        r = subprocess.run(
            ["curl", "-s", f"http://localhost:8000{path}"],
            capture_output=True, text=True, timeout=10,
        )
        return json.loads(r.stdout)

    def test_root_endpoint(self):
        data = self._get("/")
        assert data["service"] == "AI Orchestrator Agent"

    def test_health_endpoint(self):
        data = self._get("/health")
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_history_endpoint(self):
        data = self._get("/history")
        assert "history" in data

    def test_run_rejects_empty_prompt(self):
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:8000/run",
             "-H", "Content-Type: application/json",
             "-d", '{"prompt": ""}'],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        # Pydantic validation should return 422
        assert "detail" in data
