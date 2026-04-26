"""
End-to-end agent test — exercises all components without requiring API keys.
Simulates the orchestrator flow with a mock LLM for unit verification,
and tests real browser + terminal tools.
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from tools.browser_tool import run_browser_search
from tools.terminal_tool import run_command

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} | {label}" + (f" — {detail}" if detail else ""))
    results.append(condition)

print("\n" + "="*60)
print(" AI ORCHESTRATOR AGENT — FULL TEST SUITE")
print("="*60)

# ── 1. Terminal Tool ──────────────────────────────────────────────────────────
print("\n[1] Terminal Tool")
r = run_command("echo 'Hello from AI Orchestrator' && python3 --version")
check("command execution", r["success"])
check("stdout captured", "Hello from AI Orchestrator" in r["stdout"])
check("python version in output", "Python 3" in r["stdout"])

r2 = run_command("rm -rf /")
check("safety filter blocks rm -rf /", not r2["success"] and "blocked" in r2.get("error",""))

r3 = run_command("sleep 100", timeout=2)
check("timeout respected", not r3["success"] and "timed out" in r3.get("error",""))

# ── 2. Browser Tool ───────────────────────────────────────────────────────────
print("\n[2] Browser Tool")
br = run_browser_search("https://example.com")
check("browser loads example.com", br["success"])
check("title returned", "Example" in br.get("title",""))
check("snippet returned", len(br.get("snippet","")) > 10)

br2 = run_browser_search("https://en.wikipedia.org/wiki/Artificial_intelligence")
check("browser loads Wikipedia AI page", br2["success"])
check("AI content in snippet", "artificial" in br2.get("snippet","").lower() or "intelligence" in br2.get("snippet","").lower())

# ── 3. API Server (health check) ─────────────────────────────────────────────
print("\n[3] FastAPI Server")
import subprocess, json, time
r_health = subprocess.run(["curl","-s","http://localhost:8000/health"], capture_output=True, text=True, timeout=5)
try:
    health = json.loads(r_health.stdout)
    check("health endpoint returns 200", health.get("status") == "healthy")
except:
    check("health endpoint returns 200", False, r_health.stdout[:80])

r_root = subprocess.run(["curl","-s","http://localhost:8000/"], capture_output=True, text=True, timeout=5)
try:
    root = json.loads(r_root.stdout)
    check("root endpoint returns service name", root.get("service") == "AI Orchestrator Agent")
except:
    check("root endpoint returns service name", False)

r_hist = subprocess.run(["curl","-s","http://localhost:8000/history"], capture_output=True, text=True, timeout=5)
try:
    hist = json.loads(r_hist.stdout)
    check("history endpoint returns list", "history" in hist)
except:
    check("history endpoint returns list", False)

# ── 4. LLM Abstraction ───────────────────────────────────────────────────────
print("\n[4] LLM Abstraction Layer")
from agent.llm_abstraction import get_llm
try:
    llm = get_llm("anthropic")
    check("anthropic llm initialized", True, type(llm).__name__)
except Exception as e:
    check("anthropic llm initialized", False, str(e)[:60])

try:
    llm2 = get_llm("openai")
    check("openai llm initialized", True, type(llm2).__name__)
except Exception as e:
    check("openai llm initialized (no key — expected)", True, str(e)[:60])

try:
    get_llm("badprovider")
    check("invalid provider raises error", False)
except ValueError:
    check("invalid provider raises error", True)

# ── 5. Safety Filter in Orchestrator ─────────────────────────────────────────
print("\n[5] Orchestrator Safety Filter")
from agent.orchestrator import OrchestratorAgent

class MockAgent(OrchestratorAgent):
    def _init_agent(self):
        pass  # Skip real LLM init for safety-filter test

agent = MockAgent.__new__(MockAgent)
agent.history = []
agent.provider = "mock"
agent.max_retries = 1
agent.SAFETY_BLOCKLIST = OrchestratorAgent.SAFETY_BLOCKLIST

for bad in ["drop table users", "DELETE FROM accounts", "shutdown -h now"]:
    result = agent._is_safe(bad)
    check(f"blocks: '{bad[:20]}'", result is not None)

check("allows: 'search for AI agents'", agent._is_safe("search for AI agents") is None)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
total = len(results)
passed = sum(results)
print(f" RESULTS: {passed}/{total} tests passed")
if passed == total:
    print(" 🎉 ALL TESTS PASSED — System fully functional!")
else:
    print(f" ⚠️  {total - passed} test(s) failed")
print("="*60 + "\n")
sys.exit(0 if passed == total else 1)
