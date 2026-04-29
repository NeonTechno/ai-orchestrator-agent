"""Browser Tool — headless Chromium automation via Playwright."""
import asyncio
import logging
from urllib.parse import quote_plus

import nest_asyncio
from playwright.async_api import async_playwright

# Allow asyncio.run() inside an already-running event loop (e.g. FastAPI / Jupyter)
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# Search engines tried in order until one succeeds (bot-detection avoidance)
_SEARCH_ENGINES = [
    "https://html.duckduckgo.com/html/?q={q}",   # DDG HTML — no JS, low bot-detection
    "https://www.bing.com/search?q={q}",
]

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_SNIPPET_LIMIT = 1000  # characters of visible page text returned to the agent
_TIMEOUT_MS = 45_000


async def _fetch_page(url: str) -> dict:
    """Core Playwright fetch. Opens a fresh browser + context per call."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=_TIMEOUT_MS)
            await page.wait_for_timeout(1200)
            title = await page.title()
            current_url = page.url
            text = await page.evaluate("() => document.body.innerText")
            snippet = (text or "").strip()[:_SNIPPET_LIMIT]
            logger.info(f"[Browser] OK  title={title[:60]!r}  chars={len(snippet)}")
            return {"success": True, "title": title, "url": current_url, "snippet": snippet}
        except Exception as exc:
            logger.error(f"[Browser] Error fetching {url!r}: {exc}")
            return {"success": False, "error": str(exc)}
        finally:
            # Always close context AND browser to avoid resource leaks
            await context.close()
            await browser.close()


async def _browser_search_async(query: str) -> dict:
    """
    Async entry point.
    - If query looks like a URL, fetch it directly.
    - Otherwise try each search engine in _SEARCH_ENGINES until one works.
    """
    if query.startswith(("http://", "https://")):
        return await _fetch_page(query)

    encoded = quote_plus(query)
    last_error = "no engines tried"
    for template in _SEARCH_ENGINES:
        url = template.format(q=encoded)
        logger.info(f"[Browser] Searching: {url}")
        result = await _fetch_page(url)
        if result.get("success"):
            return result
        last_error = result.get("error", "unknown")
        logger.warning(f"[Browser] Engine failed ({url}): {last_error}")

    return {"success": False, "error": f"All search engines failed. Last: {last_error}"}


def run_browser_search(query: str) -> dict:
    """
    Synchronous wrapper — safe to call from sync code AND from within a running
    event loop (nest_asyncio handles the nesting).
    """
    return asyncio.run(_browser_search_async(query))
