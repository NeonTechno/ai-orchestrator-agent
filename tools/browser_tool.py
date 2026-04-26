"""Browser Tool using Playwright - headless Chromium automation"""
import asyncio
import logging
from urllib.parse import quote_plus
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


async def browser_search(query: str) -> dict:
    """
    Navigate to a URL or perform a web search.
    Direct URLs (starting with http/https) are loaded as-is.
    Text queries are searched via Bing (good bot tolerance in headless mode).
    """
    if query.startswith("http://") or query.startswith("https://"):
        target_url = query
    else:
        target_url = f"https://www.bing.com/search?q={quote_plus(query)}"

    logger.info(f"[Browser] Navigating to: {target_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1500)
            title = await page.title()
            current_url = page.url
            text = await page.evaluate("() => document.body.innerText")
            snippet = text[:800].strip() if text else ""
            logger.info(f"[Browser] Loaded: '{title[:50]}' | {len(snippet)} chars")
            return {
                "success": True,
                "title": title,
                "url": current_url,
                "snippet": snippet,
            }
        except Exception as e:
            logger.error(f"[Browser] Error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            await browser.close()


def run_browser_search(query: str) -> dict:
    """Sync wrapper for async browser_search."""
    return asyncio.run(browser_search(query))
