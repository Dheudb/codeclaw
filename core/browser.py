import os
import time

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None


class BrowserSessionManager:
    """
    Lightweight single-page browser session manager backed by Playwright.
    """

    def __init__(self, artifact_dir=".codeclaw/browser"):
        self.artifact_dir = artifact_dir
        self._playwright = None
        self._browser = None
        self._page = None
        self._context = None
        self._last_url = None

    async def ensure_page(self):
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Install with `pip install playwright` "
                "and then run `playwright install`."
            )

        if self._page is not None:
            return self._page

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(viewport={"width": 1440, "height": 960})
        self._page = await self._context.new_page()
        return self._page

    async def navigate(self, url: str, wait_until: str = "domcontentloaded"):
        page = await self.ensure_page()
        response = await page.goto(url, wait_until=wait_until, timeout=30000)
        self._last_url = page.url
        return {
            "url": page.url,
            "status": response.status if response else "unknown",
            "title": await page.title(),
        }

    async def snapshot(self):
        page = await self.ensure_page()
        title = await page.title()
        url = page.url
        text_preview = await page.evaluate(
            """
            () => {
              const text = (document.body?.innerText || '').trim().replace(/\\s+/g, ' ');
              return text.slice(0, 2500);
            }
            """
        )
        interactive = await page.evaluate(
            """
            () => {
              const nodes = Array.from(document.querySelectorAll('a, button, input, textarea, select, [role="button"]')).slice(0, 25);
              return nodes.map((node, index) => {
                const text = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('placeholder') || '').trim().replace(/\\s+/g, ' ');
                return {
                  index: index + 1,
                  tag: node.tagName.toLowerCase(),
                  text: text.slice(0, 120),
                  selector: node.id ? `#${node.id}` : null
                };
              });
            }
            """
        )
        self._last_url = url
        return {
            "url": url,
            "title": title,
            "text_preview": text_preview,
            "interactive_elements": interactive,
        }

    async def click(self, selector: str):
        page = await self.ensure_page()
        await page.locator(selector).first.click(timeout=15000)
        self._last_url = page.url
        return {"url": page.url, "title": await page.title(), "selector": selector}

    async def type(self, selector: str, text: str, clear: bool = True):
        page = await self.ensure_page()
        locator = page.locator(selector).first
        if clear:
            await locator.fill(text, timeout=15000)
        else:
            await locator.type(text, timeout=15000)
        self._last_url = page.url
        return {"url": page.url, "title": await page.title(), "selector": selector, "characters": len(text)}

    async def wait(self, seconds: float = 1.0):
        page = await self.ensure_page()
        await page.wait_for_timeout(max(0, seconds) * 1000)
        self._last_url = page.url
        return {"url": page.url, "title": await page.title(), "waited_seconds": seconds}

    async def screenshot(self, path: str = None, full_page: bool = True):
        page = await self.ensure_page()
        os.makedirs(self.artifact_dir, exist_ok=True)
        if not path:
            path = os.path.join(self.artifact_dir, f"screenshot-{int(time.time())}.png")
        await page.screenshot(path=path, full_page=full_page)
        self._last_url = page.url
        return {"url": page.url, "title": await page.title(), "path": os.path.abspath(path)}

    async def close(self):
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._last_url = None

    def export_state(self):
        return {
            "last_url": self._last_url,
            "available": PLAYWRIGHT_AVAILABLE,
        }

    def load_state(self, payload: dict):
        if not isinstance(payload, dict):
            self._last_url = None
            return

        self._last_url = payload.get("last_url") or None
