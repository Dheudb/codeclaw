import re
import asyncio
import sys
import httpx
from bs4 import BeautifulSoup
if sys.version_info >= (3, 10):
    from ddgs import DDGS
else:
    from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from xuloom.tools.base import BaseAgenticTool
from xuloom.context.tool_results import build_tool_result

MAX_SEARCH_RESULTS = 50

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    max_results: int = Field(5, ge=1, le=MAX_SEARCH_RESULTS,
        description="Maximum number of results to request (1–50, default 5). Use 15–30 for broader research when useful; the provider may return fewer results.")

class WebSearchTool(BaseAgenticTool):
    name = "web_search_tool"
    description = "Search the web for real-time information about any topic. Returns summarized information from search results and relevant URLs."
    input_schema = WebSearchInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        import datetime
        month_year = datetime.datetime.now().strftime("%B %Y")
        return f"""Search the web for real-time information about any topic. Returns search result information formatted as search result blocks, including links as markdown hyperlinks.

Use this tool directly when the user asks to search the public web, research papers/models, or verify current information. It is already active and needs no tool discovery or model-provider API feature. Keep public research scoped to online sources; inspect project files only for project-specific requests.
If the search service fails, report the service error and try a revised query or a configured search MCP. A provider failure does not mean the tool is unavailable.
max_results accepts 1–50 (default 5). Request more results for broad research when needed, not automatically for every query. The search provider may return fewer; this is a maximum, not a guaranteed count.

CRITICAL REQUIREMENT:
- After answering the user's question, you MUST include a "Sources:" section at the end of your response
- In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)

IMPORTANT - Use the correct year in search queries:
- The current month is {month_year}. Include an appropriate year when recency matters without excluding relevant earlier research."""

    async def execute(self, query: str, max_results: int = 5) -> dict:
        try:
            limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
            # DDGS is synchronous; its network calls must not block streaming or Stop.
            results = await asyncio.wait_for(asyncio.to_thread(lambda: list(DDGS(timeout=8).text(query, max_results=limit))), timeout=35)
            results = [r for r in results if isinstance(r, dict) and str(r.get('href', '')).startswith(('https://', 'http://'))][:limit]
            if not results:
                return build_tool_result(ok=False, is_error=True, content="The built-in web search service returned no usable results. Try a revised query or a configured search MCP. The web search tool exists; local project files are not a substitute for this public-web research.", metadata={"backend": "ddgs", "query": query, "reason": "no_results"})
            out = f"Web Search Results for: '{query}'\n" + "="*40 + "\n"
            for r in results:
                out += f"Title: {str(r.get('title', ''))[:500]}\nURL: {r.get('href')}\nSnippet: {str(r.get('body', ''))[:2500]}\n\n"
            return build_tool_result(ok=True, content=out, metadata={"backend": "ddgs", "query": query, "result_count": len(results)})
        except Exception as e:
            return build_tool_result(ok=False, is_error=True, content=f"Built-in web search service failed ({type(e).__name__}). Check network access, retry, or use a configured search MCP. This is a search-provider failure, not a missing tool.", metadata={"backend": "ddgs", "query": query, "reason": type(e).__name__})

    def input_error_hint(self) -> str:
        return 'Use {"query":"your search terms","max_results":15}. max_results must be between 1 and 50; omit it for the default of 5. The provider may return fewer results.'

class WebFetchInput(BaseModel):
    url: str = Field(..., description="The URL to fetch content from.")
    prompt: str = Field("", description="Optional description of what information to extract from the page.")

class WebFetchTool(BaseAgenticTool):
    name = "web_fetch_tool"
    description = "Fetch content from a specified URL and return its contents in a readable format. Use a prompt to specify what information to extract."
    input_schema = WebFetchInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        return """Fetches static HTML from a URL, removes scripts and navigation, and returns plain text for you to analyze. The optional extraction prompt is included with the returned text; no separate AI extraction is performed.

Usage notes:
- The URL must be a fully-formed valid URL
- The prompt should describe what information you want to extract from the page
- This tool is read-only and does not modify any files
- Content is truncated at 100,000 characters. JavaScript is not executed; use browser_tool for dynamic pages.
- Redirects are followed and the final URL is reported.
- For GitHub URLs, prefer using the gh CLI via bash_tool instead (e.g., gh pr view, gh issue view, gh api)"""

    async def execute(self, url: str, prompt: str = "", **kwargs) -> str:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'lxml')

            for script in soup(["script", "style", "nav", "footer", "meta", "noscript"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)

            if len(text) > 100000:
                text = text[:100000] + "\n\n...[Content truncated due to size]..."

            result = f"Content from {resp.url}:\n\n{text}"
            if prompt:
                result += f"\n\n---\nExtraction prompt: {prompt}"
            return result

        except Exception as e:
            return build_tool_result(ok=False, is_error=True, content=f"Webpage request failed ({type(e).__name__}). Check the URL and network, or try another source.", metadata={"url": url, "reason": type(e).__name__})
