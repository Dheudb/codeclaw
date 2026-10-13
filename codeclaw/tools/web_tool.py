import re
import asyncio
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (1–10).")

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

Use this tool for accessing information beyond your knowledge cutoff.

CRITICAL REQUIREMENT:
- After answering the user's question, you MUST include a "Sources:" section at the end of your response
- In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)

IMPORTANT - Use the correct year in search queries:
- The current month is {month_year}. You MUST use this year when searching for recent information.
- Example: search for "React documentation {datetime.datetime.now().year}", NOT older years"""

    async def execute(self, query: str, max_results: int = 5) -> str:
        try:
            limit = max(1, min(int(max_results), 10))
            # DDGS is synchronous; its network calls must not block streaming or Stop.
            results = await asyncio.to_thread(lambda: list(DDGS(timeout=12).text(query, max_results=limit)))
            if not results:
                return "No web search results were returned. Try another query or an available search MCP."
            out = f"Web Search Results for: '{query}'\n" + "="*40 + "\n"
            for r in results:
                out += f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n\n"
            return out
        except Exception as e:
            return f"Error executing Web Search: {str(e)}"

class WebFetchInput(BaseModel):
    url: str = Field(..., description="The URL to fetch content from.")
    prompt: str = Field(..., description="A prompt describing what information to extract from the page.")

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
            return f"Error fetching webpage: {str(e)}"
