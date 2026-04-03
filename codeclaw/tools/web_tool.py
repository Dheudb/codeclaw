import re
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    max_results: int = Field(5, description="Number of results to return (max 10).")

class WebSearchTool(BaseAgenticTool):
    name = "web_search_tool"
    description = "Search the web for real-time information about any topic. Returns summarized information from search results and relevant URLs."
    input_schema = WebSearchInput
    is_read_only = True
    risk_level = "low"
    
    async def execute(self, query: str, max_results: int = 5) -> str:
        try:
            results = DDGS().text(query, max_results=max_results)
            out = f"Web Search Results for: '{query}'\\n" + "="*40 + "\\n"
            for r in results:
                out += f"Title: {r.get('title')}\\nURL: {r.get('href')}\\nSnippet: {r.get('body')}\\n\\n"
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

            if len(text) > 30000:
                text = text[:30000] + "\n\n...[Content truncated due to size]..."

            result = f"Content from {url}:\n\n{text}"
            if prompt:
                result += f"\n\n---\nExtraction prompt: {prompt}"
            return result

        except Exception as e:
            return f"Error fetching webpage: {str(e)}"
