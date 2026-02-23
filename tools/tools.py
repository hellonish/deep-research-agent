from typing import Any
from agents.config import AgentConfig
from .search import TavilySearch, DuckDuckGoSearch, SerpAPISearch
from .scrape import FirecrawlScraper, BeautifulSoupScraper
from .loaders import ArxivLoader, PDFLoader, TextLoader, HTMLLoader, DocxLoader


class ToolExecutor:
    """
    Manages and executes all research tools with fallback logic.
    """

    def __init__(self):
        self.config = AgentConfig()

        # Search Tools
        self.tavily = TavilySearch(api_key=self.config.TAVILY_API_KEY) if self.config.TAVILY_API_KEY else None
        self.serpapi = SerpAPISearch(api_key=self.config.SERPAPI_API_KEY) if self.config.SERPAPI_API_KEY else None
        self.ddg = DuckDuckGoSearch()

        # Scrape Tools
        self.firecrawl = FirecrawlScraper(api_key=self.config.FIRECRAWL_API_KEY) if self.config.FIRECRAWL_API_KEY else None
        self.bs4 = BeautifulSoupScraper()

        # Loaders
        self.arxiv = ArxivLoader()
        self.pdf = PDFLoader()
        self.text_loader = TextLoader()
        self.html_loader = HTMLLoader()
        self.docx_loader = DocxLoader()

    async def execute(self, tool_name: str, **kwargs) -> Any:
        try:
            print(f"Executing tool: {tool_name} with args: {kwargs}")

            if tool_name == "tavily_search":
                return await self.tavily.search(**kwargs) if self.tavily else await self.ddg.search(**kwargs)

            elif tool_name == "serpapi_search":
                return await self.serpapi.search(**kwargs) if self.serpapi else "Error: SerpAPI key not configured."

            elif tool_name == "duckduckgo_search":
                return await self.ddg.search(**kwargs)

            elif tool_name == "firecrawl_scrape":
                return await self.firecrawl.scrape(**kwargs) if self.firecrawl else await self.bs4.scrape(**kwargs)

            elif tool_name == "bs4_scrape":
                return await self.bs4.scrape(**kwargs)

            elif tool_name == "arxiv_loader":
                return await self.arxiv.load(**kwargs)

            elif tool_name == "pdf_loader":
                return await self.pdf.load(**kwargs)

            elif tool_name == "text_loader":
                return await self.text_loader.load(**kwargs)

            elif tool_name == "html_loader":
                return await self.html_loader.load(**kwargs)

            elif tool_name == "docx_loader":
                return await self.docx_loader.load(**kwargs)

            else:
                return f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            print(f"Tool Execution Error [{tool_name}]: {e}")
            return f"Error executing {tool_name}: {str(e)}"
