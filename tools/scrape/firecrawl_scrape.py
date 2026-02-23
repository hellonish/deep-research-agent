from typing import Optional
from firecrawl import FirecrawlApp
from models.models import Document
from .base import BaseScraper
import asyncio

class FirecrawlScraper(BaseScraper):
    """
    Scraper using Firecrawl API for high-quality markdown extraction.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # FirecrawlApp seems to be sync based on typical usage, wrapping in to_thread
        self.app = FirecrawlApp(api_key=api_key)

    async def scrape(self, url: str, **kwargs) -> Document:
        """
        Scrape URL using Firecrawl.
        """
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not provided")

        try:
            # scrape_url returns a dict
            # Check if scrape_url exists, otherwise try scrape
            if hasattr(self.app, 'scrape_url'):
                result = await asyncio.to_thread(self.app.scrape_url, url)
            else:
                result = await asyncio.to_thread(self.app.scrape, url)
            
            # The result might be a dict or an object depending on version
            if isinstance(result, dict):
                content = result.get("markdown", "") or result.get("content", "")
                metadata = result.get("metadata", {})
            else:
                # Metadata is also an object
                content = getattr(result, "markdown", "") or getattr(result, "content", "")
                meta_obj = getattr(result, "metadata", None)
                if meta_obj:
                     metadata = {
                         "title": getattr(meta_obj, "title", ""),
                         "description": getattr(meta_obj, "description", "")
                     }
                else:
                    metadata = {}
            
            return Document(
                id=url,
                content=content,
                metadata={
                    "source": "firecrawl",
                    "url": url,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", "")
                }
            )
        except Exception as e:
            print(f"Error in FirecrawlScraper: {e}")
            # Return empty doc on failure? Or raise?
            # Better to raise so fallback can be used
            raise e
