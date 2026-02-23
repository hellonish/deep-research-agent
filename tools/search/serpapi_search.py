from typing import List
import asyncio
import os
import aiohttp
from models.models import Document
from .base import BaseSearch

class SerpAPISearch(BaseSearch):
    """
    Search implementation using SerpAPI (Google Search API).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    async def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        """
        Execute search using SerpAPI.
        """
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not provided")

        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": limit,
            "hl": "en",
            "gl": "us"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, ssl=False) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"SerpAPI Error: {response.status} - {text}")
                        return []
                    
                    data = await response.json()
            
            documents = []
            
            # organic_results is the main list
            results = data.get("organic_results", [])
            
            for result in results:
                documents.append(Document(
                    id=result.get("link", ""),
                    content=result.get("snippet", ""),
                    metadata={
                        "source": "serpapi",
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "position": result.get("position", 0)
                    }
                ))
                
            return documents
            
        except Exception as e:
            print(f"Error in SerpAPISearch: {e}")
            return []
