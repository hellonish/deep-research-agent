import os
import asyncio
from typing import List
from tavily import TavilyClient
from models.models import Document, SearchResult
from .base import BaseSearch

class TavilySearch(BaseSearch):
    """
    Search implementation using Tavily API (LLM-optimized).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = TavilyClient(api_key=api_key)

    async def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        """
        Execute search using Tavily.
        """
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not provided")

        try:
            # content=True to get the scraped text
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="advanced",
                max_results=limit,
                include_answer=True,
                include_raw_content=False,
                include_images=False
            )
            
            documents = []
            
            # Add the AI answer as a document if present and not empty
            answer = response.get("answer")
            if answer:
                documents.append(Document(
                    id="tavily_answer",
                    content=answer,
                    metadata={"source": "tavily_answer", "query": query, "title": "Tavily Answer"}
                ))
            
            # Process search results
            for result in response.get("results", []):
                documents.append(Document(
                    id=result.get("url", ""),
                    content=result.get("content", ""),
                    metadata={
                        "source": "tavily",
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0)
                    }
                ))
                
            return documents
            
        except Exception as e:
            # In a real agent, we might log this. For now, re-raise or return empty.
            print(f"Error in TavilySearch: {e}")
            return []
