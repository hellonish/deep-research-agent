from typing import List, Dict
import asyncio
from ddgs import DDGS
from models.models import Document
from .base import BaseSearch

class DuckDuckGoSearch(BaseSearch):
    """
    Search implementation using DuckDuckGo (Open Source / Free).
    """
    
    async def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        """
        Execute search using DDGS in a thread.
        """
        try:
            def fetch():
                # Try multiple backends as fallback
                backends = ["html", "lite", "auto"]
                for backend in backends:
                    try:
                        with DDGS() as ddgs:
                            print(f"Trying DDG backend: {backend}")
                            results = list(ddgs.text(query, max_results=limit, backend=backend))
                            if results:
                                return results
                    except Exception as e:
                        print(f"DDG backend '{backend}' failed: {e}")
                        continue
                return []

            results = await asyncio.to_thread(fetch)
            
            documents = []
            for result in results:
                documents.append(Document(
                    id=result.get("href", ""),
                    content=result.get("body", ""),
                    metadata={
                        "source": "duckduckgo",
                        "title": result.get("title", ""),
                        "url": result.get("href", "")
                    }
                ))
            return documents
            
        except Exception as e:
            print(f"Error in DuckDuckGoSearch: {e}")
            return []
