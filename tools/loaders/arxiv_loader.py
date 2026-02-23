import arxiv
import asyncio
from typing import List
from models.models import Document
from .base import BaseLoader

class ArxivLoader(BaseLoader):
    """
    Loader for Arxiv papers using the arxiv library.
    """
    async def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load paper by ID or query. 
        source: can be an Arxiv ID (e.g. "2101.12345") or a query.
        """
        try:
            # We assume source is an ID or list of IDs for specific loading, 
            # OR a search query if it doesn't look like an ID.
            # Simple heuristic: if it has spaces, it's a query.
            
            client = arxiv.Client()
            
            if " " in source:
                search = arxiv.Search(
                    query=source,
                    max_results=kwargs.get("limit", 1),
                    sort_by=arxiv.SortCriterion.Relevance
                )
            else:
                search = arxiv.Search(id_list=[source])
            
            documents = []
            
            # Run generator in thread to avoid blocking
            def fetch():
                return list(client.results(search))
                
            results = await asyncio.to_thread(fetch)
            
            for result in results:
                # We can download the PDF and parse it, OR just use the abstract/summary.
                # For deep research, PDF content is better. 
                # But arxiv lib only gives metadata + summary.
                # Use PDFLoader for full content if needed.
                # For now, we return the summary + metadata.
                
                content = f"Title: {result.title}\n\nAbstract: {result.summary}\n\nPublished: {result.published}"
                
                documents.append(Document(
                    id=result.entry_id,
                    content=content,
                    metadata={
                        "source": "arxiv",
                        "title": result.title,
                        "url": result.pdf_url,
                        "authors": [a.name for a in result.authors],
                        "published": str(result.published)
                    }
                ))
                
            return documents
            
        except Exception as e:
            print(f"Error in ArxivLoader: {e}")
            return []
