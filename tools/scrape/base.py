from abc import ABC, abstractmethod
from models.models import Document

class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.
    """
    
    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> Document:
        """
        Scrape content from a URL and return a Document.
        """
        pass
