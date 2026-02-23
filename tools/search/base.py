from abc import ABC, abstractmethod
from typing import List
from models.models import Document

class BaseSearch(ABC):
    """
    Abstract base class for search tools.
    """
    
    @abstractmethod
    async def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        """
        Execute a search query and return a list of Documents.
        """
        pass
