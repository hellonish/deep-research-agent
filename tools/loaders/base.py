from abc import ABC, abstractmethod
from typing import List
from models.models import Document

class BaseLoader(ABC):
    """
    Abstract base class for document loaders (PDF, Arxiv, etc.).
    """
    
    @abstractmethod
    async def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load content from a source (path or ID) and return a list of Documents.
        """
        pass
