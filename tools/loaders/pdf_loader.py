import fitz  # PyMuPDF
import asyncio
from typing import List
from models.models import Document
from .base import BaseLoader

class PDFLoader(BaseLoader):
    """
    Loader for PDF files using PyMuPDF.
    """
    async def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load PDF from a local path.
        source: local file path.
        """
        try:
            def parse_pdf():
                doc = fitz.open(source)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n\n"
                doc.close()
                return text

            content = await asyncio.to_thread(parse_pdf)
            
            return [Document(
                id=source,
                content=content,
                metadata={
                    "source": "pdf",
                    "path": source
                }
            )]
            
        except Exception as e:
            print(f"Error in PDFLoader: {e}")
            return []
