"""
Loader for HTML files — strips tags and extracts readable text.
"""
import asyncio
from typing import List

import aiofiles
from bs4 import BeautifulSoup

from models.models import Document
from .base import BaseLoader


class HTMLLoader(BaseLoader):
    """
    Loader for .html files.  Strips tags, scripts, and styles to produce
    clean text content.
    """

    async def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load and clean HTML from a local file path.

        Args:
            source: Absolute path to an .html file.
        """
        try:
            async with aiofiles.open(source, mode="r", encoding="utf-8") as f:
                raw_html = await f.read()

            def parse():
                soup = BeautifulSoup(raw_html, "html.parser")
                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)

            content = await asyncio.to_thread(parse)

            return [
                Document(
                    id=source,
                    content=content,
                    metadata={"source": "html", "path": source},
                )
            ]
        except Exception as e:
            print(f"Error in HTMLLoader: {e}")
            return []
