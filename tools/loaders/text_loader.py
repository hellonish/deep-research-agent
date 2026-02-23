"""
Loader for plain text and Markdown files.
"""
import asyncio
from typing import List

import aiofiles

from models.models import Document
from .base import BaseLoader


class TextLoader(BaseLoader):
    """
    Loader for .txt and .md files.
    """

    async def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load text from a local file path.

        Args:
            source: Absolute path to a .txt or .md file.
        """
        try:
            async with aiofiles.open(source, mode="r", encoding="utf-8") as f:
                content = await f.read()

            return [
                Document(
                    id=source,
                    content=content,
                    metadata={"source": "text", "path": source},
                )
            ]
        except Exception as e:
            print(f"Error in TextLoader: {e}")
            return []
