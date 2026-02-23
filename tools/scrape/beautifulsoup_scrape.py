import aiohttp
from bs4 import BeautifulSoup
from models.models import Document
from .base import BaseScraper

class BeautifulSoupScraper(BaseScraper):
    """
    Basic scraper using aiohttp and BeautifulSoup4.
    """
    async def scrape(self, url: str, **kwargs) -> Document:
        """
        Scrape URL using BS4.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, ssl=False) as response:
                    response.raise_for_status()
                    html = await response.text()
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator="\n")
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            title = soup.title.string if soup.title else ""
            
            return Document(
                id=url,
                content=clean_text,
                metadata={
                    "source": "beautifulsoup",
                    "url": url,
                    "title": title
                }
            )

        except Exception as e:
            print(f"Error in BeautifulSoupScraper: {e}")
            raise e
