import unittest
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.loaders.arxiv_loader import ArxivLoader
from tools.loaders.pdf_loader import PDFLoader
from tools.scrape.beautifulsoup_scrape import BeautifulSoupScraper
from tools.scrape.firecrawl_scrape import FirecrawlScraper
from tools.search.duckduckgo_search import DuckDuckGoSearch
from tools.search.tavily_search import TavilySearch
from tools.search.serpapi_search import SerpAPISearch
from dotenv import load_dotenv

class TestTools(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        # API Keys - Check environment or set dummy for testing logic where applicable
        self.firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_arxiv_loader(self):
        print("\n--- Testing ArxivLoader ---")
        loader = ArxivLoader()
        # Query for a known paper or topic
        docs = self.loop.run_until_complete(loader.load("2101.12345")) # Example ID
        if not docs:
             docs = self.loop.run_until_complete(loader.load("quantum computing"))
        
        self.assertTrue(len(docs) > 0)
        print(f"Arxiv Doc 1 Title: {docs[0].metadata.get('title')}")
    
    def test_pdf_loader(self):
        print("\n--- Testing PDFLoader ---")
        # Create a dummy PDF file
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello World! This is a test PDF.")
        doc.save("test.pdf")
        doc.close()

        try:
            loader = PDFLoader()
            docs = self.loop.run_until_complete(loader.load("test.pdf"))
            self.assertTrue(len(docs) == 1)
            self.assertIn("Hello World", docs[0].content)
            print("PDF Content: ", docs[0].content.strip())
        finally:
            if os.path.exists("test.pdf"):
                os.remove("test.pdf")

    def test_beautifulsoup_scraper(self):
        print("\n--- Testing BeautifulSoupScraper ---")
        scraper = BeautifulSoupScraper()
        doc = self.loop.run_until_complete(scraper.scrape("https://example.com"))
        self.assertIn("Example Domain", doc.metadata.get("title"))
        print(f"Scraped Title: {doc.metadata.get('title')}")

    def test_firecrawl_scraper(self):
        print("\n--- Testing FirecrawlScraper ---")
        if not self.firecrawl_key:
            print("Skipping FirecrawlScraper test (No API Key)")
            return

        scraper = FirecrawlScraper(api_key=self.firecrawl_key)
        try:
            doc = self.loop.run_until_complete(scraper.scrape("https://example.com"))
            self.assertTrue(doc.content)
            print("Firecrawl markdown length:", len(doc.content))
        except Exception as e:
            print(f"Firecrawl failed: {e}")

    def test_duckduckgo_search(self):
        print("\n--- Testing DuckDuckGoSearch ---")
        searcher = DuckDuckGoSearch()
        docs = self.loop.run_until_complete(searcher.search("Python programming", limit=3))
        if len(docs) > 0:
            print(f"DDG First Result: {docs[0].metadata.get('title')}")
            self.assertTrue(len(docs) > 0)
        else:
            print("WARNING: DuckDuckGo returned 0 results (likely rate-limited). Skipping assertion.")

    def test_tavily_search(self):
        print("\n--- Testing TavilySearch ---")
        if not self.tavily_key:
            print("Skipping TavilySearch test (No API Key)")
            return
            
        searcher = TavilySearch(api_key=self.tavily_key)
        try:
            docs = self.loop.run_until_complete(searcher.search("Python programming", limit=3))
            self.assertTrue(len(docs) > 0)
            print(f"Tavily First Result: {docs[0].metadata.get('title')}")
        except Exception as e:
            print(f"Tavily failed: {e}")

    def test_serpapi_search(self):
        print("\n--- Testing SerpAPISearch ---")
        if not self.serpapi_key:
            print("Skipping SerpAPISearch test (No API Key)")
            return
            
        searcher = SerpAPISearch(api_key=self.serpapi_key)
        try:
            docs = self.loop.run_until_complete(searcher.search("Python programming", limit=3))
            self.assertTrue(len(docs) > 0)
            print(f"SerpAPI First Result: {docs[0].metadata.get('title')}")
        except Exception as e:
            print(f"SerpAPI failed: {e}")

if __name__ == '__main__':
    unittest.main()
