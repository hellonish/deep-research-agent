import pytest
import asyncio
from agents.researcher import ResearcherAgent
from agents.config import ReportConfig
from vector_store import QdrantStore
from tests.agents.mock_llm import MockLLMClient

@pytest.mark.asyncio
async def test_researcher_populate_and_resolve():
    llm = MockLLMClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research")
    
    config = ReportConfig.STANDARD()
    config.max_tool_pairs = 1
    
    researcher = ResearcherAgent(llm, vector_store, config=config)
    
    topic = "dummy topic"
    urls = await researcher.populate(topic)
    
    # We used our mock tool pair (arxiv_loader), let's ensure it returned something or executed without crash
    assert isinstance(urls, list)
    
    item, gap_response = await researcher.resolve(topic, "dummy original query")
    
    assert item is not None
    assert item.content != ""
    assert gap_response is not None
    assert gap_response.is_complete is True # Mock was set to return true
