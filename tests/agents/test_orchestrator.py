import pytest
import asyncio
from agents.orchestrator import OrchestratorAgent
from agents.config import ReportConfig
from vector_store import QdrantStore
from tests.agents.mock_llm import MockLLMClient

@pytest.mark.asyncio
async def test_orchestrator_bfs_flow():
    llm = MockLLMClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research")
    
    config = ReportConfig.STANDARD()
    config.max_depth = 1 # Keep it short
    config.num_plan_steps = 1
    config.max_tool_pairs = 1

    orchestrator = OrchestratorAgent(llm, vector_store, config=config)
    
    # Run the full V3 pipeline (Mock LLM will return GapAnalysis.is_complete=True to prevent expansion)
    report = await orchestrator.run("dummy query")
    
    assert report is not None
    assert len(report.blocks) > 0
