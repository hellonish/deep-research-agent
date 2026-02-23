import pytest
import asyncio
from agents.writer import WriterAgent
from states import ResearchNode, KnowledgeItem
from tests.agents.mock_llm import MockLLMClient

@pytest.mark.asyncio
async def test_writer_write_report():
    llm = MockLLMClient()
    writer = WriterAgent(llm)

    # Construct a dummy state tree
    root = ResearchNode(topic="Root topic", depth=0)
    root.knowledge = KnowledgeItem(source="Agent", content="Root found info.", summary="Root summary")
    
    child = ResearchNode(topic="Child topic", depth=1)
    child.knowledge = KnowledgeItem(source="Agent", content="Child found info.", summary="Child summary")
    root.children.append(child)

    report = await writer.write("dummy query", [root])
    
    assert report is not None
    assert report.title != ""
    assert len(report.blocks) > 0
