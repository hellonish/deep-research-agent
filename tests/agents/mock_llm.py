import json
from pathlib import Path

from pydantic import BaseModel
from llm import BaseLLMClient
from models import ResearchPlan, ToolPlan, ResearchReport, ContentBlock, BlockType
from models import GapAnalysis
from states import KnowledgeItem, NodeDraft

RESULTS_DIR = Path(__file__).parent / "results"

class MockLLMClient(BaseLLMClient):
    """
    Mocks the LLM to return static dummy data collected from real executions.
    """
    def __init__(self):
        super().__init__()
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        with open(RESULTS_DIR / "dummy_knowledge_item.json") as f:
            data = json.load(f)
        return data["content"]
        
    def generate_structured(self, prompt: str, system_prompt: str, schema: type[BaseModel], temperature: float = 0.5) -> BaseModel:
        if schema == ResearchPlan:
            with open(RESULTS_DIR / "dummy_plan.json") as f:
                return schema.model_validate(json.load(f))
        
        elif schema == ToolPlan:
            # Create a simple mock tool plan to avoid real network searching
            from models import ToolPair
            return schema(pairs=[
                ToolPair(tool_name="arxiv_loader", query="dummy search")
            ])
            
        elif schema == GapAnalysis:
            # Return an empty gap to stop expanding tree infinitely
            return schema(is_complete=True, gaps=[])
            
        elif schema == ResearchReport:
            with open(RESULTS_DIR / "dummy_report.json") as f:
                return schema.model_validate(json.load(f))

        elif schema == NodeDraft:
            return NodeDraft(
                node_topic="Topic",
                blocks=[
                    ContentBlock(block_type=BlockType.TEXT, markdown="Section content."),
                ],
                compressed_summary="Key findings summary.",
                local_sources=[],
            )

        elif schema.__name__ == "ExecutiveSummaryOutput":
            from prompts.get_executive_summary import ExecutiveSummaryOutput
            return ExecutiveSummaryOutput(
                title="Report Title",
                summary="Executive summary of findings.",
            )

        raise ValueError(f"No mock defined for schema {schema}")
