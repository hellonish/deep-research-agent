"""
Test researcher functionality using a single node from the plan.
Loads one step from planner_output.json, runs populate + resolve, prints result.
"""
import asyncio
import json
import os

from llm import DeepSeekClient
from vector_store.qdrant_store import QdrantStore
from agents.config.config import ReportConfig
from agents.researcher.researcher import ResearcherAgent
from models import Gap

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "planner_output.json")
OUTPUT_PATH = os.path.join(HERE, "researcher_output.json")

# Overall research goal (used as original_goal in resolve)
USER_QUERY = "Research AegisAI and what kind of problems they solve."


def _load_one_plan_step(step_index: int = 0) -> str:
    """Load plan from planner_output.json; return description of step at step_index (topic for one node)."""
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    steps = data.get("plan", [])
    
    if not steps or step_index >= len(steps):
        raise IndexError(f"Plan has {len(steps)} steps; step_index={step_index} invalid")
    
    step = steps[step_index]
    return step.get("description") or step.get("action", "")


async def run_researcher_one_node(step_index: int = 0):
    """Run researcher for a single plan step: populate then resolve."""
    topic = _load_one_plan_step(step_index)
    print(f"Topic (one node): {topic[:120]}...")
    
    print()

    llm = DeepSeekClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research", dense_dim=384)

    config = ReportConfig.STANDARD()

    researcher = ResearcherAgent(
        llm_client=llm,
        vector_store=vector_store,
        collection_name="research",
        config=config,
    )

    async def on_progress(event_type: str, data: dict):
        print(f"  [{event_type}] {data.get('tool', data.get('probe', data))}")

    print("Populating (tools + vector store)...")
    await researcher.populate(topic, progress_callback=on_progress)
    
    print()

    print("Resolving (synthesize + gaps)...")
    
    knowledge_item, gap_response = await researcher.resolve(
        topic, USER_QUERY, progress_callback=on_progress
    )

    print("\n--- KnowledgeItem ---")
    print(f"Source: {knowledge_item.source}")
    print(f"Summary: {(knowledge_item.summary or '')[:200]}...")
    print(f"Content length: {len(knowledge_item.content)} chars")
    print(f"Sources: {knowledge_item.sources[:5]}")
    print("\n--- GapAnalysis ---")
    print(f"is_complete: {gap_response.is_complete}")
    print(f"Gaps: {len(gap_response.gaps)}")
    for g in gap_response.gaps[:5]:
        print(f"  - severity={g.severity}: {g.query[:80]}...")

    # Persist using project models (KnowledgeItem from states, GapAnalysis/Gap from models)
    out = {
        "topic": topic,
        "user_query": USER_QUERY,
        "step_index": step_index,
        "knowledge_item": knowledge_item.model_dump(),
        "gap_analysis": gap_response.model_dump(),
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to {OUTPUT_PATH}")

    return knowledge_item, gap_response

# test drop function
def test_drop_function():
    researcher = ResearcherAgent(llm_client=DeepSeekClient(), vector_store=QdrantStore(in_memory=True), collection_name="research", config=ReportConfig.STANDARD()) 
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(researcher._threshold_filter([Gap(query=gap.get("query"), severity=gap.get("severity")) for gap in data.get("gap_analysis", {}).get("gaps", [])]))


if __name__ == "__main__":
    test_drop_function()
    # asyncio.run(run_researcher_one_node(step_index=0))
