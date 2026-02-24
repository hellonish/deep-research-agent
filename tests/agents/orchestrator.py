"""
Test orchestrator tree executor only: one plan step from JSON → BFS research tree → output tree to JSON.
No planner call, no writer; only _execute_research_tree for a single root node.

Depth-1 gap children are always processed (orchestrator skips duplicate check at depth 1).
"""
import asyncio
import json
import os

from llm import DeepSeekClient
from vector_store import QdrantStore
from agents.config import ReportConfig
from agents.orchestrator import OrchestratorAgent
from states import ResearchNode

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "planner_output.json")
OUTPUT_PATH = os.path.join(HERE, "orchestrator_tree_output.json")

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


def _tree_to_dict(nodes: list[ResearchNode]) -> list[dict]:
    """Serialize research tree (root nodes + children recursively) to JSON-serializable dicts."""
    return [node.model_dump() for node in nodes]


async def run_tree_executor(step_index: int = 0):
    """Run only the tree executor for a single plan step; save the resulting tree to JSON."""
    topic = _load_one_plan_step(step_index)
    print(f"Single root node topic: {topic[:100]}...")
    print(f"Query: {USER_QUERY[:80]}...")
    print()

    root_nodes = [
        ResearchNode(topic=topic, depth=0, node_id="0"),
    ]

    llm = DeepSeekClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research", dense_dim=384)

    config = ReportConfig.STANDARD()
    orchestrator = OrchestratorAgent(
        llm_client=llm,
        vector_store=vector_store,
        collection_name="research",
        config=config,
    )

    async def on_progress(event_type: str, data: dict):
        probe = data.get("probe", data.get("node_id", event_type))

        if isinstance(probe, str) and len(probe) > 60:
            probe = probe[:60] + "..."

        print(f"  [{event_type}] {probe}")

    print("Running tree executor only (no planner, no writer)...")
    await orchestrator._execute_research_tree(
        root_nodes,
        USER_QUERY,
        progress_callback=on_progress,
    )

    tree_dict = _tree_to_dict(root_nodes)

    out = {
        "user_query": USER_QUERY,
        "step_index": step_index,
        "tree": tree_dict,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nTree output saved to {OUTPUT_PATH}")
    return root_nodes


if __name__ == "__main__":
    asyncio.run(run_tree_executor(step_index=0))
