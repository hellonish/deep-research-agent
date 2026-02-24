import asyncio
import json
import os
from pathlib import Path

from llm import GeminiClient
from vector_store import QdrantStore
from agents.config import ReportConfig
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.orchestrator import OrchestratorAgent

RESULTS_DIR = Path("tests/agents/results")

async def generate_dummy_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Initializing components...")
    llm = GeminiClient()
    vector_store = QdrantStore(in_memory=True)
    config = ReportConfig(
        name="Test",
        num_plan_steps=2,
        max_depth=1,
        max_probes=2,
        max_tool_pairs=2,
        dupe_threshold=0.85,
        rag_top_k=3
    )
    
    query = "Recent advancements in Solid State Batteries 2024"
    
    print("Creating in-memory vector collection...")
    await vector_store.create_collection("research")

    # --- 1. PLANNER ---
    print("\n--- Running Planner ---")
    planner = PlannerAgent(llm)
    plan = planner.create_plan(query, num_plan_steps=config.num_plan_steps)
    
    with open(RESULTS_DIR / "dummy_plan.json", "w") as f:
        f.write(plan.model_dump_json(indent=2))
    print("Saved dummy_plan.json")

    # --- 2. RESEARCHER ---
    print("\n--- Running Researcher ---")
    researcher = ResearcherAgent(llm, vector_store, config)
    
    topic = plan.plan[0].description
    print(f"Populating memory for: {topic}")
    urls = await researcher.populate(topic)
    
    print(f"Resolving topic: {topic}")
    item, gap_response = await researcher.resolve(topic, query)
    
    with open(RESULTS_DIR / "dummy_knowledge_item.json", "w") as f:
        f.write(item.model_dump_json(indent=2))
    print("Saved dummy_knowledge_item.json")

    # --- 3. ORCHESTRATOR & WRITER ---
    # Orchestrator runs the whole flow and returns a Report
    print("\n--- Running Orchestrator (Full Flow) ---")
    orchestrator = OrchestratorAgent(llm, vector_store, config)
    report = await orchestrator.run(query)
    
    with open(RESULTS_DIR / "dummy_report.json", "w") as f:
        f.write(report.model_dump_json(indent=2))
    print("Saved dummy_report.json")

if __name__ == "__main__":
    asyncio.run(generate_dummy_data())
