"""
Test script to run the Researcher Agent independently.
Outputs the resulting KnowledgeItem to a JSON file to use in the agent chain.
"""
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from llm.gemini import GeminiClient
from vector_store.qdrant_store import QdrantStore
from agents.config.config import ReportConfig
from agents.researcher.researcher import ResearcherAgent
from states import KnowledgeItem

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"

TEST_GAP_QUERY = "What are the core technical differences between Anthropic Claude 3.5 Sonnet and OpenAI GPT-4o in terms of benchmark performance and architecture?"

async def run_demo():
    print(f"[Test] Initializing Researcher Agent...")
    
    # 1. Initialize dependencies
    llm_client = GeminiClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research", dense_dim=384)
    
    config = ReportConfig(
        max_tool_pairs=3, 
        rag_top_k=5
    )
    
    # 2. Setup Researcher Agent
    researcher = ResearcherAgent(
        llm_client=llm_client,
        vector_store=vector_store,
        config=config,
    )
    
    print(f"\n[Test] Running Researcher.resolve() for gap:\n'{TEST_GAP_QUERY}'\n")
    
    # 3. Run the targeted resolution
    knowledge_item: KnowledgeItem = await researcher.resolve(TEST_GAP_QUERY)
    
    # 4. Save results to JSON for pipelining
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "researcher_output.json"
    
    # Assuming KnowledgeItem is a Pydantic model
    out_path.write_text(knowledge_item.model_dump_json(indent=2))
    
    print(f"\n[Test] Success! Result saved to {out_path}")
    print(f"[Test] Title: {knowledge_item.title}")
    print(f"[Test] Sources used: {len(knowledge_item.sources)}")
    print(f"[Test] Summary snippet: {knowledge_item.summary[:150]}...\n")

if __name__ == "__main__":
    asyncio.run(run_demo())
