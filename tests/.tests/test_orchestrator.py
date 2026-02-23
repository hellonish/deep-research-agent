"""
Integration test for the full Orchestrator pipeline.

Runs real LLM + real search APIs end-to-end on a single query.
Results are written to tests/results/test_orchestrator_result.md and .json.

Run with:
    python -m pytest tests/test_orchestrator.py -v -s
Or directly:
    python tests/test_orchestrator.py
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pytest
from agents.orchestrator.orchestrator import OrchestratorAgent
from models.content_blocks import ResearchReport
from llm.gemini import GeminiClient
from vector_store.qdrant_store import QdrantStore

load_dotenv()

# ── Cost Control Constants ─────────────────────────────────────────────────────
MAX_DEPTH = 4
NUM_PLAN_STEPS = 5
MAX_PROBES = 3
MAX_TOOL_PAIRS = 3
DUPE_THRESHOLD = 0.75
# TEST_QUERY = "Explain origin of Neural Networks, then how the CNN came to play, and then RNNs, what happened next that we are now working with LLMs and World Models?, I want comparisons, market share of each type of model, and how they evolved. You can use Bar charts and Pie charts to represent the data. Show me code snippets to show implementation of each type of model."
TEST_QUERY = """
Conduct a comprehensive, deep-dive technical research report on the company Aegis AI (official website: https://www.aegisai.ai). I am actively preparing for an interview for an 'AI Engineer' role at this company. 

Please systematically gather and synthesize the following Company Intel:
1. Core Business & Mission: What are their primary products or services? What specific problems do they solve for their customers, and what is their business model? Please explicitly scrape their main website to find this.
2. Technical Architecture & AI Stack: Detail their technology stack. What kind of AI/ML models, frameworks (e.g., PyTorch, TensorFlow), cloud infrastructure, or data pipelines do they use or mention in their engineering blogs, job postings, or technical documentation?
3. Market & Competitors: Who are their main competitors? How does Aegis AI differentiate itself technically and commercially? 
4. Recent Traits & News: Summarize their recent funding rounds, major partnerships, leadership changes, or notable news articles from the past 12 months.
5. Interview Preparation: Based on their technical focus, what specific machine learning concepts, system design patterns, or coding skills should an AI Engineer prepare for to succeed in their interview?

Use varied block types for the report: Use 'table' blocks to compare competitors, 'text' blocks for deep explanations of their tech stack, and 'code' blocks to illustrate the specific types of algorithms or integrations an AI Engineer there might work on.
"""
# TEST_QUERY="""
# Create an intense report of all YC backed startups in 2025-2026, and how they supported Agentic AI, what type of companies were found to be successful, and what type of companies failed.  
# Find genres of companies that were successful and failed. How much was the average funding of successful companies and what is the project size of successful companies? 
# """

RESULTS_DIR = Path(__file__).parent / "results"


# ── Result Writer ──────────────────────────────────────────────────────────────

def write_results_md(query: str, report: ResearchReport, elapsed: float):
    """Render a ResearchReport to markdown, showing each content block."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "test_orchestrator_result.md"

    lines = [
        f"# {report.title}",
        "",
        f"**Query:** {query}",
        f"**Cost controls:** max_depth={MAX_DEPTH}, num_plan_steps={NUM_PLAN_STEPS}, max_probes={MAX_PROBES}, max_tool_pairs={MAX_TOOL_PAIRS}, dupe_thresh={DUPE_THRESHOLD}",
        f"**Total blocks:** {len(report.blocks)}",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "---",
        "",
    ]

    for i, block in enumerate(report.blocks, 1):
        block_title = block.title or f"Block {i}"
        lines += [f"## [{i}] {block.block_type.value}: {block_title}", ""]

        if block.block_type.value == "text" and block.markdown:
            lines += [block.markdown, ""]

        elif block.block_type.value == "table" and block.headers:
            lines.append(f"| {' | '.join(block.headers)} |")
            lines.append(f"| {' | '.join('---' for _ in block.headers)} |")
            for row in (block.rows or []):
                lines.append(f"| {' | '.join(str(c) for c in row)} |")
            lines.append("")

        elif block.block_type.value == "chart":
            lines.append(f"**Chart type:** {block.chart_type or 'unknown'}")
            chart_data = {"labels": block.labels or [], "datasets": []}
            for ds in (block.datasets or []):
                chart_data["datasets"].append({"label": ds.label, "data": ds.data})
            lines += ["```json", json.dumps(chart_data, indent=2), "```", ""]

        elif block.block_type.value == "code" and block.code:
            lines += [f"```{block.language or ''}", block.code, "```", ""]

        elif block.block_type.value == "source_list" and block.sources:
            for src in block.sources:
                lines.append(f"- {src}")
            lines.append("")

        lines += ["---", ""]

    path.write_text("\n".join(lines))
    print(f"\n[Test] Results written to {path}")

    # Also write the raw JSON for frontend testing
    json_path = RESULTS_DIR / "test_orchestrator_result.json"
    json_path.write_text(report.model_dump_json(indent=2))
    print(f"[Test] JSON written to {json_path}")

    return path


# ── Integration Test ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_full_pipeline():
    """
    LIVE integration test — calls real Gemini LLM + real search tools.
    Results written to tests/results/test_orchestrator_result.md and .json.
    """
    import time
    start = time.time()

    llm_client = GeminiClient()
    vector_store = QdrantStore(in_memory=True)
    await vector_store.create_collection("research", dense_dim=384)

    from agents.config.config import ReportConfig

    config = ReportConfig(
        name="Test Profile",
        num_plan_steps=NUM_PLAN_STEPS,
        max_depth=MAX_DEPTH,
        max_probes=MAX_PROBES,
        max_tool_pairs=MAX_TOOL_PAIRS,
        dupe_threshold=DUPE_THRESHOLD,
        rag_top_k=5,
    )

    try:
        orchestrator = OrchestratorAgent(
            llm_client=llm_client,
            vector_store=vector_store,
            config=config,
        )
        report = await orchestrator.run(TEST_QUERY)
    finally:
        pass # No cleanup needed now that we aren't patching

    elapsed = time.time() - start

    write_results_md(TEST_QUERY, report, elapsed)

    # Assertions
    assert isinstance(report, ResearchReport), "Expected a ResearchReport"
    assert report.title, "Report should have a title"
    assert report.summary, "Report should have a summary"
    assert len(report.blocks) > 0, "Expected at least one content block"
    print(f"\n[Test] PASSED — {len(report.blocks)} blocks generated")
    for b in report.blocks:
        print(f"  [{b.block_type.value}] {b.title}")


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(test_orchestrator_full_pipeline())
