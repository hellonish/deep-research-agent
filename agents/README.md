# Agents — Research pipeline

The agents package implements the **deep research** pipeline: plan → explore (BFS over topics) → write. It is used by the backend when a user starts a research job; the backend runs `OrchestratorAgent.run()` in a background task and streams progress via a callback.

## Pipeline overview

```
User query
    │
    ▼
┌─────────────┐
│  Planner    │  create_plan(query) → ResearchPlan (list of steps/topics)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Orchestrator│  Level-based BFS over ResearchNodes
│             │  For each node: Researcher.populate → Researcher.resolve
│             │  Builds tree of nodes + knowledge; may add child nodes (gaps)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Writer     │  write(query, root_nodes) → ResearchReport (title, summary, blocks)
└─────────────┘
```

- **Planner** — Turns the user’s research question into an ordered list of topics (steps). Implemented in `agents/planner/`; uses `prompts.get_plan`.
- **Orchestrator** — Entry point. Builds root nodes from the plan, then runs level-by-level BFS: at each level it processes nodes in parallel (populate + resolve), then collects children (gaps) for the next level. Handles deduplication and max depth. Lives in `agents/orchestrator/`.
- **Researcher** — For a single topic: (1) **populate**: get tool map from LLM, run tools (search, loaders), store results in the vector store; (2) **resolve**: RAG over stored data, synthesize answer, run gap analysis (LLM). Returns knowledge and optional child gaps. Lives in `agents/researcher/`; uses `prompts.get_tool_map`, `get_synthesis`, `get_gaps`, and `tools.ToolExecutor`.
- **Writer** — Converts the finished research tree into a structured report (title, summary, content blocks: text, table, chart, code, sources). Supports single-shot or bottom-up (per-node) mode. Lives in `agents/writer/`; uses `prompts.get_write`, `get_write_node`, `get_executive_summary`.

## Directory layout

| Path | Role |
|------|------|
| `orchestrator/orchestrator.py` | `OrchestratorAgent`: `run(query, progress_callback)` runs plan → BFS → write. |
| `planner/planner.py` | `PlannerAgent`: `create_plan(query, num_plan_steps)` → `ResearchPlan`. |
| `researcher/researcher.py` | `ResearcherAgent`: `populate()`, `resolve()`, `is_duplicate()`. |
| `writer/writer.py` | `WriterAgent`: `write()` → `ResearchReport`. |
| `config/` | `ReportConfig`: depth, breadth, writer mode, RAG/dedup settings. |

## Progress callback

The orchestrator (and thus the researcher and writer) accept an optional `progress_callback(event_type: str, data: dict)`. The backend wires this to Redis so the frontend can show live progress over WebSocket.

- Events include: `plan_ready`, `level_start`, `level_complete`, `probe_start`, `tool_call`, `thinking`, `probe_complete`, `writing`, `complete`, `error`.
- The orchestrator uses `_scoped_callback(node, progress_callback)` so events from a given node are tagged with `node_id`.

See `docs/research-streaming-guide.md` (if present) for the full streaming design.

## Dependencies

- **llm** — `BaseLLMClient` (e.g. Gemini).
- **vector_store** — `BaseVectorStore` (Qdrant); used for RAG and deduplication.
- **models** — `ResearchPlan`, `ResearchReport`, `GapAnalysis`, etc.
- **states** — `ResearchNode`, `KnowledgeItem`, `NodeDraft`.
- **prompts** — `get_plan`, `get_tool_map`, `get_synthesis`, `get_gaps`, `get_write`, `get_write_node`, `get_executive_summary`.
- **tools** — `ToolExecutor` and concrete tools (search, loaders).

## Running agents in isolation

Agents are designed to be driven by the backend (e.g. `app.services.research_service.run_research_job`). For local experiments you can instantiate an orchestrator with an LLM client and vector store and call `orchestrator.run(query, progress_callback=...)`; ensure config (e.g. `ReportConfig`), Qdrant, and any tool env vars are set.

See also the main [README.md](../README.md) and [app/README.md](../app/README.md) for how the backend runs research jobs.
