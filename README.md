# Wort — Deep Research Engine

AI-powered research platform with **Chat** and **Deep Research** modes. Users can have conversations (with optional web search), run long-running research jobs that produce structured reports, and ingest documents into a vector store for RAG.

## High-level architecture

```
┌─────────────────┐     HTTP / WebSocket      ┌─────────────────┐
│  wort-frontend  │ ◄──────────────────────► │  app (FastAPI)  │
│  Next.js 16     │                          │  Backend         │
└─────────────────┘                          └────────┬────────┘
                                                       │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                              │                               │
                        ▼                              ▼                               ▼
                 ┌─────────────┐              ┌─────────────┐                 ┌─────────────┐
                 │  Redis      │              │  SQLite /    │                 │  Qdrant     │
                 │  (cache,    │              │  Postgres    │                 │  (vectors)  │
                 │  pub/sub)   │              │  (users,    │                 │              │
                 └─────────────┘              │  sessions)  │                 └─────────────┘
                                              └─────────────┘
```

- **Frontend** — Next.js app: chat, research job UI with live progress, ingest, settings. Talks to backend via `NEXT_PUBLIC_API_URL`.
- **Backend** — FastAPI: auth (Google OAuth + JWT), chat (SSE), research (start job, WebSocket stream, result), history, models, ingest. Uses Redis for context/cache and research progress pub/sub.
- **Agents** — Python pipeline (Planner → Researcher → Writer) run inside the backend as a background task; progress is streamed via Redis → WebSocket.

## Quick start

1. **Prerequisites:** Python 3.12+, Node 20+, Docker (for Redis).
2. **Backend:** Create a venv, install dependencies, set `.env` (see [app/README.md](app/README.md)). Run Redis: `docker-compose up -d`. Start API: `uvicorn app.main:app --reload --port 8000`.
3. **Frontend:** `cd wort-frontend && npm ci && npm run dev` (default: http://localhost:3000).
4. Or use the project script: `./start.sh` (starts Redis, backend, and frontend).

## Repository layout

| Path | Purpose |
|------|---------|
| **[app/](app/)** | Backend: FastAPI app, routers, services, DB, config. Entry: `app/main.py`. |
| **[agents/](agents/)** | Research pipeline: Orchestrator, Planner, Researcher, Writer. Used by the backend research job. |
| **[wort-frontend/](wort-frontend/)** | Next.js frontend: chat, research, ingest, settings. |
| **llm/** | LLM client abstraction (e.g. Gemini). |
| **vector_store/** | Qdrant-based vector store (dense + sparse, RRF). |
| **tools/** | Loaders and search tools used by the Researcher. |
| **models/** | Pydantic/structure models for plans, reports, blocks. |
| **prompts/** | LLM prompts (get_plan, get_gaps, get_write, etc.). |
| **states/** | Research state (ResearchNode, KnowledgeItem, etc.). |

## Where to read next

- **[app/README.md](app/README.md)** — Backend: setup, env vars, routers, services, DB, how chat and research are wired.
- **[agents/README.md](agents/README.md)** — Agents: Orchestrator, Planner, Researcher, Writer; pipeline flow and progress callbacks.
- **[wort-frontend/README.md](wort-frontend/README.md)** — Frontend: structure, pages, API usage, running and building.

Optional docs (if present):

- `docs/hosting-guide.md` — How to host the full stack.
- `docs/research-streaming-guide.md` — How research event streaming works and how to reuse it.

## License

See repository license file.
