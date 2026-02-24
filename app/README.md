# App — Backend (FastAPI)

FastAPI backend for Wort: auth, chat, deep research (with live progress), history, model settings, and document ingest. Entry point is `main.py`; it creates the app, wires CORS, registers routers under `/api`, and initializes the DB on lifespan.

## Running the backend

1. **Environment** — Copy or create `.env` in the project root. Key variables (see `core/config.py`):
   - `DATABASE_URL` — Default `sqlite+aiosqlite:///./wort.db`
   - `REDIS_URL` — Default `redis://localhost:6379/0` (required for chat context and research progress)
   - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` — Google OAuth
   - `JWT_SECRET` — Signing secret for JWTs (change in production)
   - `ENCRYPTION_KEY` — Fernet key for encrypting stored API keys
   - `GOOGLE_API_KEY` — Server default Gemini key (optional if users supply their own)
   - `FRONTEND_URL` — CORS origin (e.g. `http://localhost:3000`)
   - `QDRANT_LOCATION`, `QDRANT_API_KEY` — Vector store (optional; in-memory if unset)

2. **Redis** — Start Redis (e.g. `docker-compose up -d`).

3. **Start server** — From project root: `uvicorn app.main:app --reload --port 8000`.

## Directory layout

| Path | Purpose |
|------|---------|
| **main.py** | App factory, lifespan (DB init), CORS, router registration, `/health`, `/health/vector`. |
| **core/** | Config (`settings` from env) and dependencies (`get_current_user`, `create_jwt`, `decode_token_user_id`). |
| **db/** | Async SQLAlchemy engine, session factory, `get_db`, `init_db`; ORM models (User, ChatSession, Message, ResearchJob). |
| **routers/** | HTTP and WebSocket handlers. |
| **schemas/** | Pydantic request/response models. |
| **services/** | Business logic and integrations. |
| **middleware/** | e.g. rate limiting. |

## Routers (API surface)

All under prefix `/api`:

| Router | Prefix | Description |
|--------|--------|-------------|
| **auth** | `/api/auth` | Google OAuth callback, JWT creation. |
| **chat** | `/api/chat` | Chat: POST message, SSE stream; research: POST start job, GET result, WebSocket stream. |
| **history** | `/api/history` | List chats, get/delete messages, list/delete/rename research jobs. |
| **models** | `/api/models` | List available models, set API key, set selected model. |
| **ingest** | `/api/ingest` | File upload and vector store ingest. |

Auth: most endpoints depend on `get_current_user` (JWT in `Authorization: Bearer <token>`).

## Chat and research

- **Chat** — Single conversation surface. User sends a message; optional web search (Tavily). Session history and (if present) the latest research report are injected into context. Response is streamed via SSE.
- **Research** — Part of chat: a message can start a deep research job in the same session. Routes:
  - `POST /api/chat/research` — Start job (body: query, session_id?, model_id?, config?). Returns `job_id`, `session_id`, status.
  - `GET /api/chat/research/result/{job_id}` — Get job status and report.
  - `WS /api/chat/research/stream/{job_id}?token=...` — Live progress events (Redis Pub/Sub → WebSocket).

Research jobs run in a background task (`research_service.run_research_job`); progress is emitted via `ResearchProgressEmitter` → Redis → WebSocket handler.

## Services (selected)

| Service | Role |
|---------|------|
| **api_key_service** | Resolve Gemini API key: Redis cache → DB (decrypt) → server default. |
| **crypto_service** | Encrypt/decrypt API keys at rest (Fernet). |
| **memory_service** | Redis: chat context, research progress publish, API key cache. |
| **research_service** | Run orchestrator pipeline in background; emit progress via Redis. |
| **model_service** | List models (curated ∩ user key), validate key. |

## Config and dependencies

- **Config** — `app.core.config.settings` (or `from app.config import settings`). All settings from env; see `core/config.py` for the full list.
- **Auth** — `get_current_user` for protected routes; `decode_token_user_id` for WebSocket query param. JWT created in auth router after Google OAuth.

## Database

Async SQLAlchemy with `async_session` and `get_db`. Models: User, ChatSession, Message, ResearchJob. Migrations: create tables via `init_db()` on startup; no separate migration runner in this doc.

See the main [README.md](../README.md) for repo overview and [agents/README.md](../agents/README.md) for the research pipeline.
