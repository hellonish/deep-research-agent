# Wort Backend (`app/`)

## Gist

FastAPI app with **core** (config, auth), **db** (async SQLAlchemy), **routers** (auth, chat, research, history, models, ingest), **schemas** (Pydantic), **services** (API key, crypto, memory, model), and **middleware** (rate limit). Entry point is `main.py`; it creates the app, wires CORS, registers routers under `/api`, and runs DB init on lifespan. Auth is JWT-from-Google-OAuth; chat and research use the same user/session/job models and resolve API keys via Redis → DB → server default.

## Directory layout

- **`core/`** — Config and shared dependencies (auth, DB).
  - `config.py` — Settings from env (DB, Redis, JWT, Gemini, CORS, curated model IDs).
  - `dependencies.py` — `get_current_user`, `create_jwt`.

- **`db/`** — Database and ORM.
  - `database.py` — Async engine, session factory, `get_db`, `init_db`.
  - `models.py` — SQLAlchemy models (User, ChatSession, Message, ResearchJob).

- **`routers/`** — HTTP and WebSocket route handlers.
  - `auth.py` — Google OAuth, JWT.
  - `chat.py` — Chat (SSE stream, optional web search) and research (start job, result, progress WS). Research is part of chat: jobs belong to a session and the report grounds follow-ups.
  - `history.py` — Chat sessions, messages, research jobs list.
  - `models.py` — API key set, model list (curated ∩ user key), set active model.
  - `ingest.py` — File upload and vector store ingest.

- **`schemas/`** — Pydantic request/response models for the API.
  - `chat.py`, `research.py`, `models.py` — Used by the corresponding routers.

- **`services/`** — Business logic and external integrations.
  - `api_key_service.py` — Resolve user API key (cache → DB → server default).
  - `crypto_service.py` — Encrypt/decrypt stored API keys.
  - `memory_service.py` — Redis: chat context, research progress pub/sub, API key cache.
  - `model_service.py` — Gemini model list (curated ∩ API key), key validation.
  - `research_service.py` — Research job runner (orchestrator pipeline, progress emitter); used by chat router.

- **`middleware/`** — Rate limiting, etc.

- **`config.py`**, **`dependencies.py`** (at root) — Re-export from `core/` for backward compatibility.

## Best practices (clean code)

- **Imports** — Prefer `from app.core.config` and `from app.core.dependencies` in new code so config and auth live under one place; keep root `config.py` / `dependencies.py` only as re-exports for legacy call sites.
- **Routers** — Keep routers thin: validate input (schemas), call services or DB, return schemas. Move multi-step or reusable logic into `services/` (e.g. chat context building, research job runner).
- **Schemas** — Put all request/response shapes in `schemas/` and import in routers. Avoid defining Pydantic models inside router files (e.g. `GoogleAuthRequest`, `RenameRequest`, `IngestResponse`).
- **DB in background tasks** — Do not pass the request-scoped `get_db()` session into a fire-and-forget task; the session is closed when the request ends. Inside the background task, create a new session (e.g. `async with async_session() as db`) and use it for the whole run.
- **Response types** — Prefer Pydantic response models for all endpoints so OpenAPI and clients stay consistent (e.g. research result, history list items).
- **Config** — Have `db/database.py` and services import from `app.core.config` so there is a single source of truth.

## Chat and research

- **Chat** — One surface: conversation with optional **web search** (Tavily). Session history and (when present) the latest research report are injected into context for follow-up answers.
- **Research** — Part of chat: a message can start a **deep research job** in the same session. Routes live under `/api/chat/research` (start job, get result, stream progress). The report is stored on the job and used as context for later messages in that session.

## Model selection

- **Curated list** — Only model IDs in `core.config.settings.CURATED_MODEL_IDS` are eligible.
- **Available models** — `GET /api/models/available` returns the **intersection** of (1) models the user’s API key can access and (2) the curated list. The user picks from this list in Settings.
