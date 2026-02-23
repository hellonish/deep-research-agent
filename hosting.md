---
name: Wort Hosting Guide
overview: Create a single markdown hosting guide (docs/hosting-guide.md) that teaches how to host the Wort stack (FastAPI backend, Next.js frontend, Redis, optional Qdrant, SQLite/Postgres) with clear architecture, env checklist, and step-by-step options (VPS vs PaaS) without actually deploying.
todos: []
isProject: false
---

# Wort Hosting Guide (Learning-Oriented)

## Goal

Produce one markdown file that serves as a **learning-focused hosting guide**: it explains what runs where, what to configure, and how to deploy each piece. The guide will be saved as **[docs/hosting-guide.md](docs/hosting-guide.md)**. No deployment or code changes—only the guide document.

## App architecture (what the guide will describe)

- **Backend**: FastAPI ([app/main.py](app/main.py)), served with `uvicorn app.main:app --port 8000`. Uses async SQLAlchemy ([app/db/database.py](app/db/database.py)) with `DATABASE_URL` (default SQLite), Redis ([app/core/config.py](app/core/config.py) `REDIS_URL`) for memory, rate limiting, and research progress pub/sub, and optional Qdrant ([vector_store/qdrant_store.py](vector_store/qdrant_store.py)) for vectors (in-memory or `QDRANT_LOCATION` + `QDRANT_API_KEY`).
- **Frontend**: Next.js 16 in `wort-frontend/`, build with `npm run build`, run with `npm run start`. API base URL is set via `NEXT_PUBLIC_API_URL` ([wort-frontend/src/lib/api.ts](wort-frontend/src/lib/api.ts)); backend CORS is restricted to `FRONTEND_URL` ([app/core/config.py](app/core/config.py)).
- **External**: Google OAuth (auth), Gemini API (LLM), optional Tavily for web search. No Dockerfile exists today; [docker-compose.yml](docker-compose.yml) only runs Redis.

## Guide structure and content

1. **Overview**
  - Short description of the stack and a simple diagram (mermaid) showing: User → Frontend (Next) → Backend (FastAPI) → Redis / DB / Qdrant / Gemini.
2. **What you need to run**
  - Backend (Python 3.12+, uvicorn), Frontend (Node 20+, Next), Redis (required), Database (SQLite for small scale or Postgres for production), Qdrant (optional: in-memory for dev, Qdrant Cloud or self-hosted for prod), and env vars.
3. **Environment variables checklist**
  - Table or list of every variable from [app/core/config.py](app/core/config.py): `DATABASE_URL`, `REDIS_URL`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `JWT_SECRET`, `ENCRYPTION_KEY`, `GOOGLE_API_KEY`, `DEFAULT_MODEL`, `FRONTEND_URL`, `QDRANT_LOCATION`, `QDRANT_API_KEY`. Frontend: `NEXT_PUBLIC_API_URL`. For each: purpose, example value, and “required in production” note.
4. **Deployment options (learning paths)**
  - **Option A — Single VPS (e.g. Ubuntu on DigitalOcean, Hetzner, Linode)**: One server runs backend, frontend (Node), Redis, and optionally Postgres and Qdrant. Steps: install Python/Node/Redis, clone repo, set env, run backend with uvicorn (and optionally a process manager like systemd or supervisord), build and run Next.js, put Nginx (or Caddy) in front for reverse proxy and SSL. Explain why CORS and `FRONTEND_URL` must match the real frontend origin.
  - **Option B — PaaS (Railway, Render, Fly.io)**: Backend and frontend as separate services; use managed Redis (e.g. Upstash, Railway Redis) and managed Postgres if needed. Explain setting build/start commands and env vars per service, and that `FRONTEND_URL` / `NEXT_PUBLIC_API_URL` must point to the deployed URLs.
  - Brief note on **serverless**: Not a focus (long-running research jobs and WebSockets don’t fit typical serverless limits), but mention why.
5. **Database**
  - SQLite: fine for single-instance, small teams; mention `sqlite+aiosqlite:///./wort.db` and file path/backups.
  - Postgres: for multi-worker or managed DB; give example `DATABASE_URL` and note async driver (e.g. `postgresql+asyncpg://...`).
6. **Redis**
  - Required for chat context, rate limiting, research progress. Include how to get Redis on a VPS (install + systemd) or use a managed service (Upstash, Redis Cloud), and set `REDIS_URL`.
7. **Qdrant (vector store)**
  - In-memory for dev; for production: Qdrant Cloud (set `QDRANT_LOCATION` and `QDRANT_API_KEY`) or self-hosted container and point `QDRANT_LOCATION` at it.
8. **Backend deployment steps**
  - Install dependencies (assume `pip` or `uv` from a requirements file; note that the repo doesn’t currently include `requirements.txt` and suggest generating one from the venv for reproducibility). Run with `uvicorn app.main:app --host 0.0.0.0 --port 8000` (and workers if desired). Health checks: `/health`, `/health/vector`.
9. **Frontend deployment steps**
  - `cd wort-frontend && npm ci && npm run build && npm run start`. Set `NEXT_PUBLIC_API_URL` to the public backend URL so the browser can call the API.
10. **SSL and domain**
  - Use a reverse proxy (Nginx/Caddy) or PaaS TLS. Point domain to the server; set `FRONTEND_URL` and `NEXT_PUBLIC_API_URL` to `https://...` origins.
11. **Post-deploy checks**
  - List: backend `/health` and `/health/vector`, frontend loads and login works, API calls use correct origin (CORS), research job can start and stream progress (Redis + WebSocket).
12. **Optional: Docker (future)**
  - Short “if you want to containerize later” section: idea of Dockerfile for backend, multi-stage build for frontend, and docker-compose for backend + Redis + optional Qdrant/Postgres, without writing full Dockerfiles (learning-only).

## File to create

- **Path**: [docs/hosting-guide.md](docs/hosting-guide.md)
- **Format**: Markdown with headers, code blocks for commands and example env values, one mermaid diagram, and no emojis.

## Out of scope

- Writing a `requirements.txt` or Dockerfile (only suggest generating requirements in the guide).
- Actual deployment or config file edits; the guide is reference-only until the user chooses to deploy.

