# Wort Frontend — Next.js

Next.js 16 frontend for Wort: chat, deep research (with live progress), document ingest, and settings. Consumes the backend API via `NEXT_PUBLIC_API_URL` and uses JWT (Bearer) for authenticated requests.

## Running the frontend

1. **Environment** — Set `NEXT_PUBLIC_API_URL` if the API is not at `http://localhost:8000` (e.g. `http://localhost:8000` for local dev).

2. **Install and run** — From `wort-frontend/`:
   - `npm ci`
   - `npm run dev` — Dev server (default http://localhost:3000)
   - `npm run build` && `npm run start` — Production build and serve.

## Tech stack

- **Next.js 16** — App Router.
- **React 19** — UI.
- **TypeScript** — Typing.
- **Tailwind CSS v4** — Styling.
- **Framer Motion** — Animations.
- **react-resizable-panels** — Resizable layout (e.g. chat + report).
- **react-markdown**, **remark-gfm**, **rehype-katex**, **remark-math** — Rendered markdown and math.
- **recharts** — Charts in reports.
- **@react-oauth/google** — Google sign-in.

## App structure (App Router)

| Path | Purpose |
|------|---------|
| **app/page.tsx** | Landing / entry. |
| **app/(app)/layout.tsx** | Authenticated app layout (sidebar, etc.). |
| **app/(app)/chat/page.tsx** | Chat list / new chat. |
| **app/(app)/chat/[id]/page.tsx** | Single chat thread (messages, input, optional web/research). |
| **app/(app)/research/new/page.tsx** | Start new research. |
| **app/(app)/research/[id]/page.tsx** | Research job page: live progress (WebSocket), then report + follow-up chat. |
| **app/(app)/ingest/page.tsx** | Document upload and ingest. |
| **app/(app)/settings/page.tsx** | API key, model selection. |

## API usage

- **Base URL** — `NEXT_PUBLIC_API_URL` or default `http://localhost:8000/api`. Requests use `fetch` or `fetchApi` (which attaches JWT from localStorage).
- **Auth** — Token stored in localStorage (e.g. `wort_token`); sent as `Authorization: Bearer <token>`.
- **Chat** — POST to `/chat/stream` with body `{ message, session_id, mode }`; consume SSE for tokens and status.
- **Research** — POST `/chat/research` to start; GET `/chat/research/result/{job_id}` for status/report; WebSocket `/chat/research/stream/{job_id}?token=...` for progress events (JSON per message).
- **History** — GET/POST/DELETE for chats and messages; GET/DELETE/PUT for research jobs.

See `src/lib/api.ts` (or equivalent) for `fetchApi` and `API_BASE`; research page for WebSocket URL construction (ws/wss from current origin, host from API_BASE).

## Key components

- **AuthProvider** — Auth state and token.
- **ChatPanel** — Message list, input, send; handles streamed chat.
- **ResearchEventTree** — Renders research progress log as a tree (by depth).
- **Blocks** — TextBlock, TableBlock, ChartBlock, CodeBlock, SourceListBlock for report content.

## Build and lint

- `npm run build` — Production build.
- `npm run lint` — ESLint.

See the main [README.md](../README.md) for the full project and [app/README.md](../app/README.md) for the backend API.
