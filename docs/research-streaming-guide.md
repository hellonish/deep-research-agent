# Research Event Streaming: Reference and Tutorial

This document explains how the Wort app streams research progress events from backend to frontend, and how to integrate the same pattern into any new application.

---

## Part 1: How Wort Does It (Reference)

### Architecture Overview

```
┌─────────────┐     POST /research      ┌─────────────┐
│   Frontend  │ ──────────────────────► │   Backend   │  Creates job, returns job_id
│   (React)  │                         │  (FastAPI)  │  Starts background task
└─────────────┘                         └──────┬──────┘
       │                                        │
       │  WS /research/stream/{job_id}          │  asyncio.create_task(run_research_job(...))
       │  (with ?token=JWT)                     │
       ▼                                        ▼
┌─────────────┐                         ┌──────────────┐
│  WebSocket  │ ◄─── JSON events ───── │ Redis Pub/Sub│
│  onmessage  │                         │ channel:     │
└─────────────┘                         │ research:    │
       │                                │   {job_id}:  │
       │  setState(logs),                │   progress  │
       │  status complete/error          └──────▲──────┘
       ▼                                        │
┌─────────────┐                         ┌──────┴──────┐
│ Research    │                         │  Background │
│ EventTree   │                         │  task       │
│ (UI)        │                         │  emitter.   │
└─────────────┘                         │  emit(...)  │
                                        └─────────────┘
```

- The **HTTP request** that starts research returns immediately with `job_id`.
- The **long-running work** runs in a background asyncio task (no request scope).
- **Progress** is published to a **Redis channel** per job: `research:{job_id}:progress`.
- A **WebSocket endpoint** subscribes to that channel and forwards every message to the client.
- The **frontend** opens one WebSocket per job, parses JSON events, and updates UI (logs + status).

### Backend: Event Emission

**1. Emitter ([app/services/research_service.py](app/services/research_service.py))**

`ResearchProgressEmitter` wraps Redis publishing. Every event is a JSON object with at least `type` and `timestamp`:

```python
class ResearchProgressEmitter:
    def __init__(self, memory: MemoryService, job_id: str):
        self.memory = memory
        self.job_id = job_id

    async def emit(self, event_type: str, data: dict):
        await self.memory.publish_progress(self.job_id, {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        })
```

Convenience methods map pipeline steps to events: `phase_start`, `plan_ready`, `probe_start`, `tool_call`, `thinking`, `probe_complete`, `writing`, `complete`, `error`.

**2. Redis publish ([app/services/memory_service.py](app/services/memory_service.py))**

```python
async def publish_progress(self, job_id: str, event: dict):
    await self.redis.publish(
        f"research:{job_id}:progress",
        json.dumps(event),
    )
```

**3. Pipeline callback**

The orchestrator (and planner/researcher/writer) accept an optional `progress_callback(event_type, data)`. The research job connects that to the emitter:

```python
# Inside run_research_job()
async def on_progress(event_type: str, data: dict):
    await emitter.emit(event_type, data)
report = await orchestrator.run(query, progress_callback=on_progress)
```

So every `phase_start`, `plan_ready`, `probe_start`, `tool_call`, `thinking`, `probe_complete`, `level_start`, `level_complete`, `writing`, etc., becomes one Redis message, then one WebSocket message.

### Backend: WebSocket Bridge

**Endpoint ([app/routers/chat.py](app/routers/chat.py))**

- Authenticate via query param `token` (JWT), resolve `user_id`.
- Verify the job exists and belongs to that user (DB).
- Accept WebSocket, then subscribe to `research:{job_id}:progress`.
- Loop: for each Redis message, send the same JSON string to the client. Stop on `complete` or `error`, or on WebSocket disconnect.

```python
@router.websocket("/research/stream/{job_id}")
async def research_stream(ws: WebSocket, job_id: str, redis=Depends(get_redis)):
    # ... auth and job ownership check ...
    await ws.accept()
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"research:{job_id}:progress")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await ws.send_text(data)
                parsed = json.loads(data)
                if parsed.get("type") in ("complete", "error"):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"research:{job_id}:progress")
        await pubsub.close()
```

If the job is already `complete` or `failed`, the handler sends one final event (e.g. `complete` or `error`) and returns, so the client can update state without opening Redis.

### Event Types (contract)

| type            | Typical payload (examples)     | Meaning                    |
|-----------------|--------------------------------|----------------------------|
| phase_start     | phase, message                 | New phase (e.g. planning)  |
| plan_ready      | probes, count                  | Plan created, N probes     |
| level_start     | depth, probes, total_in_level  | Starting a BFS level       |
| level_complete  | depth, completed               | Level done                 |
| probe_start     | probe, depth                   | Starting one research probe|
| tool_call       | tool, query                     | Running a tool             |
| thinking        | probe, message                 | LLM synthesizing           |
| probe_complete  | probe, knowledge_items         | Probe done                 |
| writing         | message                        | Writing report             |
| complete        | job_id, blocks_count           | Job finished successfully  |
| error           | message                        | Job failed                 |

All events include `timestamp`. Many include `node_id` when scoped to a research node.

### Frontend: Connecting and Handling Events

**1. Initial status**

Before opening the WebSocket, the page fetches `GET /api/chat/research/result/{jobId}`. If status is already `complete` or `failed`, it skips the WebSocket and shows report or error.

**2. WebSocket URL**

- Protocol: `wss:` if page is HTTPS, else `ws:`.
- Host: same as API (derived from `NEXT_PUBLIC_API_URL`).
- Path: `/chat/research/stream/{jobId}?token={JWT}`.

**3. Message handling**

Each `event.data` is one JSON object. The frontend switches on `type` and updates React state (logs, status). See [wort-frontend/src/app/(app)/research/[id]/page.tsx](wort-frontend/src/app/(app)/research/[id]/page.tsx): `ws.current.onmessage` parses `data.type` and calls `addLog(...)` or sets `status` / `report` / `errorMsg`.

**4. Logs and tree UI**

`addLog(type, text, depth)` appends to a `logs` array. [ResearchEventTree](wort-frontend/src/components/research/ResearchEventTree.tsx) builds a tree from `depth` and renders expandable rows with icons per log type.

---

## Part 2: Tutorial — Integrate This Pattern in Any App

### Requirements

- **Backend**: Any stack that supports WebSockets and Redis (or another pub/sub).
- **Frontend**: Any framework that can open a WebSocket and parse JSON.

### Step 1: Define your event contract

Decide event types and payloads (e.g. `phase_start`, `progress`, `complete`, `error`). Keep a stable schema so the frontend can switch on `type` and read known fields.

Example:

```json
{ "type": "phase_start", "phase": "planning", "message": "Breaking down query...", "timestamp": 1234567890.1 }
{ "type": "complete", "job_id": "uuid", "result_id": "xyz", "timestamp": 1234567890.2 }
```

### Step 2: Run long work in a background task

- Do **not** run the long job in the HTTP request that creates the job.
- Create the job record (e.g. in DB), return `job_id` to the client, then start the job in a **background task** (e.g. `asyncio.create_task`, Celery, or a job queue).
- The background task must be able to publish to a channel that the WebSocket handler can subscribe to (e.g. Redis).

### Step 3: Publish progress to a channel

- Use a **per-job channel** so multiple jobs don't mix: e.g. `job:{job_id}:progress`.
- Serialize each event as JSON and publish one message per event.
- At the end, publish exactly one terminal event: `complete` or `error`, so the client and the WebSocket loop know when to stop.

### Step 4: Expose a WebSocket endpoint

- Accept connections for a given `job_id` (and auth: token or cookie).
- Verify the job exists and the user is allowed to see it.
- Subscribe to the same channel the background task publishes to.
- For each message received from the channel, send it to the WebSocket client unchanged (or re-serialize to JSON if needed).
- On terminal event (`complete` / `error`) or client disconnect, unsubscribe and close the connection.

### Step 5: Frontend — connect and consume

- After creating the job via POST, receive `job_id`.
- Optionally poll or fetch job status once; if already finished, show result and skip WebSocket.
- Build WebSocket URL: same host as your API, path like `/stream/{job_id}`, with auth (e.g. `?token=...`).
- `new WebSocket(url)`, then in `onmessage`: `const data = JSON.parse(event.data)` and handle `data.type` (update UI, set status, fetch final result on `complete`).
- On `close`, if status is not terminal, optionally refetch job status to recover from brief disconnects.

### Step 6: Optional — reconnection and idempotency

- If the client disconnects and reconnects, the WebSocket handler will only see **new** messages (Redis Pub/Sub does not replay). So either:
  - Persist a short history of events per job (e.g. Redis list, capped) and send a replay after subscribe, then continue live; or
  - Rely on the client refetching job status and, if still running, reconnecting and accepting that it may miss some events.

### Summary

| Layer           | Responsibility |
|-----------------|----------------|
| Job creation    | Return `job_id` immediately; start work in background. |
| Background task | Publish JSON events to a per-job Redis (or other) channel; end with `complete` or `error`. |
| WebSocket       | Subscribe to that channel; forward each message to the client; close on terminal event or disconnect. |
| Frontend        | Open WS to `/stream/{job_id}`, parse JSON, update UI by `type`; on `complete`, fetch full result if needed. |

Using this pattern keeps the request/response boundary short, uses a single pub/sub channel per job for clear ordering, and lets any number of clients subscribe to the same job if needed (e.g. multiple tabs).
