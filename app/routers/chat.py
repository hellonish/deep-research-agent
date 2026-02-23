"""
Chat router — conversation and research as one surface.

- Chat: SSE streaming (optional web search), session history, research report as context.
- Research: deep research jobs belong to a chat session; start job, get result, stream progress.

Research is part of chat: jobs are tied to a session and the report grounds follow-up answers.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import async_session, get_db
from app.db.models import User, ChatSession, Message, ResearchJob
from app.core.dependencies import get_current_user, decode_token_user_id
from app.middleware.rate_limit import check_rate_limit
from app.schemas.chat import ChatRequest, ChatCreateResponse
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.api_key_service import resolve_api_key
from app.services.memory_service import MemoryService, get_redis
from app.services.research_service import run_research_job
from llm.router import get_llm_client
from prompts import get_chat_system_prompt_base

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Session history and user context limits
SESSION_HISTORY_LIMIT = 50
RECENT_SESSIONS_LIMIT = 5
RECENT_RESEARCH_LIMIT = 3


# ── Helper: resolve model ID ────────────────────────────────────────

async def _resolve_model(user_id: str, model_id: str | None, db: AsyncSession) -> str:
    """Get the model to use: explicit → user default → system default."""
    if model_id:
        return model_id
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user and user.selected_model:
        return user.selected_model
    return settings.DEFAULT_MODEL


# ── Helper: build LLM messages from history ──────────────────────────

def _build_chat_prompt(history: list[dict], current_message: str) -> str:
    """Build a prompt string from conversation history + current message."""
    parts = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    parts.append(f"USER: {current_message}")
    return "\n\n".join(parts)


# ── Helper: session history from PostgreSQL ──────────────────────────

async def _get_session_history(
    session_id: str, user_id: str, db: AsyncSession, limit: int = SESSION_HISTORY_LIMIT
) -> list[dict]:
    """Load last N messages for this session from DB (user must own session)."""
    if not session_id:
        return []
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return []
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    messages = list(reversed(rows))  # Chronological for prompt
    return [{"role": m.role, "content": m.content or ""} for m in messages]


# ── Helper: long-term user context ──────────────────────────────────

async def _get_user_context(user_id: str, db: AsyncSession) -> str:
    """Build a short user context block: preferences, facts, recent activity."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return ""
    parts = []
    # Preferences / identity from User
    if user.name:
        parts.append(f"User's name: {user.name}")
    if user.selected_model:
        parts.append(f"Preferred model: {user.selected_model}")
    meta = getattr(user, "metadata_", None) or {}
    if isinstance(meta, dict):
        prefs = meta.get("preferences") or {}
        if prefs:
            parts.append("Preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items()))
        facts = meta.get("facts") or []
        if facts:
            parts.append("Known facts about user: " + "; ".join(facts[:10]))
    # Recent activity: last K sessions and research jobs
    sess_result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(RECENT_SESSIONS_LIMIT)
    )
    sessions = sess_result.scalars().all()
    if sessions:
        activity = [f"- {s.title or 'Chat'} (updated {s.updated_at})" for s in sessions if s.title]
        if activity:
            parts.append("Recent sessions:\n" + "\n".join(activity[:RECENT_SESSIONS_LIMIT]))
    job_result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.user_id == user_id)
        .order_by(ResearchJob.created_at.desc())
        .limit(RECENT_RESEARCH_LIMIT)
    )
    jobs = job_result.scalars().all()
    if jobs:
        job_lines = [f"- {j.query[:80]}... ({j.status})" if len(j.query) > 80 else f"- {j.query} ({j.status})" for j in jobs]
        parts.append("Recent research:\n" + "\n".join(job_lines))
    if not parts:
        return ""
    return "User context (use for personalization and continuity):\n" + "\n".join(parts)


# ── Helper: get research report context for follow-ups ───────────────

async def _get_research_context(session_id: str, db: AsyncSession) -> str:
    """If this session has a completed research report, return it as context."""
    if not session_id:
        return ""

    result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.session_id == session_id)
        .where(ResearchJob.status == "complete")
        .order_by(ResearchJob.completed_at.desc())
    )
    job = result.scalar_one_or_none()

    if not job or not job.report_json:
        return ""

    # Extract text content from the report blocks
    report = job.report_json
    parts = [f"Title: {report.get('title', '')}"]
    parts.append(f"Summary: {report.get('summary', '')}")

    for block in report.get("blocks", []):
        if block.get("block_type") == "text" and block.get("markdown"):
            parts.append(block["markdown"])
        elif block.get("block_type") == "source_list" and block.get("sources"):
            parts.append("Sources: " + ", ".join(block["sources"]))

    return "\n\n".join(parts)


# ── Main Chat Endpoint ──────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    Streams an LLM response via Server-Sent Events.

    SSE Events:
      - status:  {phase: "searching_web" | "thinking" | "retrieving_context"}
      - token:   {content: "partial text"}
      - sources: {urls: [...]}
      - done:    {full_content: "..."}
      - error:   {message: "..."}
    """
    memory = MemoryService(redis)
    await check_rate_limit(user_id, redis, "chat", max_per_hour=200)

    # Resolve API key and model
    api_key = await resolve_api_key(user_id, memory, db)
    model_id = await _resolve_model(user_id, req.model_id, db)

    if not api_key:
        raise HTTPException(status_code=401, detail="No API key available")

    llm = get_llm_client(model_id, api_key=api_key)

    # Get or create session
    session_id = req.session_id
    if not session_id:
        session = ChatSession(user_id=user_id, title=req.message[:80])
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session_id = session.id

    # Load session history from PostgreSQL (single source of truth for prompt)
    history = await _get_session_history(session_id, user_id, db, limit=SESSION_HISTORY_LIMIT)
    # Long-term user context (preferences, facts, recent activity)
    user_context = await _get_user_context(user_id, db)

    async def event_stream():
        web_context = ""
        sources = []

        # ── Phase 1: Retrieve research context (needed to decide if we run web for follow-ups) ──────
        yield f"event: status\ndata: {json.dumps({'phase': 'retrieving_context'})}\n\n"
        research_context = await _get_research_context(session_id, db)

        # ── Phase 2: Web search when user chose web mode OR when this is a follow-up (session has report) ──────
        use_web = req.mode == "web" or bool(research_context)
        if use_web:
            yield f"event: status\ndata: {json.dumps({'phase': 'searching_web'})}\n\n"
            try:
                from tools.tools import ToolExecutor
                executor = ToolExecutor()
                results = await executor.execute("tavily_search", query=req.message)
                # ToolExecutor returns list of Document (content, metadata with url)
                chunks = []
                if isinstance(results, list):
                    for r in results[:5]:
                        url = getattr(r, "metadata", None) and (r.metadata.get("url") or r.metadata.get("source")) or getattr(r, "id", "")
                        if url and isinstance(url, str) and url.startswith("http"):
                            sources.append(url)
                    chunks = [getattr(r, "content", str(r))[:500] for r in results[:3]]
                web_context = "\n".join(chunks) if chunks else ""
                if sources:
                    yield f"event: sources\ndata: {json.dumps({'urls': sources})}\n\n"
            except Exception as e:
                yield f"event: sources\ndata: {json.dumps({'urls': [], 'warning': str(e)})}\n\n"

        # ── Phase 3: LLM streaming ──────────────────────────────
        yield f"event: status\ndata: {json.dumps({'phase': 'thinking'})}\n\n"
        system_prompt = get_chat_system_prompt_base()
        if user_context:
            system_prompt += f"\n\n{user_context}"
        if web_context:
            system_prompt += f"\n\nRelevant web search results:\n{web_context}"
        if research_context:
            system_prompt += (
                f"\n\nThe user previously conducted deep research in this session. "
                f"Here is the full report:\n{research_context}\n\n"
                f"Answer follow-up questions using this report as your primary source. "
            )
            if web_context:
                system_prompt += (
                    "You also have fresh web search results above; use them to supplement or update the report when relevant and cite those sources. "
                )
            system_prompt += (
                f"If the answer is NOT in the report or search results, say so and suggest they use the app's Research mode to run a new report—do not claim you will run it yourself."
            )

        prompt = _build_chat_prompt(history, req.message)
        full_response = ""

        try:
            for token in llm.generate_text_stream(prompt=prompt, system_prompt=system_prompt):
                full_response += token
                yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        # ── Phase 4: Persist & finalize ──────────────────────────
        await memory.push_message(user_id, session_id, "user", req.message)
        await memory.push_message(user_id, session_id, "assistant", full_response)

        # Save to PostgreSQL (cold storage)
        db.add(Message(session_id=session_id, role="user", content=req.message, mode=req.mode))
        db.add(Message(session_id=session_id, role="assistant", content=full_response, mode=req.mode, sources=sources or None))
        # Touch session so it appears at top of "recent" in sidebar
        session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        session_result.scalar_one().updated_at = datetime.now(timezone.utc)
        await db.commit()

        yield f"event: done\ndata: {json.dumps({'full_content': full_response, 'session_id': session_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Research (part of chat: jobs live in a session, report used for follow-ups) ──

@router.post("/research", response_model=ResearchResponse)
async def start_research(
    req: ResearchRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """Start a deep research job in the current (or new) chat session. Returns job_id."""
    await check_rate_limit(user_id, redis, "research", max_per_hour=10)

    session_id = req.session_id
    if not session_id:
        session = ChatSession(user_id=user_id, title=f"Research: {req.query[:60]}")
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session_id = session.id

    model_id = req.model_id
    if not model_id:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        model_id = (user.selected_model if user else None) or settings.DEFAULT_MODEL

    config = req.config or {}
    job = ResearchJob(
        user_id=user_id,
        session_id=session_id,
        query=req.query,
        model_id=model_id,
        config_json=config,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    async def _run_with_logging():
        try:
            await run_research_job(job.id, user_id, req.query, model_id, config, redis)
        except Exception as e:
            logger.exception("Research background task failed for job_id=%s: %s", job.id, e)

    asyncio.create_task(_run_with_logging())

    return ResearchResponse(job_id=job.id, session_id=session_id, status="pending")


@router.get("/research/result/{job_id}")
async def get_research_result(
    job_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the research report for a job in the user's session."""
    result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.id == job_id)
        .where(ResearchJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")

    return {
        "job_id": job.id,
        "session_id": job.session_id,
        "status": job.status,
        "query": job.query,
        "report": job.report_json,
        "error": job.error_message,
        "created_at": str(job.created_at),
        "completed_at": str(job.completed_at) if job.completed_at else None,
    }


@router.websocket("/research/stream/{job_id}")
async def research_stream(ws: WebSocket, job_id: str, redis=Depends(get_redis)):
    """WebSocket stream of research progress events for the given job."""
    token = ws.query_params.get("token")
    user_id = decode_token_user_id(token or "")
    if not user_id:
        await ws.close(code=4001)
        return

    async with async_session() as db:
        result = await db.execute(
            select(ResearchJob).where(ResearchJob.id == job_id).where(ResearchJob.user_id == user_id)
        )
        job = result.scalar_one_or_none()
    if not job:
        await ws.close(code=4004)
        return

    await ws.accept()

    # If job already finished (e.g. failed before any event was published), send status immediately
    if job.status == "failed":
        await ws.send_text(json.dumps({"type": "error", "message": job.error_message or "Job failed."}))
        return
    if job.status == "complete":
        blocks = len(job.report_json.get("blocks", [])) if isinstance(job.report_json, dict) else 0
        await ws.send_text(json.dumps({"type": "complete", "job_id": job_id, "blocks_count": blocks}))
        return

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
