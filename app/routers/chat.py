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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import async_session, get_db
from app.db.models import User, ChatSession, Message, ResearchJob
from app.core.dependencies import get_current_user
from app.middleware.rate_limit import check_rate_limit
from app.schemas.chat import ChatRequest, ChatCreateResponse, WebSearchDecision
from app.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    ResearchScopeRequest,
    ResearchScopeResponse,
)
from app.services.api_key_service import resolve_api_key_for_model
from app.services.memory_service import MemoryService, get_redis
from app.services.research_service import run_research_job
from agents.planner import PlannerAgent
from llm.router import get_llm_client
from prompts import get_chat_system_prompt_base, get_chat_research_context_suffix, get_web_search_decision

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

SESSION_HISTORY_LIMIT = 50


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

    model_id = await _resolve_model(user_id, req.model_id, db)
    api_key = await resolve_api_key_for_model(user_id, model_id, memory, db)

    if not api_key:
        raise HTTPException(status_code=401, detail="No API key available for this model. Add a key in Settings.")

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

    async def event_stream():
        web_context = ""
        sources = []

        # ── Phase 1: Retrieve research context (needed to decide if we run web for follow-ups) ──────
        yield f"event: status\ndata: {json.dumps({'phase': 'retrieving_context'})}\n\n"
        research_context = await _get_research_context(session_id, db)

        # ── Phase 2: LLM decides if web search is needed and what query to run ──────
        try:
            decision = get_web_search_decision(
                req.message,
                has_research_context=bool(research_context),
                llm_client=llm,
            )
        except Exception as e:
            logger.warning("Web search decision LLM call failed: %s", e)
            decision = WebSearchDecision(use_web=False, search_query=None)

        use_web = decision.use_web and (decision.search_query or "").strip()
        search_query = (decision.search_query or req.message).strip() if use_web else ""

        if use_web and search_query:
            yield f"event: status\ndata: {json.dumps({'phase': 'searching_web'})}\n\n"
            try:
                from tools.tools import ToolExecutor
                executor = ToolExecutor()
                results = await executor.execute("tavily_search", query=search_query)
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
        if web_context:
            system_prompt += f"\n\nRelevant web search results:\n{web_context}"
        if research_context:
            system_prompt += get_chat_research_context_suffix(research_context, has_web_context=bool(web_context))

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

SSE_KEEPALIVE_INTERVAL = 15


@router.post("/research/scope", response_model=ResearchScopeResponse)
async def research_scope(
    req: ResearchScopeRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    Create a research plan and clarifying questions only (no job).
    User can refine the plan and provide answers, then POST /research with refined_plan and user_context.
    """
    memory = MemoryService(redis)
    model_id = req.model_id
    if not model_id:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        model_id = (user.selected_model if user else None) or settings.DEFAULT_MODEL
    api_key = await resolve_api_key_for_model(user_id, model_id, memory, db)
    llm = get_llm_client(model_id, api_key=api_key)
    planner = PlannerAgent(llm_client=llm)
    scoped = await asyncio.to_thread(
        planner.create_scoped_plan,
        req.query,
        num_plan_steps=req.num_plan_steps,
    )
    return ResearchScopeResponse(
        query_type=scoped.query_type.value,
        plan=[{"step_number": s.step_number, "action": s.action, "description": s.description} for s in scoped.plan],
        clarifying_questions=scoped.clarifying_questions or [],
    )


@router.post("/research")
async def start_research(
    req: ResearchRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    Start a deep research job and return an SSE stream.
    First event: {"type": "started", "job_id": "...", "session_id": "..."}.
    Then progress events from the orchestrator; stream ends on complete/error.
    """
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

    config = dict(req.config or {})
    if req.refined_plan is not None:
        config["refined_plan"] = req.refined_plan
    if req.user_context is not None:
        config["user_context"] = req.user_context

    # Normalize refined_plan to PlanStep list at API boundary (orchestrator expects PlanStep only)
    refined_plan_steps = None
    if req.refined_plan:
        from models import PlanStep
        try:
            refined_plan_steps = [PlanStep(**s) for s in req.refined_plan]
        except Exception:
            refined_plan_steps = None

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
            await run_research_job(
                job.id, user_id, req.query, model_id, config, redis,
                refined_plan=refined_plan_steps,
                user_context=req.user_context,
            )
        except Exception as e:
            logger.exception("Research background task failed for job_id=%s: %s", job.id, e)

    asyncio.create_task(_run_with_logging())

    job_id = job.id

    async def event_generator():
        yield f"data: {json.dumps({'type': 'started', 'job_id': job_id, 'session_id': session_id})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"research:{job_id}:progress")

        async def redis_listener():
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode()
                        await queue.put(("message", data))
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(f"research:{job_id}:progress")
                await pubsub.close()

        listener_task = asyncio.create_task(redis_listener())
        try:
            while True:
                try:
                    _, data = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") in ("complete", "error"):
                        break
                except json.JSONDecodeError:
                    pass
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/research/stream/{job_id}")
async def stream_research_progress(
    job_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    SSE stream of progress events for a research job. Subscribe after starting a job
    (e.g. from the research page) to receive plan_ready, probe_start, tool_call, etc.
    Stream ends when type is "complete" or "error".
    """
    result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.id == job_id)
        .where(ResearchJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")

    async def event_generator():
        queue = asyncio.Queue()
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"research:{job_id}:progress")

        async def redis_listener():
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode()
                        await queue.put(("message", data))
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(f"research:{job_id}:progress")
                await pubsub.close()

        listener_task = asyncio.create_task(redis_listener())
        try:
            while True:
                try:
                    _, data = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") in ("complete", "error"):
                        break
                except json.JSONDecodeError:
                    pass
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
