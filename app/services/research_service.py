"""
Research job runner — runs the Wort Orchestrator pipeline as a background task.
Research is a capability of chat: jobs are tied to a chat session and the report
is used as context for follow-up messages in that session.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.db.database import async_session
from app.db.models import ResearchJob
from app.services.api_key_service import resolve_api_key_for_model
from app.services.memory_service import MemoryService
from llm.router import get_llm_client
from agents.config.config import ReportConfig
from models import PlanStep


class ResearchProgressEmitter:
    """Emits structured progress events via Redis Pub/Sub."""

    def __init__(self, memory: MemoryService, job_id: str):
        self.memory = memory
        self.job_id = job_id

    async def emit(self, event_type: str, data: dict):
        await self.memory.publish_progress(self.job_id, {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        })

    async def phase_start(self, phase: str, message: str):
        await self.emit("phase_start", {"phase": phase, "message": message})

    async def plan_ready(self, probes: list[str]):
        await self.emit("plan_ready", {"probes": probes, "count": len(probes)})

    async def probe_start(self, probe: str, depth: int, total: int):
        await self.emit("probe_start", {"probe": probe, "depth": depth, "total": total})

    async def tool_call(self, tool: str, query: str):
        await self.emit("tool_call", {"tool": tool, "query": query})

    async def thinking(self, probe: str):
        await self.emit("thinking", {"probe": probe, "message": "Synthesizing findings..."})

    async def probe_complete(self, probe: str, knowledge_items: int = 1):
        await self.emit("probe_complete", {"probe": probe, "knowledge_items": knowledge_items})

    async def probe_skipped(self, probe: str, reason: str):
        await self.emit("probe_skipped", {"probe": probe, "reason": reason})

    async def writing(self):
        await self.emit("writing", {"message": "Composing final report..."})

    async def complete(self, blocks_count: int):
        await self.emit("complete", {"job_id": self.job_id, "blocks_count": blocks_count})

    async def error(self, message: str):
        await self.emit("error", {"message": message})


async def run_research_job(
    job_id: str,
    user_id: str,
    query: str,
    model_id: str,
    config: dict,
    redis_client,
    *,
    refined_plan: list[PlanStep] | None = None,
    user_context: str | None = None,
):
    """
    Background task: runs the full Wort Orchestrator pipeline.
    Emits structured progress events via Redis Pub/Sub.
    Uses its own DB session (do not pass request-scoped session).
    """
    logger.info("Research job started: job_id=%s user_id=%s query=%s", job_id, user_id, query[:80] if query else "")
    memory = MemoryService(redis_client)
    emitter = ResearchProgressEmitter(memory, job_id)

    async with async_session() as db:
        try:
            api_key = await resolve_api_key_for_model(user_id, model_id, memory, db)
            llm = get_llm_client(model_id, api_key=api_key)

            from vector_store.qdrant_store import QdrantStore
            from agents.orchestrator.orchestrator import OrchestratorAgent

            # Use real Qdrant when QDRANT_LOCATION is set (e.g. http://localhost:6333 or cloud URL); else in-memory
            location = (settings.QDRANT_LOCATION or "").strip()
            use_in_memory = location in ("", ":memory:") or location.startswith("path:")
            if use_in_memory:
                vector_store = QdrantStore(in_memory=True)
                logger.info("Research job %s: using in-memory Qdrant (data not persisted)", job_id)
            else:
                vector_store = QdrantStore(
                    url=location,
                    api_key=settings.QDRANT_API_KEY or None,
                )
                logger.info("Research job %s: using remote Qdrant at %s", job_id, vector_store.location)
                connected = await vector_store.check_connection()
                if not connected:
                    logger.warning("Research job %s: Qdrant connection check failed; job may fail", job_id)
                else:
                    logger.info("Research job %s: Qdrant connection verified", job_id)
            collection_name = f"research_{user_id}"

            if not await vector_store.collection_exists(collection_name):
                await vector_store.create_collection(collection_name, dense_dim=384)

            result = await db.execute(select(ResearchJob).where(ResearchJob.id == job_id))
            job = result.scalar_one()
            job.status = "running"
            await db.commit()

            logger.info("Research job %s: emitting phase_start (planning)", job_id)
            await emitter.phase_start("planning", "Breaking down your research query...")

            base_config = ReportConfig.STANDARD()
            conf = ReportConfig(
                name="Custom",
                num_plan_steps=config.get("num_plan_steps", base_config.num_plan_steps),
                max_depth=config.get("max_depth", base_config.max_depth),
                max_probes=config.get("max_probes", base_config.max_probes),
                max_tool_pairs=config.get("max_tool_pairs", base_config.max_tool_pairs),
                dupe_threshold=config.get("dupe_threshold", base_config.dupe_threshold),
                rag_top_k=config.get("rag_top_k", base_config.rag_top_k),
                max_tavily_calls=config.get("max_tavily_calls", base_config.max_tavily_calls),
            )

            orchestrator = OrchestratorAgent(
                llm_client=llm,
                vector_store=vector_store,
                collection_name=collection_name,
                config=conf,
            )

            async def on_progress(event_type: str, data: dict):
                await emitter.emit(event_type, data)

            # refined_plan and user_context are normalized at API boundary (PlanStep list)
            report = await orchestrator.run(
                query,
                progress_callback=on_progress,
                initial_plan=refined_plan,
                user_context=user_context,
            )

            if report is None:
                raise ValueError("Report generation failed (no output from model).")

            await emitter.writing()

            result = await db.execute(select(ResearchJob).where(ResearchJob.id == job_id))
            job = result.scalar_one()
            job.status = "complete"
            job.report_json = report.model_dump() if hasattr(report, "model_dump") else report
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()

            blocks_count = len(report.blocks) if hasattr(report, "blocks") else 0
            await emitter.complete(blocks_count)

        except Exception as e:
            logger.exception("Research job %s failed: %s", job_id, e)
            try:
                result = await db.execute(select(ResearchJob).where(ResearchJob.id == job_id))
                job = result.scalar_one()
                job.status = "failed"
                job.error_message = str(e)
                await db.commit()
            except Exception as db_err:
                logger.warning("Could not update job status in DB: %s", db_err)
            try:
                await emitter.error(str(e))
            except Exception as emit_err:
                logger.warning("Could not emit error to stream (e.g. Redis down): %s", emit_err)
