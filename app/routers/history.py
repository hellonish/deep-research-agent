"""
History router — retrieve past chat sessions and research reports.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import ChatSession, Message, ResearchJob
from app.schemas.history import RenameRequest

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/chats")
async def list_chat_sessions(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """Returns the user's chat sessions, most recent first."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = result.scalars().all()

    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": str(s.created_at),
            "updated_at": str(s.updated_at),
        }
        for s in sessions
    ]


@router.get("/chats/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns all messages in a chat session."""
    # Verify ownership
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .where(ChatSession.user_id == user_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "mode": m.mode or "chat",
            "sources": m.sources if m.sources is not None else {},
            "created_at": str(m.created_at),
        }
        for m in messages
    ]

@router.delete("/chats/{session_id}")
async def delete_chat_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a chat session and its associated messages and research jobs."""
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.execute(delete(ResearchJob).where(ResearchJob.session_id == session_id))
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await db.commit()
    return {"status": "success"}

@router.put("/chats/{session_id}")
async def rename_chat_session(
    session_id: str,
    req: RenameRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Renames a chat session."""
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session.title = req.title
    await db.commit()
    return {"status": "success", "title": session.title}

@router.get("/research")
async def list_research_jobs(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """Returns the user's research jobs, most recent first."""
    result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.user_id == user_id)
        .order_by(ResearchJob.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = result.scalars().all()

    return [
        {
            "id": j.id,
            "session_id": j.session_id,
            "query": j.query,
            "status": j.status,
            "model_id": j.model_id,
            "created_at": str(j.created_at),
            "completed_at": str(j.completed_at) if j.completed_at else None,
        }
        for j in jobs
    ]

@router.delete("/research/{job_id}")
async def delete_research_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a research job."""
    result = await db.execute(select(ResearchJob).where(ResearchJob.id == job_id, ResearchJob.user_id == user_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    await db.execute(delete(ResearchJob).where(ResearchJob.id == job_id))
    await db.commit()
    return {"status": "success"}

@router.put("/research/{job_id}")
async def rename_research_job(
    job_id: str,
    req: RenameRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Renames a research job."""
    result = await db.execute(select(ResearchJob).where(ResearchJob.id == job_id, ResearchJob.user_id == user_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job.query = req.title
    await db.commit()
    return {"status": "success", "title": job.query}
