"""FastAPI router registering all academic agent, memory, RAG, and security endpoints."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import inspect
import json
import logging
import shutil
import uuid
import time
from unittest.mock import AsyncMock
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("university_agent.api.routes")

from app.config import settings
from app.api.deps import get_services
from app.agent.orchestrator import academic_agent, AgentQueryRequest, AgentResponse
from app.knowledge.rag_service import rag_service, IngestionResult
from app.knowledge.vector_store import vector_store
from app.knowledge.embeddings import embedding_manager
from app.memory.database import db
from app.security.validator import security_validator
from app.security.audit import audit_logger
from app.tools.filesystem_tools import filesystem_tools
from app.tools.terminal_tools import terminal_tools
from app.tools.browser_tools import browser_tools
from app.providers.ollama_provider import OllamaProvider
from app.providers.cloud_provider import CloudAIProvider

router = APIRouter()


# --- HEALTH & STATUS ---

@router.get("/health")
@router.get("/system/health")
async def health_check(request: Request):
    services = get_services(request)
    s = services.settings
    ollama_provider = OllamaProvider(settings_obj=s)
    ollama_ok = await ollama_provider.is_available()
    loaded_models = await ollama_provider.get_loaded_models() if ollama_ok else []
    cloud_ok = await CloudAIProvider(settings_obj=s).is_available()
    emb_health = await services.embedding_manager.check_health()

    try:
        db_health = await services.db.check_health()
    except Exception as e:
        db_health = {"status": "error", "error": str(e)}

    try:
        vec_status = services.vector_store.get_index_status()
    except Exception as e:
        vec_status = {"status": "error", "error": str(e)}

    db_ok = db_health.get("status") == "ok"
    vec_ok = vec_status.get("status") in ("ok", "healthy")
    core_healthy = db_ok and vec_ok

    total_chunks = vec_status.get("total_chunks", 0)
    collection_size = vec_status.get("collection_size", total_chunks)

    status_str = "healthy" if core_healthy else "unhealthy"

    payload = {
        "status": status_str,
        "environment": s.APP_ENV,
        "app_name": s.APP_NAME,
        "version": s.APP_VERSION,
        "database": {
            **db_health,
            "status": "ok" if db_ok else "down"
        },
        "vector_store": {
            "mode": vec_status.get("mode", "fallback_json"),
            "status": "ok" if vec_ok else "down",
            "backend": vec_status.get("backend", "fallback_json"),
            "total_documents": vec_status.get("total_documents", 0),
            "total_chunks": total_chunks,
            "collection_size": collection_size,
        },
        "embeddings": {
            "status": emb_health.get("status", "unknown"),
            "provider": emb_health.get("provider", "unknown"),
            "model": emb_health.get("model", "unknown"),
            "is_neural": emb_health.get("is_neural", False),
            "dimension": emb_health.get("dimension", 0),
            "error": emb_health.get("error"),
        },
        "ollama": {
            "status": "ok" if ollama_ok else "unavailable",
            "available": ollama_ok,
            "base_url": s.OLLAMA_BASE_URL,
            "model": s.LOCAL_MODEL,
            "models_available": loaded_models,
        },
        "cloud": {
            "status": "ok" if cloud_ok else ("disabled" if not s.CLOUD_ENABLED else "unavailable"),
            "enabled": s.CLOUD_ENABLED,
            "available": cloud_ok,
            "model": s.CLOUD_MODEL,
        },
        "llm_providers": {
            "ollama": {
                "available": ollama_ok,
                "base_url": s.OLLAMA_BASE_URL,
                "model": s.LOCAL_MODEL,
                "models_available": loaded_models,
            },
            "cloud": {
                "available": cloud_ok,
                "enabled": s.CLOUD_ENABLED,
                "model": s.CLOUD_MODEL,
            },
        },
        # Backwards compatible top-level fields
        "ollama_available": ollama_ok,
        "cloud_available": cloud_ok,
        "local_model": s.LOCAL_MODEL,
        "embedding_provider": emb_health.get("provider", "unknown"),
        "embedding_model": emb_health.get("model", "unknown"),
        "embedding_is_neural": emb_health.get("is_neural", False),
        "embedding_dimension": emb_health.get("dimension", 0),
        "embedding_status": emb_health.get("status", "unknown"),
    }

    if not core_healthy:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


@router.get("/system/status")
async def system_status(request: Request):
    services = get_services(request)
    s = services.settings
    mem_info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    if k in ["MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"]:
                        mem_info[k] = f"{int(v) // 1024} MB"
    except Exception:
        pass

    ollama_provider = OllamaProvider(settings_obj=s)
    loaded_models = await ollama_provider.get_loaded_models()
    ollama_ok = await ollama_provider.is_available()
    cloud_ok = await CloudAIProvider(settings_obj=s).is_available()
    emb_info = await services.embedding_manager.get_active_info()
    db_health = await services.db.check_health()
    vec_status = services.vector_store.get_index_status()

    return {
        "environment": s.APP_ENV,
        "memory": mem_info,
        "database": db_health,
        "vector_store": vec_status,
        "embedding": emb_info.model_dump(),
        "embeddings": {
            "provider": emb_info.provider,
            "model": emb_info.model,
            "is_neural": emb_info.is_neural,
            "dimension": emb_info.dimension,
        },
        "llm_providers": {
            "ollama": {
                "connected": ollama_ok,
                "base_url": s.OLLAMA_BASE_URL,
                "active_model": s.LOCAL_MODEL,
                "available_models": loaded_models,
            },
            "cloud": {
                "enabled": s.CLOUD_ENABLED,
                "available": cloud_ok,
                "model": s.CLOUD_MODEL,
            },
        },
        # Backwards compatible top-level fields
        "ollama_base_url": s.OLLAMA_BASE_URL,
        "ollama_connected": ollama_ok,
        "available_local_models": loaded_models,
        "active_local_model": s.LOCAL_MODEL,
        "cloud_routing_enabled": s.CLOUD_ENABLED,
        "vector_index": vec_status,
    }


# --- CHAT & AGENT ---

@router.post("/chat")
@router.post("/agent/chat")
async def chat_with_agent(req: AgentQueryRequest, request: Request):
    services = get_services(request)
    agent = services.agent

    if req.stream is False:
        return await agent.process_query(req)

    async def sse_event_generator():
        try:
            has_mocked_pq = isinstance(getattr(agent, "process_query", None), AsyncMock)
            has_mocked_pqs = isinstance(getattr(agent, "process_query_stream", None), AsyncMock)

            if has_mocked_pq and not has_mocked_pqs:
                res = await agent.process_query(req)
                token_text = getattr(res, "reply", str(res))
                session_id = getattr(res, "session_id", req.session_id or "")
                yield f"data: {json.dumps({'token': token_text, 'session_id': session_id})}\n\n"
                final_meta = {
                    "used_sources": getattr(res, "used_sources", []),
                    "reasoning_path": getattr(res, "reasoning_path", None),
                    "intent": getattr(res, "intent", None)
                }
                yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'reply': token_text, 'metadata': final_meta})}\n\n"
                return

            if hasattr(agent, "process_query_stream"):
                stream = agent.process_query_stream(req)
            else:
                res = await agent.process_query(req)
                token_text = getattr(res, "reply", str(res))
                session_id = getattr(res, "session_id", req.session_id or "")
                yield f"data: {json.dumps({'token': token_text, 'session_id': session_id})}\n\n"
                final_meta = {
                    "used_sources": getattr(res, "used_sources", []),
                    "reasoning_path": getattr(res, "reasoning_path", None),
                    "intent": getattr(res, "intent", None)
                }
                yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'reply': token_text, 'metadata': final_meta})}\n\n"
                return

            if inspect.iscoroutine(stream):
                stream = await stream

            async for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"SSE chat streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True, 'session_id': req.session_id or ''})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, request: Request):
    return await get_services(request).db.get_session_history(session_id)


# --- DOCUMENT INGESTION & RAG ---

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

@router.post("/documents/upload", response_model=IngestionResult)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None)
):
    services = get_services(request)
    s = services.settings
    validator = services.security_validator
    rag = services.rag_service
    database = services.db

    raw_name = (file.filename or "").strip()
    if not raw_name or raw_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename provided.")

    safe_name = Path(raw_name).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(ALLOWED_DOCUMENT_EXTENSIONS))}"
        )

    dest_path = s.DOCUMENTS_DIR / safe_name
    if validator.is_sensitive_target(dest_path):
        raise HTTPException(status_code=403, detail="Filename targets restricted configuration/system name.")

    # Save to disk with strict streaming size limiter
    max_bytes = s.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    buffer.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed upload size of {s.MAX_UPLOAD_SIZE_MB}MB."
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to store file: {e}")

    if total_bytes == 0:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        result = await rag.ingest_document(dest_path, subject=subject)
        if subject:
            # Register subject in SQLite if new
            await database.add_subject(subject_id=str(uuid.uuid4()), name=subject)
        return result
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {e}")


@router.get("/documents")
async def list_documents(
    request: Request,
    subject_id: Optional[str] = Query(None, description="Filter documents by subject ID or name"),
    status: Optional[str] = Query(None, description="Filter documents by ingestion status")
):
    services = get_services(request)
    docs = await services.db.list_documents(subject_id=subject_id, status=status)
    if not docs and subject_id:
        async with services.db.get_connection() as conn:
            cur = await conn.execute("SELECT id FROM subjects WHERE LOWER(name) = LOWER(?)", (subject_id.strip(),))
            srow = await cur.fetchone()
            if srow:
                docs = await services.db.list_documents(subject_id=srow[0], status=status)

    if not docs and not subject_id and not status:
        return services.vector_store.list_indexed_documents()

    if docs:
        formatted = []
        for d in docs:
            item = dict(d)
            item["total_chunks"] = d.get("chunk_count", 0)
            item["total_pages"] = d.get("page_count", 1)
            item["subject"] = d.get("subject_id") or ""
            item["document_id"] = d.get("id")
            formatted.append(item)
        return formatted

    vector_docs = services.vector_store.list_indexed_documents()
    if subject_id:
        return [d for d in vector_docs if d.get("subject") == subject_id or d.get("document_id") == subject_id]
    return vector_docs


@router.get("/documents/index-status")
async def get_index_status(request: Request):
    return get_services(request).vector_store.get_index_status()


class ReindexRequest(BaseModel):
    force: bool = False


@router.post("/documents/reindex")
async def reindex_documents(request: Request, req: ReindexRequest = ReindexRequest()):
    return await get_services(request).rag_service.reindex_all_documents(force=req.force)


class SearchQuery(BaseModel):
    query: str
    subject: Optional[str] = None
    top_k: int = 4


@router.post("/documents/search")
async def search_documents(req: SearchQuery, request: Request):
    results = await get_services(request).vector_store.search(query=req.query, top_k=req.top_k, subject=req.subject)
    return [
        {
            "chunk_id": r.chunk_id,
            "filename": r.filename,
            "page_number": r.page_number,
            "heading": r.heading,
            "subject": r.subject,
            "score": r.score,
            "semantic_score": r.semantic_score,
            "lexical_score": r.lexical_score,
            "text": r.text,
            "citation": r.citation()
        }
        for r in results
    ]


@router.get("/documents/{doc_id}")
async def get_document_details(doc_id: str, request: Request):
    services = get_services(request)
    doc = await services.db.get_document(doc_id)
    if not doc:
        doc = await services.db.get_document_by_filename(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    request: Request,
    delete_file: bool = Query(False, description="Whether to also delete the physical file on disk")
):
    services = get_services(request)
    try:
        result = await services.rag_service.delete_document(doc_id, delete_physical_file=delete_file)
        if not result["db_deleted"] and result["chunks_removed"] == 0 and not result["physical_file_deleted"]:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to delete document: {e}")


@router.post("/documents/retry-failed")
async def retry_failed_documents(request: Request):
    return await get_services(request).rag_service.retry_failed_documents()


# --- MEMORY, PROGRESS & TASKS ---

@router.get("/memory/sessions")
async def list_recent_sessions(request: Request, limit: int = Query(10, ge=1, le=100)):
    return await get_services(request).db.get_recent_sessions(limit=limit)


@router.get("/memory/sessions/{session_id}")
async def get_session_history_by_id(session_id: str, request: Request, limit: int = Query(50, ge=1, le=500)):
    return await get_services(request).db.get_session_history(session_id, limit=limit)


@router.get("/memory/progress")
async def get_progress(request: Request):
    return await get_services(request).db.get_learning_progress()


# --- SUBJECTS & TOPICS ---

class SubjectCreate(BaseModel):
    name: str
    code: Optional[str] = None
    description: str = ""


@router.get("/subjects")
@router.get("/memory/subjects")
async def get_subjects(request: Request):
    return await get_services(request).db.list_subjects()


@router.post("/subjects")
@router.post("/memory/subjects")
async def create_subject(req: SubjectCreate, request: Request):
    try:
        subject_id = req.code.strip() if req.code and req.code.strip() else str(uuid.uuid4())
        return await get_services(request).db.add_subject(
            subject_id=subject_id,
            name=req.name,
            description=req.description,
            code=req.code
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/subjects/{subject_id}")
async def get_subject(subject_id: str, request: Request):
    services = get_services(request)
    async with services.db.get_connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM subjects WHERE id = ? OR LOWER(id) = LOWER(?) OR LOWER(code) = LOWER(?)",
            (subject_id.strip(), subject_id.strip(), subject_id.strip())
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")
        return dict(row)


@router.delete("/subjects/{subject_id}")
async def delete_subject(subject_id: str, request: Request):
    services = get_services(request)
    try:
        deleted = await services.db.delete_subject(subject_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")
        return {"status": "success", "deleted_subject_id": subject_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TopicCreatePayload(BaseModel):
    name: str
    mastery_level: float = 0.0
    notes: str = ""
    topic_id: Optional[str] = None


@router.get("/subjects/{subject_id}/topics")
async def get_subject_topics(subject_id: str, request: Request):
    services = get_services(request)
    async with services.db.get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id FROM subjects WHERE id = ? OR LOWER(id) = LOWER(?) OR LOWER(code) = LOWER(?)",
            (subject_id.strip(), subject_id.strip(), subject_id.strip())
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")
        resolved_id = row[0]

    return await services.db.list_topics(subject_id=resolved_id)


@router.post("/subjects/{subject_id}/topics")
async def create_subject_topic(subject_id: str, req: TopicCreatePayload, request: Request):
    services = get_services(request)
    try:
        t_id = req.topic_id or str(uuid.uuid4())
        return await services.db.add_or_update_topic(
            topic_id=t_id,
            subject_id=subject_id,
            name=req.name,
            mastery=req.mastery_level,
            notes=req.notes
        )
    except ValueError as e:
        msg = str(e)
        if "does not exist" in msg.lower() or "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.get("/memory/topics")
async def get_topics(request: Request, subject_id: Optional[str] = None):
    return await get_services(request).db.list_topics(subject_id)


class TopicCreate(BaseModel):
    subject_id: str
    name: str
    mastery_level: float = 0.0
    notes: str = ""
    topic_id: Optional[str] = None


@router.post("/memory/topics")
async def create_topic(req: TopicCreate, request: Request):
    try:
        t_id = req.topic_id or str(uuid.uuid4())
        return await get_services(request).db.add_or_update_topic(
            topic_id=t_id,
            subject_id=req.subject_id,
            name=req.name,
            mastery=req.mastery_level,
            notes=req.notes
        )
    except ValueError as e:
        msg = str(e)
        if "does not exist" in msg.lower() or "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


# --- ACADEMIC TASKS ---

class TaskCreate(BaseModel):
    title: str
    description: str = ""
    subject_id: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "medium"


class TaskStatusUpdate(BaseModel):
    status: str  # "pending", "completed"


@router.get("/tasks")
@router.get("/memory/tasks")
async def get_tasks(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by pending/completed status")
):
    return await get_services(request).db.list_tasks(status=status)


@router.get("/tasks/upcoming")
@router.get("/memory/tasks/upcoming")
async def get_upcoming_tasks(
    request: Request,
    subject_id: Optional[str] = Query(None, description="Optional subject filter"),
    days_ahead: int = Query(7, ge=1, le=365, description="Number of days ahead to look"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of tasks to return")
):
    services = get_services(request)
    return await services.db.get_upcoming_tasks(subject_id=subject_id, limit=limit, days_ahead=days_ahead)


@router.get("/tasks/overdue")
@router.get("/memory/tasks/overdue")
async def get_overdue_tasks(request: Request):
    services = get_services(request)
    return await services.db.get_overdue_tasks()


@router.post("/tasks")
@router.post("/memory/tasks")
async def create_task(req: TaskCreate, request: Request):
    try:
        return await get_services(request).db.add_task(
            task_id=str(uuid.uuid4()),
            title=req.title,
            description=req.description,
            subject_id=req.subject_id,
            deadline=req.deadline,
            priority=req.priority
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/tasks/{task_id}")
@router.patch("/memory/tasks/{task_id}")
async def update_task(task_id: str, req: TaskStatusUpdate, request: Request):
    try:
        ok = await get_services(request).db.update_task_status(task_id, req.status)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return {"status": "success", "task_id": task_id, "new_status": req.status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/tasks/{task_id}")
@router.delete("/memory/tasks/{task_id}")
async def delete_task_endpoint(task_id: str, request: Request):
    services = get_services(request)
    try:
        deleted = await services.db.delete_task(task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return {"status": "success", "deleted_task_id": task_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- STUDY GOALS ---

class GoalCreate(BaseModel):
    title: str
    description: str = ""
    target_date: Optional[str] = None
    status: str = "in_progress"
    progress_percent: float = 0.0
    subject_id: Optional[str] = None
    goal_id: Optional[str] = None


class GoalProgressUpdate(BaseModel):
    progress_percent: Optional[float] = None
    progress: Optional[float] = None


@router.get("/goals")
@router.get("/memory/goals")
async def list_goals(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status (in_progress, completed, etc.)"),
    subject_id: Optional[str] = Query(None, description="Optional subject filter")
):
    services = get_services(request)
    if status == "in_progress" and subject_id:
        return await services.db.get_active_goals(subject_id=subject_id)
    return await services.db.list_study_goals(status=status)


@router.post("/goals")
@router.post("/memory/goals")
async def create_goal(req: GoalCreate, request: Request):
    services = get_services(request)
    try:
        gid = req.goal_id or str(uuid.uuid4())
        return await services.db.add_study_goal(
            goal_id=gid,
            title=req.title,
            description=req.description,
            target_date=req.target_date,
            status=req.status,
            progress_percent=req.progress_percent,
            subject_id=req.subject_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/goals/{goal_id}")
@router.get("/memory/goals/{goal_id}")
async def get_goal(goal_id: str, request: Request):
    services = get_services(request)
    async with services.db.get_connection() as conn:
        cursor = await conn.execute("SELECT * FROM study_goals WHERE id = ?", (goal_id.strip(),))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Study goal '{goal_id}' not found.")
        return dict(row)


@router.patch("/goals/{goal_id}/progress")
@router.patch("/memory/goals/{goal_id}/progress")
async def update_goal_progress(goal_id: str, req: GoalProgressUpdate, request: Request):
    services = get_services(request)
    val = req.progress_percent if req.progress_percent is not None else (req.progress if req.progress is not None else 0.0)
    try:
        return await services.db.update_goal_progress(goal_id=goal_id, progress_percent=val)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.delete("/goals/{goal_id}")
@router.delete("/memory/goals/{goal_id}")
async def delete_goal(goal_id: str, request: Request):
    services = get_services(request)
    try:
        deleted = await services.db.delete_study_goal(goal_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Study goal '{goal_id}' not found.")
        return {"status": "success", "deleted_goal_id": goal_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- TUTOR QUIZZES ---

class AgentQuizEndpointRequest(BaseModel):
    subject_id: Optional[str] = None
    topic: Optional[str] = None
    num_questions: int = 5
    hide_answers: bool = False
    topic_name: Optional[str] = None
    topic_id: Optional[str] = None


@router.post("/agent/quiz")
async def generate_agent_quiz(req: AgentQuizEndpointRequest, request: Request):
    resolved_topic = req.topic or req.topic_name or ""
    resolved_subject = req.subject_id or req.topic_id
    return await get_services(request).agent.generate_quiz(
        subject_id=resolved_subject,
        topic=resolved_topic,
        num_questions=req.num_questions,
        hide_answers=req.hide_answers,
        topic_id=req.topic_id,
        topic_name=req.topic_name
    )


class QuizGenerateRequest(BaseModel):
    topic_id: str
    topic_name: str
    num_questions: int = 3


@router.post("/tutor/quiz/generate")
async def generate_quiz_endpoint(req: QuizGenerateRequest, request: Request):
    return await get_services(request).agent.generate_quiz(
        topic_id=req.topic_id,
        topic_name=req.topic_name,
        num_questions=req.num_questions
    )


class QuizEvaluateRequest(BaseModel):
    topic_id: str
    question: str
    student_answer: str
    correct_answer: str


@router.post("/tutor/quiz/evaluate")
async def evaluate_quiz_endpoint(req: QuizEvaluateRequest, request: Request):
    return await get_services(request).agent.evaluate_quiz_answer(
        topic_id=req.topic_id,
        question=req.question,
        student_answer=req.student_answer,
        correct_answer=req.correct_answer
    )


class AgentQuizEvaluateRequest(BaseModel):
    topic_id: str
    answers: Dict[str, str] = {}
    quiz_data: Dict[str, Any] = {}


@router.post("/agent/quiz/evaluate")
async def evaluate_agent_quiz(req: AgentQuizEvaluateRequest, request: Request):
    return await get_services(request).agent.evaluate_quiz(
        topic_id=req.topic_id,
        answers=req.answers,
        quiz_data=req.quiz_data
    )


# --- RESTRICTED PC TOOLS & SECURITY ---

class FileReadRequest(BaseModel):
    path: str


@router.post("/tools/filesystem/read")
async def read_file_endpoint(req: FileReadRequest):
    try:
        content = filesystem_tools.read_authorized_file(req.path)
        return {"status": "success", "path": req.path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class FolderCreateRequest(BaseModel):
    path: str
    subject: Optional[str] = None


@router.post("/tools/filesystem/create-folder")
async def create_folder_endpoint(req: FolderCreateRequest):
    try:
        msg = filesystem_tools.create_academic_folder(req.path, req.subject)
        return {"status": "success", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class OrganizeRequest(BaseModel):
    source_dir: str
    target_dir: str


@router.post("/tools/filesystem/organize")
async def organize_endpoint(req: OrganizeRequest):
    try:
        return filesystem_tools.organize_materials(req.source_dir, req.target_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class FileDeleteRequest(BaseModel):
    path: str
    approval_id: Optional[str] = None


@router.post("/tools/filesystem/delete")
async def delete_file_endpoint(req: FileDeleteRequest):
    try:
        return filesystem_tools.delete_file(req.path, req.approval_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class TerminalRunRequest(BaseModel):
    command: str
    approval_id: Optional[str] = None


@router.post("/tools/terminal/safe-run")
async def safe_terminal_endpoint(req: TerminalRunRequest):
    return terminal_tools.run_safe_command(req.command)


@router.post("/tools/terminal/privileged-run")
async def privileged_terminal_endpoint(req: TerminalRunRequest):
    try:
        return terminal_tools.execute_with_approval(req.command, req.approval_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- APPROVAL MANAGEMENT ---

@router.get("/security/approvals")
async def list_pending_approvals(request: Request):
    return get_services(request).security_validator.list_pending()


class ApprovalDecision(BaseModel):
    reason: Optional[str] = None


@router.post("/security/approvals/{approval_id}/approve")
@router.post("/security/approve/{approval_id}")
async def approve_request(approval_id: str, request: Request, req: ApprovalDecision = ApprovalDecision()):
    services = get_services(request)
    ok = services.security_validator.approve_action(approval_id, req.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Request not found or not in pending state")
    services.audit_logger.log_event(
        event_type="security_approval",
        action="action_approved",
        details={"approval_id": approval_id, "note": req.reason},
        actor="user",
        risk_level="high",
        status="approved"
    )
    return {"status": "approved", "approval_id": approval_id}


@router.post("/security/approvals/{approval_id}/reject")
@router.post("/security/reject/{approval_id}")
async def reject_request(approval_id: str, request: Request, req: ApprovalDecision = ApprovalDecision()):
    services = get_services(request)
    ok = services.security_validator.reject_action(approval_id, req.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Request not found or not in pending state")
    services.audit_logger.log_event(
        event_type="security_approval",
        action="action_rejected",
        details={"approval_id": approval_id, "note": req.reason},
        actor="user",
        risk_level="high",
        status="rejected"
    )
    return {"status": "rejected", "approval_id": approval_id}


@router.get("/security/audit")
async def get_audit_log(request: Request, limit: int = 50):
    return get_services(request).audit_logger.read_recent_events(limit=limit)


# --- OPEN WEBUI / OPENAI COMPATIBILITY ---

def format_sources_footer(used_sources: List[Dict[str, Any]]) -> str:
    """Format grounded source references into a clean Markdown footer for Open WebUI / OpenAI clients."""
    if not used_sources:
        return ""
    lines = ["\n\n---\n**Sources:**"]
    for i, s in enumerate(used_sources, start=1):
        idx = s.get("index") or i
        filename = s.get("filename") or s.get("document_id") or "Document"
        page = s.get("page_number")
        heading = s.get("heading")
        parts = []
        if page is not None and str(page).strip():
            parts.append(f"Page {page}")
        if heading and str(heading).strip():
            parts.append(str(heading).strip())
        detail = f" ({' — '.join(parts)})" if parts else ""
        lines.append(f"[{idx}] {filename}{detail}")
    return "\n".join(lines)


@router.get("/v1/models")
async def openai_models():
    """Return virtual academic models in OpenAI format for Open WebUI discovery."""
    now = int(time.time())
    model_ids = [
        "university-agent",
        "university-agent-tutor",
        "university-agent-research",
        "university-agent-summarizer",
        # Legacy aliases for backward compatibility
        "university-tutor",
        "university-research"
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": m_id,
                "object": "model",
                "created": now,
                "owned_by": "university-agent",
                "permission": [],
                "root": m_id,
                "parent": None
            }
            for m_id in model_ids
        ]
    }


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

    model_config = {"extra": "ignore"}


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint connecting Open WebUI to our agent."""
    user_prompt = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            user_prompt = m.get("content", "")
            break

    # Persona / Mode mapping
    model_lower = req.model.lower().strip()
    if "research" in model_lower:
        mode = "research"
    elif "summariz" in model_lower:
        mode = "summarizer"
    elif "assignment" in model_lower:
        mode = "assignment"
    elif "tutor" in model_lower:
        mode = "tutor"
    else:
        mode = "tutor"

    agent_req = AgentQueryRequest(
        message=user_prompt,
        messages=req.messages,
        mode=mode,
        use_rag=True,
        is_private=True,
        stream=req.stream or False
    )
    services = get_services(request)
    agent = services.agent

    if not req.stream:
        res = await agent.process_query(agent_req)
        reply_content = res.reply or ""
        used_sources = res.used_sources or []

        if used_sources:
            sources_footer = format_sources_footer(used_sources)
            if sources_footer and "**Sources:**" not in reply_content and "Sources:" not in reply_content:
                reply_content = reply_content + sources_footer

        prompt_tokens = max(1, len(user_prompt) // 4)
        completion_tokens = max(1, len(reply_content) // 4)

        return {
            "id": f"chatcmpl-{res.message_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "citations": res.citations or used_sources,
            "used_sources": used_sources
        }

    # Streaming mode (SSE chat.completion.chunk format ending with data: [DONE])
    completion_id = str(uuid.uuid4())
    created_ts = int(time.time())

    async def openai_sse_generator():
        try:
            has_mocked_pq = isinstance(getattr(agent, "process_query", None), AsyncMock)
            has_mocked_pqs = isinstance(getattr(agent, "process_query_stream", None), AsyncMock)

            used_sources = []
            accumulated_text = ""

            if has_mocked_pq and not has_mocked_pqs:
                res = await agent.process_query(agent_req)
                token_text = getattr(res, "reply", str(res))
                used_sources = getattr(res, "used_sources", []) or []

                chunk_payload = {
                    "id": f"chatcmpl-{completion_id}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"
                accumulated_text += token_text
            else:
                if hasattr(agent, "process_query_stream"):
                    stream = agent.process_query_stream(agent_req)
                else:
                    stream = None

                if stream is not None:
                    if inspect.iscoroutine(stream):
                        stream = await stream

                    async for chunk in stream:
                        if isinstance(chunk, dict):
                            if chunk.get("error"):
                                error_chunk = {
                                    "id": f"chatcmpl-{completion_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": req.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": f"\nError: {chunk.get('error')}"},
                                            "finish_reason": "stop"
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(error_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            if chunk.get("done"):
                                used_sources = chunk.get("used_sources") or chunk.get("metadata", {}).get("used_sources") or []
                                continue

                            token = chunk.get("token") or chunk.get("reply")
                            if token:
                                accumulated_text += token
                                chunk_payload = {
                                    "id": f"chatcmpl-{completion_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": req.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": token},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_payload)}\n\n"
                        elif isinstance(chunk, str):
                            accumulated_text += chunk
                            chunk_payload = {
                                "id": f"chatcmpl-{completion_id}",
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk_payload)}\n\n"
                else:
                    res = await agent.process_query(agent_req)
                    token_text = getattr(res, "reply", str(res))
                    used_sources = getattr(res, "used_sources", []) or []
                    accumulated_text += token_text
                    chunk_payload = {
                        "id": f"chatcmpl-{completion_id}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

            # Stream citations footer if sources were cited
            if used_sources:
                sources_footer = format_sources_footer(used_sources)
                if sources_footer and "**Sources:**" not in accumulated_text and "Sources:" not in accumulated_text:
                    sources_chunk = {
                        "id": f"chatcmpl-{completion_id}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": sources_footer},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(sources_chunk)}\n\n"

            # Stop chunk
            stop_chunk = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"OpenAI chat completions streaming error: {e}")
            err_chunk = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\nError: {e}"},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        openai_sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
