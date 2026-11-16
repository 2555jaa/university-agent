"""FastAPI router registering all academic agent, memory, RAG, and security endpoints."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import uuid
import time
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

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
async def health_check(request: Request):
    services = get_services(request)
    s = services.settings
    ollama_ok = await OllamaProvider(settings_obj=s).is_available()
    cloud_ok = await CloudAIProvider(settings_obj=s).is_available()
    emb_info = await services.embedding_manager.get_active_info()
    return {
        "status": "healthy",
        "app_name": s.APP_NAME,
        "version": s.APP_VERSION,
        "ollama_available": ollama_ok,
        "cloud_available": cloud_ok,
        "local_model": s.LOCAL_MODEL,
        "embedding_provider": emb_info.provider,
        "embedding_model": emb_info.model,
        "embedding_is_neural": emb_info.is_neural,
        "embedding_dimension": emb_info.dimension
    }


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
    emb_info = await services.embedding_manager.get_active_info()

    return {
        "memory": mem_info,
        "ollama_base_url": s.OLLAMA_BASE_URL,
        "ollama_connected": await ollama_provider.is_available(),
        "available_local_models": loaded_models,
        "active_local_model": s.LOCAL_MODEL,
        "cloud_routing_enabled": s.CLOUD_ENABLED,
        "embedding": emb_info.model_dump(),
        "vector_index": services.vector_store.get_index_status()
    }


# --- CHAT & AGENT ---

@router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(req: AgentQueryRequest, request: Request):
    return await get_services(request).agent.process_query(req)


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
async def list_documents(request: Request):
    return get_services(request).vector_store.list_indexed_documents()


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


# --- MEMORY, PROGRESS & TASKS ---

@router.get("/memory/progress")
async def get_progress(request: Request):
    return await get_services(request).db.get_learning_progress()


@router.get("/memory/subjects")
async def get_subjects(request: Request):
    return await get_services(request).db.list_subjects()


class SubjectCreate(BaseModel):
    name: str
    description: str = ""


@router.post("/memory/subjects")
async def create_subject(req: SubjectCreate, request: Request):
    return await get_services(request).db.add_subject(str(uuid.uuid4()), req.name, req.description)


@router.get("/memory/topics")
async def get_topics(request: Request, subject_id: Optional[str] = None):
    return await get_services(request).db.list_topics(subject_id)


class TopicCreate(BaseModel):
    subject_id: str
    name: str
    mastery_level: float = 0.0
    notes: str = ""


@router.post("/memory/topics")
async def create_topic(req: TopicCreate, request: Request):
    return await get_services(request).db.add_or_update_topic(
        topic_id=str(uuid.uuid4()),
        subject_id=req.subject_id,
        name=req.name,
        mastery=req.mastery_level,
        notes=req.notes
    )


@router.get("/memory/tasks")
async def get_tasks(request: Request, status: Optional[str] = None):
    return await get_services(request).db.list_tasks(status)


class TaskCreate(BaseModel):
    title: str
    description: str = ""
    subject_id: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "medium"


@router.post("/memory/tasks")
async def create_task(req: TaskCreate, request: Request):
    return await get_services(request).db.add_task(
        task_id=str(uuid.uuid4()),
        title=req.title,
        description=req.description,
        subject_id=req.subject_id,
        deadline=req.deadline,
        priority=req.priority
    )


class TaskStatusUpdate(BaseModel):
    status: str  # "pending", "completed"


@router.patch("/memory/tasks/{task_id}")
async def update_task(task_id: str, req: TaskStatusUpdate, request: Request):
    ok = await get_services(request).db.update_task_status(task_id, req.status)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "success", "task_id": task_id, "new_status": req.status}


# --- TUTOR QUIZZES ---

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

@router.get("/v1/models")
async def openai_models():
    """Return models in OpenAI format for Open WebUI discovery."""
    return {
        "object": "list",
        "data": [
            {
                "id": "university-tutor",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "university-agent",
                "permission": [],
                "root": "university-tutor",
                "parent": None
            },
            {
                "id": "university-research",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "university-agent",
                "permission": [],
                "root": "university-research",
                "parent": None
            }
        ]
    }


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint connecting Open WebUI to our agent."""
    user_prompt = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            user_prompt = m.get("content", "")
            break

    mode = "tutor"
    if "research" in req.model:
        mode = "research"

    agent_req = AgentQueryRequest(
        message=user_prompt,
        mode=mode,
        use_rag=True,
        is_private=True
    )
    res = await get_services(request).agent.process_query(agent_req)

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
                    "content": res.reply
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_prompt) // 4,
            "completion_tokens": len(res.reply) // 4,
            "total_tokens": (len(user_prompt) + len(res.reply)) // 4
        }
    }
