"""Integration tests for FastAPI endpoints, RAG upload, and Open WebUI OpenAI compatibility."""

import json
from unittest.mock import patch
import pytest


@pytest.mark.asyncio
async def test_health_and_system_endpoints(async_client):
    # Health
    res = await async_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "local_model" in data

    # System status
    res_sys = await async_client.get("/api/system/status")
    assert res_sys.status_code == 200
    assert "memory" in res_sys.json()


@pytest.mark.asyncio
async def test_chat_first_ui_and_static_assets(async_client):
    page = await async_client.get("/")
    assert page.status_code == 200
    assert "Ask the University Agent" in page.text
    assert "/static/app.css" in page.text
    assert "/static/app.js" in page.text

    styles = await async_client.get("/static/app.css")
    assert styles.status_code == 200
    assert "#070b10" in styles.text
    assert "prefers-reduced-motion" in styles.text
    assert "reasoning-accordion" in styles.text
    assert "citation-badge" in styles.text
    assert "chat-approval-card" in styles.text

    # Also verify /static/styles.css alias
    styles_alias = await async_client.get("/static/styles.css")
    assert styles_alias.status_code == 200
    assert "reasoning-accordion" in styles_alias.text

    script = await async_client.get("/static/app.js")
    assert script.status_code == 200
    assert 'api("/api/chat"' in script.text
    assert "/api/chat/history/" in script.text
    assert "TextDecoder" in script.text
    assert "reasoning-accordion" in script.text
    assert "renderSourcesCard" in script.text
    assert "renderApprovalCard" in script.text
    assert "/api/memory/sessions" in script.text
    assert "/api/security/approve/" in script.text


@pytest.mark.asyncio
async def test_document_upload_and_search(async_client):
    # Upload a test lecture note
    file_content = b"Graph Theory: An Euler path visits every edge of a graph exactly once."
    files = {"file": ("discrete_math.txt", file_content, "text/plain")}
    data = {"subject": "Discrete Mathematics"}

    upload_res = await async_client.post("/api/documents/upload", files=files, data=data)
    assert upload_res.status_code == 200
    res_data = upload_res.json()
    assert res_data["filename"] == "discrete_math.txt"
    assert res_data["total_chunks"] >= 1

    # Search
    search_res = await async_client.post("/api/documents/search", json={"query": "Euler path graph edges"})
    assert search_res.status_code == 200
    matches = search_res.json()
    assert len(matches) >= 1
    assert "Euler path" in matches[0]["text"]
    assert "discrete_math.txt" in matches[0]["citation"]


@pytest.mark.asyncio
async def test_academic_tasks_api(async_client):
    # Create task
    task_payload = {
        "title": "Submit AI Term Paper",
        "description": "Final 10-page research synthesis",
        "deadline": "2026-12-01",
        "priority": "high"
    }
    create_res = await async_client.post("/api/memory/tasks", json=task_payload)
    assert create_res.status_code == 200
    task_id = create_res.json()["id"]

    # List pending
    list_res = await async_client.get("/api/memory/tasks?status=pending")
    assert list_res.status_code == 200
    assert any(t["id"] == task_id for t in list_res.json())

    # Update to completed
    patch_res = await async_client.patch(f"/api/memory/tasks/{task_id}", json={"status": "completed"})
    assert patch_res.status_code == 200
    assert patch_res.json()["new_status"] == "completed"


@pytest.mark.asyncio
async def test_openai_compatibility_for_open_webui(async_client):
    """Verify standard /v1/models and /v1/chat/completions endpoints for Open WebUI connection."""
    # Models endpoint
    res = await async_client.get("/v1/models")
    assert res.status_code == 200
    models_data = res.json()
    assert "data" in models_data
    model_ids = [m["id"] for m in models_data["data"]]
    assert "university-tutor" in model_ids

    # Chat completions endpoint
    chat_payload = {
        "model": "university-tutor",
        "messages": [
            {"role": "user", "content": "Explain what Big-O notation means in one sentence."}
        ]
    }
    chat_res = await async_client.post("/v1/chat/completions", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "choices" in chat_data
    assert chat_data["choices"][0]["message"]["role"] == "assistant"
    assert len(chat_data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_cors_rejects_untrusted_origins(async_client):
    """Verify that external untrusted origins are NOT permitted in CORS response."""
    # Evil origin
    res = await async_client.options(
        "/api/health",
        headers={
            "Origin": "http://evil-attacker.com",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res.headers.get("access-control-allow-origin") != "http://evil-attacker.com"
    assert res.headers.get("access-control-allow-origin") != "*"

    # Authorized local origin
    local_res = await async_client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert local_res.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_document_upload_validation_and_rejections(async_client):
    """Verify upload rejection for unsupported formats, empty files, and malicious filenames."""
    # Unsupported format (.exe / .sh)
    res1 = await async_client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", b"binarycontent", "application/octet-stream")}
    )
    assert res1.status_code == 400
    assert "Unsupported file format" in res1.json()["detail"]

    # Empty file (0 bytes)
    res2 = await async_client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert res2.status_code == 400
    assert "empty" in res2.json()["detail"].lower()

    # Restricted filename (.env)
    res3 = await async_client.post(
        "/api/documents/upload",
        files={"file": (".env", b"SECRET=123", "text/plain")}
    )
    assert res3.status_code in (400, 403)


@pytest.mark.asyncio
async def test_chat_sse_streaming_endpoint(async_client, test_app):
    """Verify that POST /api/chat streams SSE events with tokens and final completion metadata."""
    container = test_app.state.services

    async def mock_stream(req):
        yield {"token": "Binary ", "session_id": req.session_id or "sess-1"}
        yield {"token": "search trees", "session_id": req.session_id or "sess-1"}
        yield {
            "done": True,
            "session_id": req.session_id or "sess-1",
            "message_id": "msg-123",
            "reply": "Binary search trees",
            "metadata": {
                "used_sources": [{"filename": "trees.txt", "chunk_id": "[1]"}],
                "reasoning_path": "Reasoning step 1",
                "intent": "TUTORING"
            }
        }

    with patch.object(container.agent, "process_query_stream", side_effect=mock_stream):
        payload = {
            "session_id": "sess-1",
            "messages": [{"role": "user", "content": "Explain binary search trees step by step"}],
            "mode": "tutor"
        }
        res = await async_client.post("/api/chat", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

        lines = [line.strip() for line in res.text.split("\n") if line.strip().startswith("data:")]
        assert len(lines) >= 3

        events = [json.loads(line[len("data:"):].strip()) for line in lines]
        tokens = [e["token"] for e in events if "token" in e]
        assert "Binary " in tokens
        assert "search trees" in tokens

        final_event = next(e for e in events if e.get("done") is True)
        assert final_event["session_id"] == "sess-1"
        assert "metadata" in final_event
        assert final_event["metadata"]["intent"] == "TUTORING"
        assert len(final_event["metadata"]["used_sources"]) == 1


@pytest.mark.asyncio
async def test_agent_chat_alias_sse_streaming(async_client, test_app):
    """Verify that POST /api/agent/chat alias streams SSE events cleanly."""
    container = test_app.state.services

    async def mock_stream(req):
        yield {"token": "Hello ", "session_id": req.session_id}
        yield {"token": "student!", "session_id": req.session_id}
        yield {
            "done": True,
            "session_id": req.session_id,
            "reply": "Hello student!",
            "metadata": {"used_sources": [], "reasoning_path": None, "intent": "DEFAULT_QA"}
        }

    with patch.object(container.agent, "process_query_stream", side_effect=mock_stream):
        payload = {
            "session_id": "sess-alias",
            "message": "Hello!",
            "mode": "tutor"
        }
        res = await async_client.post("/api/agent/chat", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        lines = [line.strip() for line in res.text.split("\n") if line.strip().startswith("data:")]
        events = [json.loads(line[len("data:"):].strip()) for line in lines]
        assert any("token" in e for e in events)
        assert any(e.get("done") is True for e in events)


@pytest.mark.asyncio
async def test_chat_sse_streaming_error_handling(async_client, test_app):
    """Verify that mid-stream or upfront errors yield an SSE error block without crashing the connection."""
    container = test_app.state.services

    async def mock_stream_error(req):
        yield {"token": "Starting...", "session_id": req.session_id}
        raise RuntimeError("Inference connection reset by peer")

    with patch.object(container.agent, "process_query_stream", side_effect=mock_stream_error):
        payload = {
            "session_id": "sess-err",
            "message": "Simulate error",
        }
        res = await async_client.post("/api/chat", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        lines = [line.strip() for line in res.text.split("\n") if line.strip().startswith("data:")]
        events = [json.loads(line[len("data:"):].strip()) for line in lines]
        error_event = next(e for e in events if "error" in e)
        assert "Inference connection reset by peer" in error_event["error"]
        assert error_event.get("done") is True


@pytest.mark.asyncio
async def test_agent_quiz_endpoint(async_client, test_app):
    """Verify that POST /api/agent/quiz returns structured multiple-choice quiz JSON."""
    container = test_app.state.services
    mock_quiz = {
        "topic": "Binary Search Trees",
        "questions": [
            {
                "question_text": "What is the average lookup time in a balanced BST?",
                "options": ["O(1)", "O(log n)", "O(n)", "O(n^2)"],
                "correct_answer": None,
                "explanation": None,
                "source_chunk_id": "[1]",
                "source_metadata": {"filename": "bst.pdf", "page_number": 3}
            }
        ],
        "status": "success",
        "total_questions": 1,
        "answers_hidden": True
    }
    with patch.object(container.agent, "generate_quiz", return_value=mock_quiz) as mock_gen:
        quiz_payload = {
            "subject_id": "CS101",
            "topic": "Binary Search Trees",
            "num_questions": 1,
            "hide_answers": True
        }
        res = await async_client.post("/api/agent/quiz", json=quiz_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["topic"] == "Binary Search Trees"
        assert len(data["questions"]) == 1
        assert data["answers_hidden"] is True
        mock_gen.assert_called_once_with(
            subject_id="CS101",
            topic="Binary Search Trees",
            num_questions=1,
            hide_answers=True,
            topic_id=None,
            topic_name=None
        )


@pytest.mark.asyncio
async def test_memory_sessions_and_history_endpoints(async_client, test_app):
    """Verify GET /api/memory/sessions and GET /api/memory/sessions/{session_id} retrieve persisted history."""
    container = test_app.state.services
    db = container.db

    session_id_1 = "session-mem-1"
    session_id_2 = "session-mem-2"

    await db.create_session(session_id=session_id_1, session_type="tutor", title="Tutor Session 1")
    await db.create_session(session_id=session_id_2, session_type="research", title="Research Session 2")

    await db.save_message(
        session_id=session_id_1,
        role="user",
        content="What is quicksort?",
        mode="tutor"
    )
    await db.save_message(
        session_id=session_id_1,
        role="assistant",
        content="Quicksort is a divide-and-conquer sorting algorithm.",
        mode="tutor",
        metadata={"used_sources": [{"filename": "sort.pdf"}], "intent": "TUTORING"}
    )

    # 1. Test GET /api/memory/sessions
    res_sessions = await async_client.get("/api/memory/sessions?limit=10")
    assert res_sessions.status_code == 200
    sessions_list = res_sessions.json()
    assert isinstance(sessions_list, list)
    session_ids = [s["id"] for s in sessions_list]
    assert session_id_1 in session_ids
    assert session_id_2 in session_ids

    # 2. Test GET /api/memory/sessions/{session_id}
    res_history = await async_client.get(f"/api/memory/sessions/{session_id_1}")
    assert res_history.status_code == 200
    history = res_history.json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is quicksort?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Quicksort is a divide-and-conquer sorting algorithm."
    assert "metadata" in history[1]
    assert history[1]["metadata"]["intent"] == "TUTORING"


@pytest.mark.asyncio
async def test_subject_and_topic_crud_api(async_client, test_app):
    """Verify creating a subject via API, listing subjects, and adding/listing topics under that subject."""
    # 1. Create a subject via POST /api/subjects
    sub_payload = {
        "name": "Distributed Systems",
        "code": "CS401",
        "description": "Consensus, replication, and fault tolerance."
    }
    create_res = await async_client.post("/api/subjects", json=sub_payload)
    assert create_res.status_code == 200
    sub_data = create_res.json()
    assert sub_data["name"] == "Distributed Systems"
    assert sub_data["code"] == "CS401"
    sub_id = sub_data["id"]

    # 2. Retrieve via GET /api/subjects
    list_res = await async_client.get("/api/subjects")
    assert list_res.status_code == 200
    subjects = list_res.json()
    assert any(s["id"] == sub_id for s in subjects)

    # 3. Add topic to this subject via POST /api/subjects/{subject_id}/topics
    topic_payload = {
        "name": "Raft Consensus",
        "mastery_level": 0.85,
        "notes": "Leader election and log replication."
    }
    t_create_res = await async_client.post(f"/api/subjects/{sub_id}/topics", json=topic_payload)
    assert t_create_res.status_code == 200
    t_data = t_create_res.json()
    assert t_data["name"] == "Raft Consensus"
    assert t_data["mastery_level"] == 0.85

    # 4. List topics via GET /api/subjects/{subject_id}/topics
    t_list_res = await async_client.get(f"/api/subjects/{sub_id}/topics")
    assert t_list_res.status_code == 200
    topics = t_list_res.json()
    assert len(topics) >= 1
    assert any(t["name"] == "Raft Consensus" for t in topics)


@pytest.mark.asyncio
async def test_add_topic_to_nonexistent_subject_returns_error(async_client, test_app):
    """Verify attempting to add a topic to a non-existent subject returns 400 or 404."""
    topic_payload = {
        "name": "Quantum Entanglement",
        "mastery_level": 0.5
    }
    res = await async_client.post("/api/subjects/non-existent-sub-id-9999/topics", json=topic_payload)
    assert res.status_code in (400, 404)
    data = res.json()
    assert "detail" in data or "error" in data


@pytest.mark.asyncio
async def test_upcoming_and_overdue_tasks_api(async_client, test_app):
    """Verify fetching upcoming and overdue tasks via API returns valid lists."""
    # 1. Create a task due soon (upcoming)
    upcoming_task = {
        "title": "Complete Lab 3",
        "description": "Socket programming lab",
        "deadline": "2026-11-20T23:59:59Z",
        "priority": "high"
    }
    t1_res = await async_client.post("/api/tasks", json=upcoming_task)
    assert t1_res.status_code == 200
    t1_id = t1_res.json()["id"]

    # 2. Create an overdue task
    overdue_task = {
        "title": "Past Assignment 1",
        "description": "Old assignment",
        "deadline": "2020-01-01T00:00:00Z",
        "priority": "medium"
    }
    t2_res = await async_client.post("/api/tasks", json=overdue_task)
    assert t2_res.status_code == 200
    t2_id = t2_res.json()["id"]

    # 3. GET /api/tasks/upcoming
    up_res = await async_client.get("/api/tasks/upcoming?days_ahead=30")
    assert up_res.status_code == 200
    up_tasks = up_res.json()
    assert isinstance(up_tasks, list)

    # 4. GET /api/tasks/overdue
    od_res = await async_client.get("/api/tasks/overdue")
    assert od_res.status_code == 200
    od_tasks = od_res.json()
    assert isinstance(od_tasks, list)
    assert any(t["id"] == t2_id for t in od_tasks)


@pytest.mark.asyncio
async def test_study_goals_api_and_clamped_progress(async_client, test_app):
    """Verify creating study goals and updating progress clamps [0, 100] and auto-completes at 100%."""
    # 1. Create goal via POST /api/goals
    goal_payload = {
        "title": "Master Computer Architecture",
        "description": "Cover pipeline hazards and caches",
        "target_date": "2026-12-31",
        "progress_percent": 25.0
    }
    create_res = await async_client.post("/api/goals", json=goal_payload)
    assert create_res.status_code == 200
    goal_data = create_res.json()
    assert goal_data["title"] == "Master Computer Architecture"
    assert goal_data["progress_percent"] == 25.0
    goal_id = goal_data["id"]

    # 2. List goals via GET /api/goals
    list_res = await async_client.get("/api/goals")
    assert list_res.status_code == 200
    goals = list_res.json()
    assert any(g["id"] == goal_id for g in goals)

    # 3. Update progress exceeding 100% -> should clamp to 100.0 and mark status 'completed'
    patch_res = await async_client.patch(f"/api/goals/{goal_id}/progress", json={"progress_percent": 150.0})
    assert patch_res.status_code == 200
    updated_goal = patch_res.json()
    assert updated_goal["progress_percent"] == 100.0
    assert updated_goal["status"] == "completed"

    # 4. Update progress below 0% -> should clamp to 0.0
    patch_res_neg = await async_client.patch(f"/api/goals/{goal_id}/progress", json={"progress_percent": -20.0})
    assert patch_res_neg.status_code == 200
    assert patch_res_neg.json()["progress_percent"] == 0.0

    # 5. Non-existent goal returns 404
    bad_res = await async_client.patch("/api/goals/non-existent-goal-999/progress", json={"progress_percent": 50.0})
    assert bad_res.status_code == 404


@pytest.mark.asyncio
async def test_document_listing_with_subject_filter(async_client, test_app):
    """Verify GET /api/documents returns documents and filters by subject_id."""
    res = await async_client.get("/api/documents")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    res_filtered = await async_client.get("/api/documents?subject_id=NonExistentSubject")
    assert res_filtered.status_code == 200
    assert res_filtered.json() == []


@pytest.mark.asyncio
async def test_security_approval_endpoints_end_to_end(async_client, test_app):
    """Verify security approval and rejection endpoints work end-to-end with validator and audit log."""
    sec_val = test_app.state.services.security_validator

    # 1. Generate approval request token on test_app's security validator
    token = sec_val.generate_approval_token(
        action="terminal_run",
        params={"command": "pytest"}
    )
    assert isinstance(token, str) and len(token) > 10

    # 2. Check pending list
    pending_res = await async_client.get("/api/security/approvals")
    assert pending_res.status_code == 200
    pending_list = pending_res.json()
    assert any(req["id"] == token for req in pending_list)

    # 3. Approve via POST /api/security/approve/{token}
    approve_res = await async_client.post(
        f"/api/security/approve/{token}",
        json={"reason": "Authorized by user"}
    )
    assert approve_res.status_code == 200
    assert approve_res.json() == {"status": "approved", "approval_id": token}

    # Verify status in validator
    approval_obj = sec_val.get_approval(token)
    assert approval_obj is not None
    assert approval_obj.status.value == "approved"

    # 4. Generate another token and test reject via POST /api/security/reject/{token}
    token_reject = sec_val.generate_approval_token(
        action="delete_file",
        params={"path": "dangerous.py"}
    )
    reject_res = await async_client.post(
        f"/api/security/reject/{token_reject}",
        json={"reason": "Rejected by security policy"}
    )
    assert reject_res.status_code == 200
    assert reject_res.json() == {"status": "rejected", "approval_id": token_reject}

    approval_reject_obj = sec_val.get_approval(token_reject)
    assert approval_reject_obj is not None
    assert approval_reject_obj.status.value == "rejected"

    # 5. Non-existent ID returns 404
    bad_res = await async_client.post("/api/security/approve/nonexistent-id", json={})
    assert bad_res.status_code == 404
