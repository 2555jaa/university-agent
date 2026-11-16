"""Integration tests for FastAPI endpoints, RAG upload, and Open WebUI OpenAI compatibility."""

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

    script = await async_client.get("/static/app.js")
    assert script.status_code == 200
    assert 'api("/api/chat"' in script.text
    assert "/api/chat/history/" in script.text


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
