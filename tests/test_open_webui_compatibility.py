"""Tests for Open WebUI and OpenAI-compatible API endpoints (/v1/models and /v1/chat/completions)."""

import json
from unittest.mock import AsyncMock, patch
import pytest

from app.agent.orchestrator import AgentResponse
from app.providers.base import LLMResponse


@pytest.mark.asyncio
async def test_openai_models_listing(async_client):
    """Verify GET /v1/models returns virtual academic model profiles in strict OpenAI format."""
    res = await async_client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()

    assert data["object"] == "list"
    assert isinstance(data["data"], list)

    model_map = {m["id"]: m for m in data["data"]}

    # Required virtual academic profiles
    expected_models = [
        "university-agent",
        "university-agent-tutor",
        "university-agent-research",
        "university-agent-summarizer",
    ]

    for model_id in expected_models:
        assert model_id in model_map, f"Missing model {model_id}"
        m = model_map[model_id]
        assert m["object"] == "model"
        assert isinstance(m["created"], int)
        assert m["owned_by"] == "university-agent"

    # Legacy models maintained for backwards compatibility
    assert "university-tutor" in model_map
    assert "university-research" in model_map


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming_basic(async_client):
    """Verify POST /v1/chat/completions (stream=False) returns valid OpenAI chat.completion payload."""
    payload = {
        "model": "university-agent",
        "messages": [
            {"role": "user", "content": "What is the Big-O complexity of binary search?"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False
    }

    res = await async_client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "university-agent"

    choices = data["choices"]
    assert len(choices) == 1
    assert choices[0]["index"] == 0
    assert choices[0]["finish_reason"] == "stop"
    assert choices[0]["message"]["role"] == "assistant"
    assert len(choices[0]["message"]["content"]) > 0

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_openai_chat_completions_persona_model_routing(async_client):
    """Verify model IDs route to corresponding agent modes: tutor, research, and summarizer."""
    test_cases = [
        ("university-agent-tutor", "tutor"),
        ("university-agent-research", "research"),
        ("university-agent-summarizer", "summarizer"),
    ]

    for model_name, expected_mode in test_cases:
        with patch("app.agent.orchestrator.AcademicAgent.process_query", new_callable=AsyncMock) as mock_pq:
            mock_pq.return_value = AgentResponse(
                session_id="test-session",
                message_id="msg-123",
                reply="Mock persona reply",
                mode=expected_mode,
                model="test",
                provider="test",
                is_fallback=False
            )

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Test prompt"}],
                "stream": False
            }

            res = await async_client.post("/v1/chat/completions", json=payload)
            assert res.status_code == 200

            # Verify that process_query was called with the mapped mode
            call_req = mock_pq.call_args[0][0]
            assert call_req.mode == expected_mode
            assert call_req.message == "Test prompt"


@pytest.mark.asyncio
async def test_openai_chat_completions_citations_formatting(async_client):
    """Verify non-streaming response appends formatted Markdown Sources footer and populates citations metadata."""
    sources = [
        {
            "index": 1,
            "filename": "os_virtual_memory.pdf",
            "page_number": 42,
            "heading": "Page Table Structures"
        },
        {
            "index": 2,
            "filename": "discrete_math.txt",
            "page_number": 1,
            "heading": "Euler Circuits"
        }
    ]

    mock_agent_response = AgentResponse(
        session_id="session-cite",
        message_id="msg-cite",
        reply="Paging maps virtual addresses to physical frames [1]. Euler paths exist when degrees are even [2].",
        mode="tutor",
        model="test-model",
        provider="ollama",
        is_fallback=False,
        citations=sources,
        used_sources=sources,
        has_grounding=True
    )

    with patch("app.agent.orchestrator.AcademicAgent.process_query", new_callable=AsyncMock) as mock_pq:
        mock_pq.return_value = mock_agent_response

        payload = {
            "model": "university-agent",
            "messages": [{"role": "user", "content": "Explain paging and Euler paths."}],
            "stream": False
        }

        res = await async_client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()

        content = data["choices"][0]["message"]["content"]

        # 1. Main reply content is present
        assert "Paging maps virtual addresses" in content

        # 2. Markdown sources footer is cleanly appended
        assert "\n\n---\n**Sources:**" in content
        assert "[1] os_virtual_memory.pdf (Page 42 — Page Table Structures)" in content
        assert "[2] discrete_math.txt (Page 1 — Euler Circuits)" in content

        # 3. Top-level metadata for citations aware clients
        assert "citations" in data
        assert len(data["citations"]) == 2
        assert "used_sources" in data
        assert len(data["used_sources"]) == 2


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming_sse(async_client):
    """Verify POST /v1/chat/completions with stream=True yields valid chat.completion.chunk SSE events and terminates with [DONE]."""
    sources = [
        {
            "index": 1,
            "filename": "ai_governance.docx",
            "page_number": 5,
            "heading": "Risk Classification"
        }
    ]

    async def mock_stream_generator(req):
        yield {"token": "High-risk"}
        yield {"token": " AI systems"}
        yield {"token": " require third-party auditing [1]."}
        yield {
            "done": True,
            "reply": "High-risk AI systems require third-party auditing [1].",
            "used_sources": sources,
            "session_id": "test-sess",
            "message_id": "msg-stream"
        }

    with patch("app.agent.orchestrator.AcademicAgent.process_query_stream", side_effect=mock_stream_generator):
        payload = {
            "model": "university-agent-research",
            "messages": [{"role": "user", "content": "What are EU AI Act obligations?"}],
            "stream": True
        }

        res = await async_client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

        raw_text = res.text
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # All lines must start with "data: "
        for line in lines:
            assert line.startswith("data: "), f"Malformed SSE line: {line}"

        # Parse data payloads
        events = []
        for line in lines:
            content_part = line[len("data: "):]
            if content_part == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(content_part))

        # 1. Last event must be [DONE]
        assert events[-1] == "[DONE]"

        # 2. Second to last event must be stop chunk
        stop_chunk = events[-2]
        assert stop_chunk["object"] == "chat.completion.chunk"
        assert stop_chunk["choices"][0]["finish_reason"] == "stop"

        # 3. Content chunks validation
        content_deltas = [
            e["choices"][0]["delta"]["content"]
            for e in events[:-2]
            if "content" in e["choices"][0].get("delta", {})
        ]
        accumulated_streamed_text = "".join(content_deltas)

        # Main response text
        assert "High-risk AI systems require third-party auditing [1]." in accumulated_streamed_text

        # 4. Sources references footer streamed before stop
        assert "\n\n---\n**Sources:**" in accumulated_streamed_text
        assert "[1] ai_governance.docx (Page 5 — Risk Classification)" in accumulated_streamed_text
