"""Unit tests for provider abstraction and hybrid AI routing."""

import json
import pytest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.providers.router import ModelRouter, TaskComplexity
from app.providers.ollama_provider import (
    OllamaProvider,
    ModelNotFoundError,
    StreamInterruptedError,
    DEFAULT_OLLAMA_TIMEOUT,
)
from app.providers.cloud_provider import CloudAIProvider, CloudProvider


class MockLocalProvider(BaseLLMProvider):
    async def is_available(self) -> bool:
        return True

    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(
            content="Local model response: Explaining concept step by step.",
            model="qwen2.5:1.5b",
            provider="ollama"
        )


class MockCloudProvider(BaseLLMProvider):
    def __init__(self, should_fail: bool = False, available: bool = True):
        self.should_fail = should_fail
        self.available = available

    async def is_available(self) -> bool:
        return self.available

    async def check_health(self) -> dict:
        if not self.available:
            return {"status": "offline", "healthy": False}
        return {"status": "ready", "healthy": True}

    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        if self.should_fail:
            raise RuntimeError("Cloud rate limit exceeded")
        return LLMResponse(
            content="Cloud model response: Advanced theoretical synthesis.",
            model="gemini-2.5-flash",
            provider="cloud"
        )


@pytest.mark.asyncio
async def test_privacy_enforces_local_routing():
    """Verify that private academic queries are NEVER routed to cloud even if complexity is HIGH."""
    local_mock = MockLocalProvider()
    cloud_mock = MockCloudProvider()
    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_mock)

    messages = [LLMMessage(role="user", content="Here is my unpublished draft thesis.")]
    res = await router.route_and_generate(
        messages=messages,
        complexity=TaskComplexity.HIGH,
        is_private=True  # STRICT PRIVACY
    )

    assert res.provider == "ollama"
    assert "Local model" in res.content


@pytest.mark.asyncio
async def test_high_complexity_routes_to_cloud():
    """Verify that non-private, high-complexity tasks route to cloud when available."""
    local_mock = MockLocalProvider()
    cloud_mock = MockCloudProvider()
    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_mock)
    router.settings.RESTRICT_TO_LOCAL = False
    router.settings.CLOUD_ENABLED = True

    messages = [LLMMessage(role="user", content="Solve this complex NP-complete graph reduction.")]
    res = await router.route_and_generate(
        messages=messages,
        complexity=TaskComplexity.HIGH,
        is_private=False
    )

    assert res.provider == "cloud"
    assert "Cloud model" in res.content


@pytest.mark.asyncio
async def test_cloud_failure_falls_back_to_local():
    """Verify that if cloud fails, router falls back gracefully to local model."""
    local_mock = MockLocalProvider()
    failing_cloud = MockCloudProvider(should_fail=True)
    router = ModelRouter(local_provider=local_mock, cloud_provider=failing_cloud)
    router.settings.RESTRICT_TO_LOCAL = False
    router.settings.CLOUD_ENABLED = True

    messages = [LLMMessage(role="user", content="Complex research inquiry.")]
    res = await router.route_and_generate(
        messages=messages,
        complexity=TaskComplexity.HIGH,
        is_private=False
    )

    assert res.provider == "ollama"
    assert res.is_fallback is True


class MockStreamResponse:
    def __init__(self, status_code: int = 200, lines: Optional[List[str]] = None):
        self.status_code = status_code
        self._lines = lines or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def aread(self):
        return b'{"error": "mock error"}'

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_ollama_streaming_yields_sequential_tokens():
    """Verify that streaming yields tokens sequentially and handles successful completion correctly."""
    lines = [
        json.dumps({"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": "Virtual"}, "done": False}),
        json.dumps({"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": " memory"}, "done": False}),
        json.dumps({"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": " paging."}, "done": True, "total_duration": 5000})
    ]
    mock_resp = MockStreamResponse(status_code=200, lines=lines)
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        provider = OllamaProvider(model="qwen2.5:1.5b")
        chunks = []
        async for chunk in provider.generate_stream(prompt="Explain virtual memory"):
            chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["message"]["content"] == "Virtual"
    assert chunks[0]["response"] == "Virtual"
    assert chunks[1]["message"]["content"] == " memory"
    assert chunks[2]["done"] is True
    full_text = "".join(c.get("response", "") for c in chunks)
    assert full_text == "Virtual memory paging."


@pytest.mark.asyncio
async def test_ollama_model_not_found_404():
    """Verify that a simulated model-not-found (404) returns the expected explicit error advising ollama pull."""
    mock_resp = MockStreamResponse(status_code=404)
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_resp)

    mock_post_resp = AsyncMock(status_code=404, text='{"error": "model \'missing:latest\' not found"}')
    mock_client.post.return_value = mock_post_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        provider = OllamaProvider(model="missing:latest")

        # 1. Test generate_stream yields structured error block
        stream_chunks = [chunk async for chunk in provider.generate_stream(prompt="hello")]
        assert len(stream_chunks) == 1
        assert stream_chunks[0]["error_type"] == "model_not_found"
        assert stream_chunks[0]["status_code"] == 404
        assert stream_chunks[0]["done"] is True
        assert "ollama pull missing:latest" in stream_chunks[0]["error"]

        # 2. Test generate() returns structured response advising pull
        gen_res = await provider.generate(prompt="hello")
        assert gen_res.finish_reason == "model_not_found"
        assert "ollama pull missing:latest" in gen_res.content

        # 3. Test generate(raise_on_error=True) raises ModelNotFoundError
        with pytest.raises(ModelNotFoundError) as exc_info:
            await provider.generate(prompt="hello", raise_on_error=True)
        assert "ollama pull missing:latest" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_stream_interrupted_read_timeout():
    """Verify that a simulated httpx.ReadTimeout mid-stream safely yields [STREAM_INTERRUPTED_ERROR]."""
    class TimeoutStreamResponse:
        def __init__(self):
            self.status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            yield json.dumps({"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": "Partial token"}, "done": False})
            raise httpx.ReadTimeout("Read timed out from Ollama daemon")

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=TimeoutStreamResponse())
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        provider = OllamaProvider(model="qwen2.5:1.5b")
        chunks = []
        async for chunk in provider.generate_stream(prompt="Heavy prompt"):
            chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["message"]["content"] == "Partial token"
    # Interrupted payload safely yielded instead of crashing process
    assert chunks[1]["error"] == "[STREAM_INTERRUPTED_ERROR]"
    assert chunks[1]["error_type"] == "stream_interrupted"
    assert chunks[1]["done"] is True
    assert "[STREAM_INTERRUPTED_ERROR]" in chunks[1]["message"]["content"]


@pytest.mark.asyncio
async def test_ollama_max_tokens_mapped_to_num_predict_and_timeout():
    """Verify that max_tokens maps to num_predict in options and timeouts are bounded."""
    mock_post_resp = AsyncMock(status_code=200)
    mock_post_resp.json = MagicMock(return_value={
        "message": {"role": "assistant", "content": "Generated reply"},
        "done_reason": "stop"
    })

    mock_resp = MockStreamResponse(
        status_code=200,
        lines=[json.dumps({"message": {"content": "Streamed"}, "done": True})]
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_post_resp
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client

    captured_timeouts = []

    def mock_client_init(*args, **kwargs):
        captured_timeouts.append(kwargs.get("timeout"))
        return mock_client

    with patch("app.providers.ollama_provider.httpx.AsyncClient", side_effect=mock_client_init):
        provider = OllamaProvider(model="qwen2.5:1.5b")

        # 1. Test max_tokens in generate()
        await provider.generate(prompt="Test", max_tokens=150)
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["options"]["num_predict"] == 150

        # 2. Test max_tokens in generate_stream()
        async for _ in provider.generate_stream(prompt="Test stream", max_tokens=350):
            pass
        stream_call_kwargs = mock_client.stream.call_args.kwargs
        assert stream_call_kwargs["json"]["options"]["num_predict"] == 350

        # 3. Verify bounded TTFT timeouts
        assert len(captured_timeouts) >= 2
        for t in captured_timeouts:
            assert isinstance(t, httpx.Timeout)
            assert t.read == 120.0
            assert t.connect == 5.0


@pytest.mark.asyncio
async def test_restrict_to_local_forces_all_requests_to_local_provider():
    """Verify that RESTRICT_TO_LOCAL=True strictly routes all requests to local provider."""
    local_mock = MockLocalProvider()
    cloud_mock = MockCloudProvider(available=True)
    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_mock)
    router.settings.RESTRICT_TO_LOCAL = True
    router.settings.CLOUD_ENABLED = True

    messages = [LLMMessage(role="user", content="Prove that graph isomorphism is NP-complete.")]

    # 1. route_query must return "local" regardless of HIGH complexity
    target = await router.route_query(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert target == "local"

    # 2. route_and_generate executes on local provider
    res = await router.route_and_generate(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert res.provider == "ollama"
    assert "Local model" in res.content


@pytest.mark.asyncio
async def test_sensitive_rag_context_fencing_forces_local_routing():
    """Verify that prompts containing <context> or <source> tags are strictly fenced to local provider."""
    local_mock = MockLocalProvider()
    cloud_mock = MockCloudProvider(available=True)
    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_mock)
    router.settings.RESTRICT_TO_LOCAL = False
    router.settings.CLOUD_ENABLED = True

    # Prompt containing injected RAG context with <context> and <source> tags
    rag_prompt = (
        "<context>\n"
        '<source id="[1]" document="thesis_draft.pdf" page="12" heading="Proof">\n'
        "Proprietary university research data.\n"
        "</source>\n"
        "</context>\n"
        "Please prove that the reduction holds."
    )
    messages = [LLMMessage(role="user", content=rag_prompt)]

    # Sensitive data fencing must force local routing
    target = await router.route_query(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert target == "local"

    res = await router.route_and_generate(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert res.provider == "ollama"
    assert "Local model" in res.content


@pytest.mark.asyncio
async def test_high_complexity_heuristic_and_explicit_routes_to_cloud():
    """Verify high-complexity tasks without RAG context route to cloud when enabled and not restricted."""
    local_mock = MockLocalProvider()
    cloud_mock = MockCloudProvider(available=True)
    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_mock)
    router.settings.RESTRICT_TO_LOCAL = False
    router.settings.CLOUD_ENABLED = True

    # 1. Explicit HIGH complexity
    messages_explicit = [LLMMessage(role="user", content="Calculate asymptotic complexity bounds.")]
    target_explicit = await router.route_query(messages=messages_explicit, complexity=TaskComplexity.HIGH, is_private=False)
    assert target_explicit == "cloud"
    res_explicit = await router.route_and_generate(messages=messages_explicit, complexity=TaskComplexity.HIGH, is_private=False)
    assert res_explicit.provider == "cloud"

    # 2. Heuristic complexity detection (no explicit complexity passed)
    messages_heuristic = [LLMMessage(role="user", content="Derive the mathematical proof for the NP-complete graph reduction.")]
    target_heuristic = await router.route_query(messages=messages_heuristic, complexity=None, is_private=False)
    assert target_heuristic == "cloud"
    res_heuristic = await router.route_and_generate(messages=messages_heuristic, complexity=None, is_private=False)
    assert res_heuristic.provider == "cloud"

    # 3. Simple explanation defaults to local
    messages_simple = [LLMMessage(role="user", content="What is the definition of a binary tree?")]
    target_simple = await router.route_query(messages=messages_simple, complexity=None, is_private=False)
    assert target_simple == "local"


@pytest.mark.asyncio
async def test_cloud_failover_when_offline_or_missing_api_key():
    """Verify that if cloud provider is offline, missing API key, or fails mid-generation, request falls back to local."""
    local_mock = MockLocalProvider()

    # 1. Cloud provider is offline -> router immediately chooses local
    offline_cloud = MockCloudProvider(available=False)
    router_offline = ModelRouter(local_provider=local_mock, cloud_provider=offline_cloud)
    router_offline.settings.RESTRICT_TO_LOCAL = False
    router_offline.settings.CLOUD_ENABLED = True

    messages = [LLMMessage(role="user", content="Derive the mathematical proof for the graph theorem.")]
    target = await router_offline.route_query(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert target == "local"
    res = await router_offline.route_and_generate(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert res.provider == "ollama"

    # 2. Cloud provider fails during generation -> catches error and falls back to local with is_fallback=True
    failing_cloud = MockCloudProvider(should_fail=True, available=True)
    router_failing = ModelRouter(local_provider=local_mock, cloud_provider=failing_cloud)
    router_failing.settings.RESTRICT_TO_LOCAL = False
    router_failing.settings.CLOUD_ENABLED = True

    res_fallback = await router_failing.route_and_generate(messages=messages, complexity=TaskComplexity.HIGH, is_private=False)
    assert res_fallback.provider == "ollama"
    assert res_fallback.is_fallback is True


@pytest.mark.asyncio
async def test_cloud_provider_health_check_and_cost_controls():
    """Verify CloudAIProvider check_health, missing API key handling, and max_tokens cost control."""
    from app.config import Settings

    # 1. Missing API key returns status="missing"
    settings_no_key = Settings(APP_ENV="testing", CLOUD_ENABLED=False)
    cloud_no_key = CloudAIProvider(api_key="", settings_obj=settings_no_key)
    cloud_no_key.settings.CLOUD_ENABLED = True
    health_no_key = await cloud_no_key.check_health()
    assert health_no_key["status"] == "missing"
    assert health_no_key["healthy"] is False
    assert await cloud_no_key.is_available() is False

    # 2. Disabled cloud returns status="disabled"
    settings_disabled = Settings(APP_ENV="testing", CLOUD_ENABLED=False, CLOUD_API_KEY="sk-fake")
    cloud_disabled = CloudAIProvider(api_key="sk-fake", settings_obj=settings_disabled)
    health_disabled = await cloud_disabled.check_health()
    assert health_disabled["status"] == "disabled"
    assert health_disabled["healthy"] is False
    assert await cloud_disabled.is_available() is False

    # 3. Cost control: verify max_tokens is passed in generate request payload
    settings_ok = Settings(APP_ENV="testing", CLOUD_ENABLED=True, CLOUD_API_KEY="sk-test-key-123")
    cloud_ok = CloudAIProvider(api_key="sk-test-key-123", settings_obj=settings_ok)

    mock_res = AsyncMock(status_code=200)
    mock_res.json = MagicMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "Proof result"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 42}
    })

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_res
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.cloud_provider.httpx.AsyncClient", return_value=mock_client):
        res = await cloud_ok.generate(
            messages=[LLMMessage(role="user", content="Proof task")],
            max_tokens=500
        )
        assert res.content == "Proof result"
        assert res.provider == "cloud"
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["max_tokens"] == 500
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key-123"


@pytest.mark.parametrize(
    "provider_name,expected_url,expected_model",
    [
        ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
        ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("grok", "https://api.x.ai/v1", "grok-2-latest"),
        ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ],
)
def test_cloud_provider_presets_resolution_in_provider(provider_name, expected_url, expected_model):
    """Verify CloudProvider correctly auto-resolves base_url and model from provider preset."""
    provider = CloudProvider(provider=provider_name)
    assert provider.base_url == expected_url
    assert provider.model == expected_model


def test_cloud_provider_explicit_overrides_in_provider():
    """Verify CloudProvider constructor explicit overrides take precedence."""
    provider = CloudProvider(
        provider="groq",
        base_url="https://my-proxy.com/v1",
        model="custom-llama-model",
    )
    assert provider.base_url == "https://my-proxy.com/v1"
    assert provider.model == "custom-llama-model"


@pytest.mark.asyncio
async def test_cloud_provider_health_check_disabled_and_ok():
    """Verify health_check() returns 'disabled' when no key is set and 'ok' when configured & reachable."""
    from app.config import Settings

    # 1. No key configured -> 'disabled'
    cfg_disabled = Settings(APP_ENV="testing", CLOUD_ENABLED=False, CLOUD_API_KEY=None, _env_file=None)
    provider_no_key = CloudProvider(api_key="", settings_obj=cfg_disabled)
    res_disabled = await provider_no_key.health_check()
    assert res_disabled == "disabled"
    assert res_disabled["status"] == "disabled"
    assert res_disabled["healthy"] is False

    # 2. Key configured and HTTP 200 -> 'ok'
    cfg_ok = Settings(APP_ENV="testing", CLOUD_ENABLED=True, CLOUD_API_KEY="sk-active-key", _env_file=None)
    provider_ok = CloudProvider(api_key="sk-active-key", settings_obj=cfg_ok)

    mock_resp = AsyncMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.cloud_provider.httpx.AsyncClient", return_value=mock_client):
        res_ok = await provider_ok.health_check()
        assert res_ok == "ok"
        assert res_ok["status"] == "ok"
        assert res_ok["healthy"] is True
        mock_client.get.assert_called_once_with(
            f"{provider_ok.base_url}/models",
            headers={"Authorization": "Bearer sk-active-key"},
        )


@pytest.mark.asyncio
async def test_cloud_provider_stream():
    """Verify CloudProvider stream() parses SSE tokens properly."""
    provider = CloudProvider(api_key="sk-stream-key", provider="groq")
    provider.settings.CLOUD_ENABLED = True

    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Deep"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " learning"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " proof"}, "finish_reason": "stop"}]}',
        'data: [DONE]',
    ]
    mock_resp = MockStreamResponse(status_code=200, lines=sse_lines)
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client

    with patch("app.providers.cloud_provider.httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in provider.stream(
            messages=[LLMMessage(role="user", content="Explain deep learning")]
        ):
            chunks.append(chunk)

    assert len(chunks) == 4
    assert chunks[0]["response"] == "Deep"
    assert chunks[1]["response"] == " learning"
    assert chunks[2]["response"] == " proof"
    assert chunks[2]["done"] is True
    assert chunks[3]["done"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 500])
async def test_cloud_provider_http_error_falls_back_to_local_ollama(status_code):
    """Verify CloudProvider HTTP 401/500 errors gracefully trigger router fallback to local Ollama."""
    from app.config import Settings

    local_mock = MockLocalProvider()
    cfg = Settings(APP_ENV="testing", CLOUD_ENABLED=True, CLOUD_API_KEY="sk-failing-key", _env_file=None)
    cfg.RESTRICT_TO_LOCAL = False
    cloud_failing = CloudProvider(api_key="sk-failing-key", settings_obj=cfg)

    # Mock is_available() so router attempts cloud routing
    cloud_failing.is_available = AsyncMock(return_value=True)

    mock_error_resp = AsyncMock(status_code=status_code, text=f"Error {status_code} from Cloud")
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_error_resp
    mock_client.__aenter__.return_value = mock_client

    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_failing, settings_obj=cfg)

    with patch("app.providers.cloud_provider.httpx.AsyncClient", return_value=mock_client):
        messages = [LLMMessage(role="user", content="Calculate polynomial bounds.")]
        res = await router.route_and_generate(
            messages=messages,
            complexity=TaskComplexity.HIGH,
            is_private=False,
        )

        assert res.provider == "ollama"
        assert res.is_fallback is True
        assert "Local model response" in res.content


@pytest.mark.asyncio
async def test_cloud_provider_timeout_falls_back_to_local_ollama():
    """Verify CloudProvider timeout gracefully triggers router fallback to local Ollama."""
    from app.config import Settings

    local_mock = MockLocalProvider()
    cfg = Settings(APP_ENV="testing", CLOUD_ENABLED=True, CLOUD_API_KEY="sk-timeout-key", _env_file=None)
    cfg.RESTRICT_TO_LOCAL = False
    cloud_timeout = CloudProvider(api_key="sk-timeout-key", settings_obj=cfg)

    cloud_timeout.is_available = AsyncMock(return_value=True)

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ReadTimeout("Cloud read timeout")
    mock_client.__aenter__.return_value = mock_client

    router = ModelRouter(local_provider=local_mock, cloud_provider=cloud_timeout, settings_obj=cfg)

    with patch("app.providers.cloud_provider.httpx.AsyncClient", return_value=mock_client):
        messages = [LLMMessage(role="user", content="Complex proof")]
        res = await router.route_and_generate(
            messages=messages,
            complexity=TaskComplexity.HIGH,
            is_private=False,
        )

        assert res.provider == "ollama"
        assert res.is_fallback is True
        assert "Local model response" in res.content



