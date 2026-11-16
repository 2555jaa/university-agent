"""Unit tests for provider abstraction and hybrid AI routing."""

import pytest
from typing import List, Optional

from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.providers.router import ModelRouter, TaskComplexity


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

    messages = [LLMMessage(role="user", content="Complex research inquiry.")]
    res = await router.route_and_generate(
        messages=messages,
        complexity=TaskComplexity.HIGH,
        is_private=False
    )

    assert res.provider == "ollama"
    assert res.is_fallback is True
