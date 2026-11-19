"""Unified Cloud AI Provider implementation (OpenAI, Gemini, Groq, Grok, OpenRouter, Custom)."""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import httpx

from app.config import CLOUD_PROVIDER_PRESETS, settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger("university_agent.providers.cloud")


class HealthResult(dict):
    """Result object that acts as a dictionary while supporting equality checks against string statuses."""

    @property
    def status(self) -> str:
        return self.get("status", "")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("status") == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return self.get("status", "")


class CloudProvider(BaseLLMProvider):
    """Unified Cloud AI provider supporting OpenAI-compatible APIs across multiple cloud vendors."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        settings_obj: Optional[Any] = None,
    ):
        self.settings = settings_obj or settings
        self.api_key = api_key or getattr(self.settings, "CLOUD_API_KEY", None)

        if provider is not None:
            self.provider = provider.strip().lower()
            preset = CLOUD_PROVIDER_PRESETS.get(self.provider, {})
            self.base_url = (base_url or preset.get("base_url", "https://api.openai.com/v1")).rstrip("/")
            self.model = model or preset.get("default_model", "gpt-4o-mini")
        else:
            self.provider = getattr(self.settings, "CLOUD_PROVIDER", "openai").strip().lower()
            preset = CLOUD_PROVIDER_PRESETS.get(self.provider, {})
            resolved_base_url = (
                base_url
                or getattr(self.settings, "CLOUD_BASE_URL", None)
                or preset.get("base_url", "https://api.openai.com/v1")
            )
            self.base_url = resolved_base_url.rstrip("/")
            self.model = (
                model
                or getattr(self.settings, "CLOUD_MODEL", None)
                or preset.get("default_model", "gpt-4o-mini")
            )

        if timeout is None:
            self.timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        elif isinstance(timeout, httpx.Timeout):
            self.timeout = timeout
        elif isinstance(timeout, (int, float)):
            self.timeout = httpx.Timeout(connect=5.0, read=float(timeout), write=10.0, pool=5.0)
        else:
            self.timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)

    async def check_health(self) -> HealthResult:
        """Check cloud provider readiness, API key configuration, and availability."""
        if not getattr(self.settings, "CLOUD_ENABLED", False):
            return HealthResult({
                "status": "disabled",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud AI provider is disabled in settings (CLOUD_ENABLED=False).",
            })
        if not self.api_key or not str(self.api_key).strip():
            return HealthResult({
                "status": "missing",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud API key is missing (CLOUD_API_KEY is not configured).",
            })
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if res.status_code == 200:
                    return HealthResult({
                        "status": "ok",
                        "healthy": True,
                        "provider": "cloud",
                        "model": self.model,
                    })
                return HealthResult({
                    "status": "offline",
                    "healthy": False,
                    "provider": "cloud",
                    "error": f"Cloud provider returned HTTP {res.status_code}",
                })
        except Exception as e:
            return HealthResult({
                "status": "offline",
                "healthy": False,
                "provider": "cloud",
                "error": str(e),
            })

    async def health_check(self) -> HealthResult:
        """Accurately report 'ok' when CLOUD_API_KEY is configured and reachable, or 'disabled' when no key is set."""
        if not self.api_key or not str(self.api_key).strip():
            return HealthResult({
                "status": "disabled",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud API key is not configured (disabled).",
            })
        if not getattr(self.settings, "CLOUD_ENABLED", False):
            return HealthResult({
                "status": "disabled",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud AI provider is disabled in settings (CLOUD_ENABLED=False).",
            })
        return await self.check_health()

    async def is_available(self) -> bool:
        """Check if cloud provider is enabled and key is present and healthy."""
        if not getattr(self.settings, "CLOUD_ENABLED", False) or not self.api_key or not str(self.api_key).strip():
            return False
        health = await self.check_health()
        return health.get("status") not in ("offline", "missing", "disabled") and health.get("healthy", False)

    @staticmethod
    def _format_messages(
        messages: Optional[Union[List[LLMMessage], List[Dict[str, str]], List[Any]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Format polymorphic message inputs into OpenAI-compatible chat message dicts."""
        formatted: List[Dict[str, str]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        if messages:
            for m in messages:
                if isinstance(m, LLMMessage):
                    formatted.append({"role": m.role, "content": m.content})
                elif isinstance(m, dict):
                    formatted.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                else:
                    formatted.append({"role": "user", "content": str(m)})
        if prompt:
            formatted.append({"role": "user", "content": prompt})
        return formatted

    async def generate(
        self,
        messages: List[Union[LLMMessage, Dict[str, Any], str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Send chat completion request to cloud API with cost control max_tokens."""
        if not self.api_key or not str(self.api_key).strip():
            raise RuntimeError("Cloud AI provider is disabled or missing API key.")

        formatted_messages = self._format_messages(messages=messages)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        token_limit = max_tokens if max_tokens is not None else 2048
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": token_limit,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code != 200:
                    raise RuntimeError(f"Cloud AI API error HTTP {res.status_code}: {res.text}")

                data = res.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"Cloud AI API returned empty choices: {data}")
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="cloud",
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
        except Exception as e:
            logger.error(f"Cloud AI generation error: {e}")
            raise RuntimeError(f"Cloud provider call failed: {e}") from e

    async def generate_stream(
        self,
        messages: Optional[Union[List[LLMMessage], List[Dict[str, str]]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tokens sequentially from OpenAI-compatible cloud endpoint."""
        if not self.api_key or not str(self.api_key).strip():
            raise RuntimeError("Cloud AI provider is disabled or missing API key.")

        formatted_messages = self._format_messages(
            messages=messages, prompt=prompt, system_prompt=system_prompt
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        token_limit = max_tokens if max_tokens is not None else 2048
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": token_limit,
            "stream": True,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        raw_body = await response.aread()
                        err_text = raw_body.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Cloud provider returned HTTP {response.status_code}: {err_text}"
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                        else:
                            continue

                        if data_str == "[DONE]":
                            yield {
                                "model": self.model,
                                "message": {"role": "assistant", "content": ""},
                                "response": "",
                                "done": True,
                            }
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        delta_content = delta.get("content", "") or ""
                        finish_reason = choice.get("finish_reason")

                        yield {
                            "model": self.model,
                            "message": {"role": "assistant", "content": delta_content},
                            "response": delta_content,
                            "done": finish_reason is not None,
                            "finish_reason": finish_reason,
                        }
        except Exception as e:
            logger.error(f"Cloud AI streaming error: {e}")
            raise RuntimeError(f"Cloud streaming call failed: {e}") from e

    stream = generate_stream


# Backward compatibility alias
CloudAIProvider = CloudProvider

__all__ = ["CloudProvider", "CloudAIProvider", "HealthResult", "CLOUD_PROVIDER_PRESETS"]
