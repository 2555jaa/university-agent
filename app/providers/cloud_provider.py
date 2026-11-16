"""Cloud AI Provider implementation (OpenAI / Gemini compatible)."""

import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger("university_agent.providers.cloud")


class CloudAIProvider(BaseLLMProvider):
    """Cloud AI model provider for complex academic reasoning tasks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.api_key = api_key or getattr(self.settings, "CLOUD_API_KEY", None)
        self.base_url = (base_url or getattr(self.settings, "CLOUD_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")).rstrip("/")
        self.model = model or getattr(self.settings, "CLOUD_MODEL", "gemini-2.5-flash")
        if timeout is None:
            self.timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        elif isinstance(timeout, httpx.Timeout):
            self.timeout = timeout
        elif isinstance(timeout, (int, float)):
            self.timeout = httpx.Timeout(connect=5.0, read=float(timeout), write=10.0, pool=5.0)
        else:
            self.timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)

    async def check_health(self) -> Dict[str, Any]:
        """Check cloud provider readiness, API key configuration, and availability."""
        if not getattr(self.settings, "CLOUD_ENABLED", False):
            return {
                "status": "disabled",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud AI provider is disabled in settings (CLOUD_ENABLED=False)."
            }
        if not self.api_key or not str(self.api_key).strip():
            return {
                "status": "missing",
                "healthy": False,
                "provider": "cloud",
                "error": "Cloud API key is missing (CLOUD_API_KEY is not configured)."
            }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                if res.status_code == 200:
                    return {
                        "status": "ready",
                        "healthy": True,
                        "provider": "cloud",
                        "model": self.model
                    }
                return {
                    "status": "offline",
                    "healthy": False,
                    "provider": "cloud",
                    "error": f"Cloud provider returned HTTP {res.status_code}"
                }
        except Exception as e:
            return {
                "status": "offline",
                "healthy": False,
                "provider": "cloud",
                "error": str(e)
            }

    async def is_available(self) -> bool:
        """Check if cloud provider is enabled and key is present and healthy."""
        if not getattr(self.settings, "CLOUD_ENABLED", False) or not self.api_key or not str(self.api_key).strip():
            return False
        health = await self.check_health()
        return health.get("status") not in ("offline", "missing", "disabled") and health.get("healthy", False)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request to cloud API with cost control max_tokens."""
        if not getattr(self.settings, "CLOUD_ENABLED", False) or not self.api_key or not str(self.api_key).strip():
            raise RuntimeError("Cloud AI provider is disabled or missing API key.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Cost control: enforce max_tokens limit
        token_limit = max_tokens if max_tokens is not None else 2048
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": token_limit
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
                    finish_reason=choice.get("finish_reason", "stop")
                )
        except Exception as e:
            logger.error(f"Cloud AI generation error: {e}")
            raise RuntimeError(f"Cloud provider call failed: {e}") from e
