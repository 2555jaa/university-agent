"""Cloud AI Provider implementation (OpenAI / Gemini compatible)."""

import logging
from typing import Any, Dict, List, Optional
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
        model: Optional[str] = None
    ):
        self.api_key = api_key or settings.CLOUD_API_KEY
        self.base_url = (base_url or settings.CLOUD_BASE_URL).rstrip("/")
        self.model = model or settings.CLOUD_MODEL
        self.timeout = 60.0

    async def is_available(self) -> bool:
        """Check if cloud provider is enabled and key is present."""
        if not settings.CLOUD_ENABLED or not self.api_key:
            return False
        return True

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request to cloud API."""
        if not await self.is_available():
            raise RuntimeError("Cloud AI provider is disabled or missing API key.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens or 2048
        }

        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code != 200:
                    raise RuntimeError(f"Cloud AI API error HTTP {res.status_code}: {res.text}")
                
                data = res.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="cloud",
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason")
                )
        except Exception as e:
            logger.error(f"Cloud AI generation error: {e}")
            raise RuntimeError(f"Cloud provider call failed: {e}") from e
