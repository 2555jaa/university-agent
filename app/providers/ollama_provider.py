"""Ollama local model provider implementation."""

import logging
from typing import Any, Dict, List, Optional
import httpx

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger("university_agent.providers.ollama")


class OllamaProvider(BaseLLMProvider):
    """Local inference provider via Ollama HTTP API."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.LOCAL_MODEL
        self.timeout = 120.0

    async def is_available(self) -> bool:
        """Check if Ollama service is listening and responsive."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/version")
                return res.status_code == 200
        except Exception:
            return False

    async def get_loaded_models(self) -> List[str]:
        """Fetch list of local models available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to query Ollama models: {e}")
        return []

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request to local Ollama daemon."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 2048
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Ollama returned HTTP {res.status_code}: {res.text}")
                
                data = res.json()
                content = data.get("message", {}).get("content", "")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="ollama",
                    usage={
                        "total_duration": data.get("total_duration"),
                        "eval_count": data.get("eval_count"),
                        "eval_duration": data.get("eval_duration")
                    },
                    finish_reason=data.get("done_reason", "stop")
                )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Local AI inference error: {e}") from e

    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding vector for RAG indexing via Ollama."""
        embed_model = model or settings.EMBEDDING_MODEL or self.model
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": embed_model, "prompt": text}
                )
                if res.status_code == 200:
                    return res.json().get("embedding", [])
                
                # Fallback to /api/embed if newer endpoint
                res2 = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": embed_model, "input": text}
                )
                if res2.status_code == 200:
                    embeddings = res2.json().get("embeddings", [])
                    if embeddings:
                        return embeddings[0]
        except Exception as e:
            logger.warning(f"Ollama embedding generation failed: {e}")
        return []
