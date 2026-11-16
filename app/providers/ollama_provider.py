"""Ollama local model provider implementation."""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import httpx

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger("university_agent.providers.ollama")

DEFAULT_OLLAMA_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ModelNotFoundError(ProviderError, RuntimeError):
    """Raised when the specified model is missing on the Ollama instance."""
    pass


class StreamInterruptedError(ProviderError, RuntimeError):
    """Raised when streaming from Ollama is interrupted mid-generation."""
    pass


class OllamaProvider(BaseLLMProvider):
    """Local inference provider via Ollama HTTP API with resilient streaming."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.base_url = (base_url or self.settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or self.settings.LOCAL_MODEL
        if timeout is None:
            self.timeout = DEFAULT_OLLAMA_TIMEOUT
        elif isinstance(timeout, httpx.Timeout):
            self.timeout = timeout
        elif isinstance(timeout, (int, float)):
            self.timeout = httpx.Timeout(connect=5.0, read=float(timeout), write=10.0, pool=5.0)
        else:
            self.timeout = DEFAULT_OLLAMA_TIMEOUT

    @staticmethod
    def _format_messages(
        messages: Optional[Union[List[LLMMessage], List[Dict[str, str]]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        if messages:
            for m in messages:
                if isinstance(m, LLMMessage):
                    formatted.append({"role": m.role, "content": m.content})
                elif isinstance(m, dict):
                    formatted.append({
                        "role": m.get("role", "user"),
                        "content": m.get("content", "")
                    })
        elif prompt:
            formatted.append({"role": "user", "content": prompt})

        return formatted

    @staticmethod
    def _build_options(
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        extra_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if extra_options:
            options.update(extra_options)
        return options

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
        messages: Optional[Union[List[LLMMessage], List[Dict[str, str]]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        raise_on_error: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request to local Ollama daemon."""
        formatted_messages = self._format_messages(messages, prompt, system_prompt)
        options = self._build_options(temperature, max_tokens, kwargs.get("options"))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": options
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 404:
                    error_msg = f"Model '{self.model}' not found in local Ollama repository. Please run: ollama pull {self.model}"
                    logger.error(error_msg)
                    if raise_on_error:
                        raise ModelNotFoundError(error_msg)
                    return LLMResponse(
                        content=f"Error: {error_msg}",
                        model=self.model,
                        provider="ollama",
                        finish_reason="model_not_found",
                        usage={"error": "model_not_found", "status_code": 404}
                    )
                elif res.status_code != 200:
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
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Local AI inference error: {e}") from e

    async def generate_stream(
        self,
        messages: Optional[Union[List[LLMMessage], List[Dict[str, str]]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tokens sequentially from Ollama /api/chat with bounded timeouts and error resilience."""
        formatted_messages = self._format_messages(messages, prompt, system_prompt)
        options = self._build_options(temperature, max_tokens, kwargs.get("options"))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "options": options
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code == 404:
                        error_msg = f"Model '{self.model}' not found in local Ollama repository. Please run: ollama pull {self.model}"
                        logger.error(error_msg)
                        yield {
                            "error": error_msg,
                            "error_type": "model_not_found",
                            "status_code": 404,
                            "message": {"role": "assistant", "content": error_msg},
                            "response": error_msg,
                            "done": True
                        }
                        return
                    elif response.status_code != 200:
                        raw_body = await response.aread()
                        err_text = raw_body.decode("utf-8", errors="replace")
                        error_msg = f"Ollama returned HTTP {response.status_code}: {err_text}"
                        logger.error(error_msg)
                        yield {
                            "error": error_msg,
                            "error_type": "http_error",
                            "status_code": response.status_code,
                            "message": {"role": "assistant", "content": error_msg},
                            "response": error_msg,
                            "done": True
                        }
                        return

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Ensure both response and message are present for diverse consumers
                        if "response" not in chunk and "message" in chunk:
                            chunk["response"] = chunk["message"].get("content", "")
                        elif "message" not in chunk and "response" in chunk:
                            chunk["message"] = {"role": "assistant", "content": chunk.get("response", "")}

                        yield chunk
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPError, asyncio.CancelledError, Exception) as e:
            logger.error(f"Ollama stream interrupted or failed mid-stream: {e}")
            yield {
                "error": "[STREAM_INTERRUPTED_ERROR]",
                "error_type": "stream_interrupted",
                "detail": str(e),
                "message": {
                    "role": "assistant",
                    "content": " [STREAM_INTERRUPTED_ERROR]"
                },
                "response": " [STREAM_INTERRUPTED_ERROR]",
                "done": True
            }

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
