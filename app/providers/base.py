"""Base LLM Provider interface and response structures."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str  # "ollama" or "cloud"
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    is_fallback: bool = False


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the model."""
        pass

    async def generate_stream(
        self,
        messages: Optional[List[LLMMessage]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens sequentially if supported by provider."""
        res = await self.generate(
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        yield {
            "message": {"role": "assistant", "content": res.content},
            "response": res.content,
            "done": True
        }

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is online and reachable."""
        pass
