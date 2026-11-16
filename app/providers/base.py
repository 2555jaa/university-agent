"""Base LLM Provider interface and response structures."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
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

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is online and reachable."""
        pass
