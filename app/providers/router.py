"""Hybrid AI Routing engine between Local Ollama and Cloud AI providers."""

import logging
from enum import Enum
from typing import List, Optional

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.providers.ollama_provider import OllamaProvider
from app.providers.cloud_provider import CloudAIProvider

logger = logging.getLogger("university_agent.providers.router")


class TaskComplexity(str, Enum):
    LOW = "low"            # Simple explanation, Q&A, definitions, tutoring
    MEDIUM = "medium"      # Document summarization, quiz evaluation
    HIGH = "high"          # Deep code debugging, complex proofs, cross-paper synthesis


class ModelRouter:
    """Intelligently routes requests to local or cloud models based on privacy and complexity."""

    def __init__(
        self,
        local_provider: Optional[BaseLLMProvider] = None,
        cloud_provider: Optional[BaseLLMProvider] = None
    ):
        self.local_provider = local_provider or OllamaProvider()
        self.cloud_provider = cloud_provider or CloudAIProvider()

    async def route_and_generate(
        self,
        messages: List[LLMMessage],
        complexity: TaskComplexity = TaskComplexity.LOW,
        is_private: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048
    ) -> LLMResponse:
        """
        Route request according to security policies and resource constraints:
        - If is_private is True: ALWAYS strictly local.
        - If complexity is HIGH and cloud is available: Route to cloud.
        - Fallback: Always fall back gracefully to local Ollama model.
        """
        target_provider = "local"

        if not is_private and complexity == TaskComplexity.HIGH:
            if await self.cloud_provider.is_available():
                target_provider = "cloud"

        if target_provider == "cloud":
            try:
                logger.info("Routing request to Cloud AI Provider for advanced reasoning.")
                return await self.cloud_provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                logger.warning(f"Cloud provider failed ({e}), falling back to local Ollama.")
                fallback_res = await self.local_provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                fallback_res.is_fallback = True
                return fallback_res

        # Default to local model
        logger.info(f"Routing request to Local Ollama model (Private={is_private}, Complexity={complexity}).")
        return await self.local_provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )


router = ModelRouter()
