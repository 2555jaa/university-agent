"""Hybrid AI Routing engine between Local Ollama and Cloud AI providers."""

import logging
import re
from enum import Enum
from typing import Any, List, Optional, Union

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.providers.ollama_provider import OllamaProvider
from app.providers.cloud_provider import CloudAIProvider, CloudProvider

logger = logging.getLogger("university_agent.providers.router")


class TaskComplexity(str, Enum):
    LOW = "low"            # Simple explanation, Q&A, definitions, tutoring
    MEDIUM = "medium"      # Document summarization, quiz evaluation
    HIGH = "high"          # Deep code debugging, complex proofs, cross-paper synthesis


class ModelRouter:
    """Intelligently routes requests to local or cloud models based on privacy and complexity."""

    COMPLEXITY_KEYWORDS = {
        "prove that", "mathematical proof", "formal proof", "theorem",
        "np-complete", "graph reduction", "dynamic programming",
        "complex algorithm", "deep code debugging", "cross-paper synthesis",
        "symbolic logic", "formal verification", "eigenvalue", "differential equation",
        "quantum mechanics", "quantum cryptography", "asymptotic complexity",
        "theoretical derivation", "derive the equation", "polynomial bounds"
    }

    def __init__(
        self,
        local_provider: Optional[BaseLLMProvider] = None,
        cloud_provider: Optional[BaseLLMProvider] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.local_provider = local_provider or OllamaProvider(settings_obj=self.settings)
        self.cloud_provider = cloud_provider or CloudAIProvider(settings_obj=self.settings)

    @staticmethod
    def _normalize_messages(messages: List[Any]) -> List[LLMMessage]:
        """Ensure all elements in messages are LLMMessage instances."""
        normalized: List[LLMMessage] = []
        for m in messages:
            if isinstance(m, LLMMessage):
                normalized.append(m)
            elif isinstance(m, dict):
                normalized.append(LLMMessage(role=m.get("role", "user"), content=m.get("content", "")))
            elif isinstance(m, str):
                normalized.append(LLMMessage(role="user", content=m))
        return normalized

    @staticmethod
    def contains_sensitive_rag_context(messages: List[Any]) -> bool:
        """Detect whether prompt contains injected RAG document context or XML source tags."""
        rag_tag_pattern = re.compile(
            r'<\s*/?\s*(?:context|source|retrieved_context)(?:\s+[^>]*)?>',
            re.IGNORECASE
        )
        for m in messages:
            content = m.content if hasattr(m, "content") else (m.get("content", "") if isinstance(m, dict) else str(m))
            if rag_tag_pattern.search(content):
                return True
        return False

    def determine_complexity(
        self,
        messages: List[Any],
        default: TaskComplexity = TaskComplexity.LOW
    ) -> TaskComplexity:
        """Heuristic analysis of messages to determine if task requires cloud-level reasoning."""
        combined_text = " ".join(
            (m.content if hasattr(m, "content") else (m.get("content", "") if isinstance(m, dict) else str(m)))
            for m in messages
        ).lower()

        for kw in self.COMPLEXITY_KEYWORDS:
            if kw in combined_text:
                return TaskComplexity.HIGH

        return default

    async def route_query(
        self,
        messages: List[Any],
        complexity: Optional[TaskComplexity] = None,
        is_private: bool = False,
        mode: Optional[str] = None,
    ) -> str:
        """Determine target provider ("local" or "cloud") enforcing privacy guardrails and complexity."""
        # 1. Strict Privacy Guardrail: RESTRICT_TO_LOCAL setting
        if getattr(self.settings, "RESTRICT_TO_LOCAL", True):
            logger.debug("Routing to local: RESTRICT_TO_LOCAL is active.")
            return "local"

        # 2. Strict Privacy Guardrail: is_private flag
        if is_private:
            logger.debug("Routing to local: Query marked is_private=True.")
            return "local"

        # 3. Sensitive Data Fencing: RAG context / XML tags detected in prompt
        if self.contains_sensitive_rag_context(messages):
            logger.info("Routing to local: Sensitive RAG document context detected in prompt.")
            return "local"

        # 4. Check if Cloud is enabled in configuration
        if not getattr(self.settings, "CLOUD_ENABLED", False):
            logger.debug("Routing to local: CLOUD_ENABLED is False.")
            return "local"

        # 5. Check cloud provider readiness / API key availability
        if not await self.cloud_provider.is_available():
            logger.warning("Routing to local: Cloud provider is offline or missing API key.")
            return "local"

        # 6. Mode & Complexity evaluation:
        # When cloud is enabled and unconstrained, do not block tutor or coder queries from cloud.
        if mode in ("coder", "tutor", "research"):
            return "cloud"

        effective_complexity = complexity
        if effective_complexity is None or effective_complexity == TaskComplexity.LOW:
            effective_complexity = self.determine_complexity(messages, default=complexity or TaskComplexity.LOW)

        if effective_complexity == TaskComplexity.HIGH:
            return "cloud"

        logger.debug(f"Routing to local: Task complexity is {effective_complexity}.")
        return "local"

    async def route_and_generate(
        self,
        messages: List[LLMMessage],
        complexity: Optional[TaskComplexity] = None,
        is_private: bool = False,
        mode: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
    ) -> LLMResponse:
        """Route request according to security policies, cost controls, and failover gracefully."""
        normalized_messages = self._normalize_messages(messages)
        target_provider = await self.route_query(
            messages=normalized_messages,
            complexity=complexity,
            is_private=is_private,
            mode=mode,
        )

        if target_provider == "cloud":
            try:
                logger.info("Routing request to Cloud AI Provider for advanced reasoning.")
                return await self.cloud_provider.generate(
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                logger.warning(f"Cloud provider failed ({e}), falling back to local Ollama.")
                fallback_res = await self.local_provider.generate(
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                fallback_res.is_fallback = True
                return fallback_res

        # Default to local model
        logger.info(f"Routing request to Local Ollama model (Private={is_private}, Complexity={complexity}).")
        try:
            return await self.local_provider.generate(
                messages=normalized_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            # Check if failover to cloud is permissible (MUST NOT be private or contain sensitive RAG context)
            can_failover_to_cloud = (
                not getattr(self.settings, "RESTRICT_TO_LOCAL", True)
                and not is_private
                and not self.contains_sensitive_rag_context(normalized_messages)
                and getattr(self.settings, "CLOUD_ENABLED", False)
                and await self.cloud_provider.is_available()
            )
            if can_failover_to_cloud:
                logger.warning(f"Local provider failed ({e}), failing over to Cloud AI Provider.")
                cloud_res = await self.cloud_provider.generate(
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                cloud_res.is_fallback = True
                return cloud_res
            raise

    async def route_and_stream(
        self,
        messages: List[LLMMessage],
        complexity: Optional[TaskComplexity] = None,
        is_private: bool = False,
        mode: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
    ):
        """Route request according to security policies and stream tokens."""
        normalized_messages = self._normalize_messages(messages)
        target_provider = await self.route_query(
            messages=normalized_messages,
            complexity=complexity,
            is_private=is_private,
            mode=mode,
        )

        if target_provider == "cloud" and hasattr(self.cloud_provider, "generate_stream"):
            try:
                async for chunk in self.cloud_provider.generate_stream(
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Cloud stream failed ({e}), falling back to local Ollama.")

        # Default to local stream
        async for chunk in self.local_provider.generate_stream(
            messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield chunk


router = ModelRouter()
