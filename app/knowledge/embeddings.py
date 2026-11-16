"""Pluggable embedding providers for academic RAG retrieval."""

import math
import logging
import re
import zlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("university_agent.knowledge.embeddings")

STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "it", "that", "this", "for", "on", "with",
    "as", "at", "by", "from", "an", "be", "or", "are", "was", "were", "been"
}


class EmbeddingInfo(BaseModel):
    provider: str  # "ollama", "deterministic_hash"
    model: str     # "nomic-embed-text", "hash-crc32-256"
    dimension: int
    is_neural: bool
    status: str = "active"
    details: Dict[str, Any] = Field(default_factory=dict)


class BaseEmbeddingProvider(ABC):
    """Abstract interface for all embedding providers."""

    @abstractmethod
    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate a normalized embedding vector for the given text."""
        pass

    @abstractmethod
    async def get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate normalized embedding vectors for a batch of texts."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider and model are available."""
        pass

    @abstractmethod
    def get_info(self) -> EmbeddingInfo:
        """Return metadata about the embedding model and dimensions."""
        pass


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Neural dense embedding provider using local Ollama model (e.g. nomic-embed-text)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: int = 768,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.base_url = (base_url or self.settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or self.settings.EMBEDDING_MODEL
        self.dimension = dimension
        self.timeout = 20.0
        self._available_cache: Optional[bool] = None

    async def is_available(self) -> bool:
        """Verify that Ollama daemon is reachable and the embedding model is installed."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code != 200:
                    self._available_cache = False
                    return False
                models = [m.get("name", "").split(":")[0] for m in res.json().get("models", [])]
                target_base = self.model.split(":")[0]
                available = target_base in models or any(self.model in m.get("name", "") for m in res.json().get("models", []))
                self._available_cache = available
                return available
        except Exception as e:
            logger.debug(f"Ollama embedding check failed: {e}")
            self._available_cache = False
            return False

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Call Ollama /api/embeddings or /api/embed and return normalized vector."""
        # Nomic embed text convention: prefix search_query or search_document if relevant
        formatted_text = text
        if "nomic" in self.model.lower():
            prefix = "search_query: " if is_query else "search_document: "
            formatted_text = prefix + text.strip()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Try /api/embeddings
                res = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": formatted_text}
                )
                if res.status_code == 200:
                    vec = res.json().get("embedding", [])
                    if vec:
                        norm = math.sqrt(sum(x * x for x in vec))
                        if norm > 0:
                            vec = [x / norm for x in vec]
                        self.dimension = len(vec)
                        return vec

                # 2. Fallback to /api/embed (newer Ollama API)
                res2 = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": formatted_text}
                )
                if res2.status_code == 200:
                    embeddings = res2.json().get("embeddings", [])
                    if embeddings and embeddings[0]:
                        vec = embeddings[0]
                        norm = math.sqrt(sum(x * x for x in vec))
                        if norm > 0:
                            vec = [x / norm for x in vec]
                        self.dimension = len(vec)
                        return vec

                raise RuntimeError(f"Ollama embedding error HTTP {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"Failed to generate neural embedding via Ollama ({self.model}): {e}")
            raise

    async def get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        results = []
        for t in texts:
            results.append(await self.get_embedding(t, is_query=is_query))
        return results

    def get_info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="ollama",
            model=self.model,
            dimension=self.dimension,
            is_neural=True,
            status="active" if self._available_cache else "unreachable",
            details={"base_url": self.base_url}
        )


class DeterministicHashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, zero-dependency token hash vector provider (dim=256).

    Used when explicitly configured or when offline without local neural models.
    NEVER presented as neural embeddings.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    async def is_available(self) -> bool:
        return True

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        vec = [0.0] * self.dimension
        words = re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())
        for w in words:
            if w in STOPWORDS:
                continue
            h = zlib.crc32(w.encode("utf-8")) % self.dimension
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        return [await self.get_embedding(t, is_query=is_query) for t in texts]

    def get_info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="deterministic_hash",
            model=f"hash-crc32-{self.dimension}",
            dimension=self.dimension,
            is_neural=False,
            status="fallback_active",
            details={"algorithm": "crc32", "stopword_filtering": True}
        )


class EmbeddingManager:
    """Manages active embedding provider selection and model switching."""

    def __init__(
        self,
        settings_obj: Optional[Any] = None,
        ollama_provider: Optional[BaseEmbeddingProvider] = None,
        hash_provider: Optional[BaseEmbeddingProvider] = None
    ):
        self.settings = settings_obj or settings
        self.ollama_provider = ollama_provider or OllamaEmbeddingProvider(settings_obj=self.settings)
        self.hash_provider = hash_provider or DeterministicHashEmbeddingProvider()
        self._active_provider: Optional[BaseEmbeddingProvider] = None

    async def get_provider(self, force_refresh: bool = False) -> BaseEmbeddingProvider:
        """Select active embedding provider based on availability and configuration."""
        if self._active_provider is not None and not force_refresh:
            return self._active_provider

        # Check configured preference
        pref = getattr(self.settings, "EMBEDDING_PROVIDER", "auto").lower()

        if pref == "deterministic_hash":
            logger.info("Using configured DeterministicHashEmbeddingProvider (non-neural).")
            self._active_provider = self.hash_provider
            return self._active_provider

        # Auto or Ollama: check if neural model is available
        ollama_ok = await self.ollama_provider.is_available()
        if ollama_ok:
            logger.info(f"Active embedding provider: Ollama ({self.ollama_provider.model}, neural).")
            self._active_provider = self.ollama_provider
            return self._active_provider

        # Explicit fallback with clear warning
        logger.warning(
            f"Neural embedding model '{settings.EMBEDDING_MODEL}' is not available in Ollama. "
            "Falling back to DeterministicHashEmbeddingProvider (non-neural token vectors)."
        )
        self._active_provider = self.hash_provider
        return self._active_provider

    async def get_active_info(self) -> EmbeddingInfo:
        provider = await self.get_provider()
        return provider.get_info()


embedding_manager = EmbeddingManager()
