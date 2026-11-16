"""Pluggable embedding providers for academic RAG retrieval."""

import asyncio
import logging
import math
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

TRANSIENT_STATUS_CODES = {502, 503, 504}
TRANSIENT_EXCEPTIONS = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError)


class EmbeddingError(Exception):
    """Base exception for embedding operations."""
    pass


class ModelNotFoundError(EmbeddingError):
    """Raised when the requested embedding model is not found in Ollama."""
    pass


class ProviderUnavailableError(EmbeddingError):
    """Raised when the configured embedding provider is unreachable or offline."""
    pass


class ModelMigrationError(EmbeddingError):
    """Raised when active embedding dimensions or model conflict with existing index."""
    pass


class VectorValidationError(EmbeddingError, ValueError):
    """Raised when an embedding vector is malformed, has NaN/Inf, zero norm, or dimension mismatch."""
    pass


def format_text_for_model(text: str, model_name: str, is_query: bool = False) -> str:
    """Apply asymmetric prefixes ('search_document: ' or 'search_query: ') for Nomic models.
    
    Ensures prefixes are not double-applied if already present.
    For non-Nomic models, leaves text intact.
    """
    if not text:
        return ""
    cleaned = text.strip()
    if "nomic" in model_name.lower():
        prefix = "search_query: " if is_query else "search_document: "
        if cleaned.startswith("search_query: ") or cleaned.startswith("search_document: "):
            return cleaned
        return f"{prefix}{cleaned}"
    return text


def validate_and_normalize_vector(
    vec: List[float],
    expected_dim: Optional[int] = None,
    allow_zero_norm: bool = False
) -> List[float]:
    """Validate vector integrity (non-empty, numeric, no NaN/Inf, non-zero L2 norm) and L2 normalize."""
    if not vec or not isinstance(vec, (list, tuple)):
        raise VectorValidationError("Embedding vector must be a non-empty list of floats.")

    if expected_dim is not None and len(vec) != expected_dim:
        raise VectorValidationError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}."
        )

    norm_sq = 0.0
    for i, x in enumerate(vec):
        if not isinstance(x, (int, float)):
            raise VectorValidationError(f"Vector component at index {i} is not numeric: {type(x).__name__}")
        if math.isnan(x):
            raise VectorValidationError(f"Vector component at index {i} is NaN.")
        if math.isinf(x):
            raise VectorValidationError(f"Vector component at index {i} is infinite ({x}).")
        norm_sq += float(x) * float(x)

    norm = math.sqrt(norm_sq)
    if norm == 0.0:
        if allow_zero_norm:
            return [0.0] * len(vec)
        raise VectorValidationError("Embedding vector has zero L2 norm (all components are 0.0).")

    return [float(x) / norm for x in vec]


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
    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False, batch_size: int = 16
    ) -> List[List[float]]:
        """Generate normalized embedding vectors for a batch of texts."""
        pass

    async def embed_text(self, text: str, input_type: str = "document") -> List[float]:
        """Embed a single text string as document or query."""
        return await self.get_embedding(text, is_query=(input_type == "query"))

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query string."""
        return await self.get_embedding(query, is_query=True)

    async def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed a list of document texts in batches."""
        return await self.get_embeddings_batch(texts, is_query=False, batch_size=batch_size)

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
        settings_obj: Optional[Any] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        batch_size: int = 16,
        initial_delay: float = 0.05,
        backoff_factor: float = 2.0,
    ):
        self.settings = settings_obj or settings
        self.base_url = (base_url or self.settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or self.settings.EMBEDDING_MODEL
        self.dimension = dimension
        self.max_retries = max(1, max_retries)
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self._available_cache: Optional[bool] = None
        self._supports_api_embed: Optional[bool] = None

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

    async def _send_request_with_retries(
        self, client: httpx.AsyncClient, url: str, json_data: Dict[str, Any]
    ) -> httpx.Response:
        """Send POST request with bounded retries for transient failures."""
        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                res = await client.post(url, json=json_data)
                # Transient HTTP server errors: retry
                if res.status_code in TRANSIENT_STATUS_CODES:
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                        logger.warning(
                            f"Ollama {url} returned HTTP {res.status_code}, retrying in {delay:.2f}s "
                            f"(attempt {attempt}/{self.max_retries})..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    return res
                # Return other responses immediately (200, 404, 400, etc.)
                return res
            except TRANSIENT_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"Ollama network error ({type(e).__name__}): {e}, retrying in {delay:.2f}s "
                        f"(attempt {attempt}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"Ollama request to {url} failed after {self.max_retries} attempts: {e}"
                ) from e

        if last_exception:
            raise EmbeddingError(
                f"Ollama request to {url} failed after {self.max_retries} attempts: {last_exception}"
            ) from last_exception
        raise EmbeddingError(f"Ollama request to {url} failed after {self.max_retries} attempts.")

    async def _embed_batch_via_api_embed(
        self, client: httpx.AsyncClient, batch: List[str]
    ) -> Optional[List[List[float]]]:
        """Call POST /api/embed with {"model": self.model, "input": batch}."""
        payload = {"model": self.model, "input": batch}
        res = await self._send_request_with_retries(client, f"{self.base_url}/api/embed", payload)

        if res.status_code == 200:
            data = res.json()
            embeddings = data.get("embeddings", [])
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise VectorValidationError(
                    f"Expected {len(batch)} embeddings from /api/embed, got {len(embeddings)}."
                )
            return embeddings

        if res.status_code == 404:
            txt = res.text.lower()
            if "model" in txt and "not found" in txt:
                raise ModelNotFoundError(
                    f"Model '{self.model}' not found in Ollama. Please run 'ollama pull {self.model}'."
                )
            # Route not found on older Ollama versions (<0.1.34)
            logger.debug(f"/api/embed not supported on Ollama server ({res.text}). Falling back to /api/embeddings.")
            self._supports_api_embed = False
            return None

        raise EmbeddingError(f"Ollama /api/embed error HTTP {res.status_code}: {res.text}")

    async def _embed_single_via_api_embeddings(
        self, client: httpx.AsyncClient, prompt: str
    ) -> List[float]:
        """Call POST /api/embeddings with {"model": self.model, "prompt": prompt}."""
        payload = {"model": self.model, "prompt": prompt}
        res = await self._send_request_with_retries(client, f"{self.base_url}/api/embeddings", payload)

        if res.status_code == 200:
            data = res.json()
            vec = data.get("embedding", [])
            if not isinstance(vec, list) or not vec:
                raise VectorValidationError("Empty embedding vector received from /api/embeddings.")
            return vec

        if res.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{self.model}' not found in Ollama. Please run 'ollama pull {self.model}'."
            )

        raise EmbeddingError(f"Ollama /api/embeddings error HTTP {res.status_code}: {res.text}")

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False, batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate normalized embedding vectors for a list of texts in batches."""
        if not texts:
            return []

        effective_batch_size = max(1, batch_size or self.batch_size)
        formatted_texts = [format_text_for_model(t, self.model, is_query=is_query) for t in texts]

        timeout_cfg = httpx.Timeout(connect=5.0, read=self.timeout, write=10.0, pool=10.0)
        all_vectors: List[List[float]] = []

        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            for i in range(0, len(formatted_texts), effective_batch_size):
                batch = formatted_texts[i : i + effective_batch_size]
                batch_vectors: Optional[List[List[float]]] = None

                # 1. Try batch endpoint /api/embed if not marked unsupported
                if self._supports_api_embed is not False:
                    batch_vectors = await self._embed_batch_via_api_embed(client, batch)
                    if batch_vectors is not None:
                        self._supports_api_embed = True

                # 2. Fallback to /api/embeddings for each item
                if batch_vectors is None:
                    batch_vectors = []
                    for item in batch:
                        vec = await self._embed_single_via_api_embeddings(client, item)
                        batch_vectors.append(vec)

                # Validate and normalize vectors
                for vec in batch_vectors:
                    norm_vec = validate_and_normalize_vector(vec, expected_dim=None)
                    # Check dimension consistency across batch
                    if all_vectors and len(norm_vec) != len(all_vectors[0]):
                        raise VectorValidationError(
                            f"Dimension mismatch in batch: expected {len(all_vectors[0])}, got {len(norm_vec)}."
                        )
                    all_vectors.append(norm_vec)

        if all_vectors:
            self.dimension = len(all_vectors[0])

        return all_vectors

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate a single normalized embedding vector for the given text."""
        results = await self.get_embeddings_batch([text], is_query=is_query, batch_size=1)
        if not results:
            raise VectorValidationError("Empty embedding returned.")
        return results[0]

    def get_info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="ollama",
            model=self.model,
            dimension=self.dimension,
            is_neural=True,
            status="active" if self._available_cache else "unreachable",
            details={
                "base_url": self.base_url,
                "supports_api_embed": self._supports_api_embed,
                "batch_size": self.batch_size
            }
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
        else:
            vec[0] = 1.0  # Unit vector fallback for empty/stopwords text
        return validate_and_normalize_vector(vec, expected_dim=self.dimension)

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False, batch_size: int = 16
    ) -> List[List[float]]:
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
    """Manages active embedding provider selection, model switching, and honest health checks."""

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
        self._dimension: Optional[int] = None
        self._model_name: Optional[str] = None

    @property
    def model_name(self) -> str:
        """Name of the currently configured or active embedding model."""
        if self._model_name is not None:
            return self._model_name
        if self._active_provider:
            return self._active_provider.get_info().model
        pref = getattr(self.settings, "EMBEDDING_PROVIDER", "auto").lower()
        if pref == "deterministic_hash":
            return f"hash-crc32-{self.hash_provider.dimension}"
        return getattr(self.settings, "EMBEDDING_MODEL", "nomic-embed-text")

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        self._model_name = value

    @property
    def dimension(self) -> int:
        """Dimension of the currently configured or active embedding model."""
        if self._dimension is not None:
            return self._dimension
        if self._active_provider:
            return self._active_provider.get_info().dimension
        pref = getattr(self.settings, "EMBEDDING_PROVIDER", "auto").lower()
        if pref == "deterministic_hash":
            return self.hash_provider.dimension
        model = getattr(self.settings, "EMBEDDING_MODEL", "").lower()
        if "nomic" in model:
            return 768
        elif "mxbai" in model or "bge-large" in model:
            return 1024
        elif "bge-small" in model or "minilm" in model:
            return 384
        return getattr(self.ollama_provider, "dimension", 768)

    @dimension.setter
    def dimension(self, value: Optional[int]) -> None:
        self._dimension = value

    async def check_health(self) -> Dict[str, Any]:
        """Ping active provider to honestly report if model is ready, missing, or offline."""
        pref = getattr(self.settings, "EMBEDDING_PROVIDER", "auto").lower()

        if pref == "deterministic_hash":
            return {
                "status": "ready",
                "provider": "deterministic_hash",
                "model": f"hash-crc32-{self.hash_provider.dimension}",
                "dimension": self.hash_provider.dimension,
                "is_neural": False,
                "error": None,
                "details": {"algorithm": "crc32"}
            }

        base_url = (self.ollama_provider.base_url).rstrip("/")
        target_model = self.ollama_provider.model

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{base_url}/api/tags")
                if res.status_code != 200:
                    return {
                        "status": "offline",
                        "provider": "ollama",
                        "model": target_model,
                        "dimension": self.dimension,
                        "is_neural": True,
                        "error": f"Ollama daemon returned HTTP {res.status_code}",
                        "details": {"base_url": base_url}
                    }

                data = res.json()
                models_info = data.get("models", [])
                model_names = [m.get("name", "") for m in models_info]
                model_bases = [m.split(":")[0] for m in model_names]

                target_base = target_model.split(":")[0]
                is_installed = (
                    target_model in model_names
                    or target_base in model_bases
                    or any(target_model in m for m in model_names)
                )

                if is_installed:
                    return {
                        "status": "ready",
                        "provider": "ollama",
                        "model": target_model,
                        "dimension": self.dimension,
                        "is_neural": True,
                        "error": None,
                        "details": {"base_url": base_url}
                    }
                else:
                    return {
                        "status": "missing",
                        "provider": "ollama",
                        "model": target_model,
                        "dimension": self.dimension,
                        "is_neural": True,
                        "error": f"Model '{target_model}' not found in Ollama. Run 'ollama pull {target_model}'.",
                        "details": {"base_url": base_url, "installed_models": model_names}
                    }
        except Exception as e:
            return {
                "status": "offline",
                "provider": "ollama",
                "model": target_model,
                "dimension": self.dimension,
                "is_neural": True,
                "error": f"Failed to connect to Ollama daemon at {base_url}: {e}",
                "details": {"base_url": base_url}
            }

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

        # Check fallback permissions: permitted in testing or when explicitly enabled
        env = getattr(self.settings, "APP_ENV", "development").lower()
        allow_fallback_cfg = getattr(self.settings, "ALLOW_FALLBACK_EMBEDDINGS", False)

        if env == "testing":
            allow_fallback = True
        elif allow_fallback_cfg is True:
            allow_fallback = True
        else:
            allow_fallback = False

        if not allow_fallback:
            # Prevent silent fallback pollution in development and production
            health = await self.check_health()
            if health.get("status") == "missing":
                raise ModelNotFoundError(
                    f"Configured embedding model '{self.ollama_provider.model}' is not pulled in Ollama. "
                    f"Run 'ollama pull {self.ollama_provider.model}'."
                )
            raise ProviderUnavailableError(
                f"Embedding provider 'ollama' is unreachable at {self.ollama_provider.base_url}. "
                f"Silent fallback to deterministic hash embeddings is disabled in '{env}' mode. "
                "Ensure Ollama is running or set ALLOW_FALLBACK_EMBEDDINGS=true."
            )

        # Explicit fallback with clear warning in testing mode
        logger.warning(
            f"Neural embedding model '{self.settings.EMBEDDING_MODEL}' is not available in Ollama. "
            "Falling back to DeterministicHashEmbeddingProvider (non-neural token vectors)."
        )
        self._active_provider = self.hash_provider
        return self._active_provider

    async def get_active_info(self) -> EmbeddingInfo:
        provider = await self.get_provider()
        return provider.get_info()

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        provider = await self.get_provider()
        return await provider.get_embedding(text, is_query=is_query)

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False, batch_size: int = 16
    ) -> List[List[float]]:
        provider = await self.get_provider()
        return await provider.get_embeddings_batch(texts, is_query=is_query, batch_size=batch_size)

    async def embed_text(self, text: str, input_type: str = "document") -> List[float]:
        provider = await self.get_provider()
        return await provider.embed_text(text, input_type=input_type)

    async def embed_query(self, query: str) -> List[float]:
        provider = await self.get_provider()
        return await provider.embed_query(query)

    async def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        provider = await self.get_provider()
        return await provider.embed_documents(texts, batch_size=batch_size)


embedding_manager = EmbeddingManager()
