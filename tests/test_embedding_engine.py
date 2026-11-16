"""Comprehensive unit & mocked tests for Ollama embedding integration, batching, Nomic prefixes, and resilience."""

import math
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.knowledge.embeddings import (
    OllamaEmbeddingProvider,
    DeterministicHashEmbeddingProvider,
    EmbeddingManager,
    ModelNotFoundError,
    ProviderUnavailableError,
    ModelMigrationError,
    VectorValidationError,
    EmbeddingError,
    format_text_for_model,
    validate_and_normalize_vector,
)
from app.config import Settings
from app.knowledge.vector_store import VectorStore
from app.knowledge.chunker import DocumentChunk


# ---------------------------------------------------------------------------
# 1. Nomic Prefixing Tests
# ---------------------------------------------------------------------------

def test_nomic_prefixing_asymmetric():
    """Verify asymmetric Nomic prefixes are applied for nomic models and avoid double-prefixing."""
    nomic_model = "nomic-embed-text"
    other_model = "bge-large-en"

    # Nomic documents
    raw_doc = "Byzantine fault tolerance ensures state consensus."
    formatted_doc = format_text_for_model(raw_doc, nomic_model, is_query=False)
    assert formatted_doc == "search_document: Byzantine fault tolerance ensures state consensus."

    # Avoid double prefixing for document
    already_prefixed_doc = "search_document: Already prefixed text."
    assert format_text_for_model(already_prefixed_doc, nomic_model, is_query=False) == already_prefixed_doc

    # Nomic queries
    raw_query = "What is Paxos consensus?"
    formatted_query = format_text_for_model(raw_query, nomic_model, is_query=True)
    assert formatted_query == "search_query: What is Paxos consensus?"

    # Avoid double prefixing for query
    already_prefixed_query = "search_query: Already prefixed query."
    assert format_text_for_model(already_prefixed_query, nomic_model, is_query=True) == already_prefixed_query

    # Non-nomic model preserves raw text untouched
    assert format_text_for_model(raw_doc, other_model, is_query=False) == raw_doc
    assert format_text_for_model(raw_query, other_model, is_query=True) == raw_query


# ---------------------------------------------------------------------------
# 2. Vector Validation & Normalization Tests
# ---------------------------------------------------------------------------

def test_validate_and_normalize_vector():
    """Verify vector validation rejects malformed vectors and L2-normalizes valid ones."""
    # Valid vector
    raw = [3.0, 4.0]
    norm_vec = validate_and_normalize_vector(raw, expected_dim=2)
    assert math.isclose(norm_vec[0], 0.6)
    assert math.isclose(norm_vec[1], 0.8)
    assert math.isclose(math.sqrt(sum(x * x for x in norm_vec)), 1.0)

    # Rejection: empty vector
    with pytest.raises(VectorValidationError, match="non-empty"):
        validate_and_normalize_vector([])

    # Rejection: NaN values
    with pytest.raises(VectorValidationError, match="NaN"):
        validate_and_normalize_vector([1.0, float("nan")])

    # Rejection: Infinite values
    with pytest.raises(VectorValidationError, match="infinite"):
        validate_and_normalize_vector([1.0, float("inf")])

    # Rejection: Zero L2 norm
    with pytest.raises(VectorValidationError, match="zero L2 norm"):
        validate_and_normalize_vector([0.0, 0.0, 0.0])

    # Rejection: Dimension mismatch
    with pytest.raises(VectorValidationError, match="dimension mismatch"):
        validate_and_normalize_vector([1.0, 2.0], expected_dim=3)


# ---------------------------------------------------------------------------
# 3. Batch Embedding & Endpoint Support Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_embedding_via_api_embed():
    """Verify OllamaEmbeddingProvider embeds in batches using /api/embed."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", batch_size=2)
    texts = [
        "First document text",
        "Second document text",
        "Third document text",
        "Fourth document text",
        "Fifth document text",
    ]

    captured_requests = []

    def mock_post(url, *args, **kwargs):
        json_data = kwargs.get("json") or (args[0] if args else {})
        captured_requests.append((str(url), json_data))
        batch_input = json_data.get("input", []) if json_data else []
        dummy_embeddings = [[1.0, 0.0, 0.0, 0.0] for _ in batch_input]
        return httpx.Response(200, json={"model": "nomic-embed-text", "embeddings": dummy_embeddings})

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        vectors = await provider.embed_documents(texts, batch_size=2)

    assert len(vectors) == 5
    assert all(len(v) == 4 for v in vectors)
    # 5 texts with batch_size=2 => 3 batch requests (sizes 2, 2, 1)
    assert len(captured_requests) == 3
    assert len(captured_requests[0][1]["input"]) == 2
    assert len(captured_requests[1][1]["input"]) == 2
    assert len(captured_requests[2][1]["input"]) == 1
    # Check that Nomic prefix was added to all batch inputs
    assert captured_requests[0][1]["input"][0].startswith("search_document: ")


@pytest.mark.asyncio
async def test_batch_embedding_fallback_to_api_embeddings():
    """Verify OllamaEmbeddingProvider falls back to /api/embeddings when /api/embed returns 404."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", batch_size=2)
    texts = ["Document one", "Document two"]

    captured_urls = []

    def mock_post(url, *args, **kwargs):
        url_str = str(url)
        captured_urls.append(url_str)
        if url_str.endswith("/api/embed"):
            # 404 route not found (older Ollama server)
            return httpx.Response(404, text="404 page not found")
        elif url_str.endswith("/api/embeddings"):
            return httpx.Response(200, json={"embedding": [0.0, 2.0, 0.0]})
        return httpx.Response(500)

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        vectors = await provider.embed_documents(texts)

    assert len(vectors) == 2
    assert all(math.isclose(v[1], 1.0) for v in vectors)  # normalized [0, 1, 0]
    # Check fallback occurred
    assert any(u.endswith("/api/embed") for u in captured_urls)
    assert any(u.endswith("/api/embeddings") for u in captured_urls)
    assert provider._supports_api_embed is False


@pytest.mark.asyncio
async def test_batch_dimension_mismatch_raises_error():
    """Verify that dimension mismatches within a batch raise VectorValidationError."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", batch_size=2)
    texts = ["Doc 1", "Doc 2"]

    def mock_post(url, *args, **kwargs):
        # Return inconsistent dimensions
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]})

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(VectorValidationError, match="Dimension mismatch"):
            await provider.embed_documents(texts)


# ---------------------------------------------------------------------------
# 4. Retry & Error Resilience Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transient_retry_and_recovery():
    """Verify transient 503 or ReadTimeout retries and succeeds on subsequent attempt."""
    provider = OllamaEmbeddingProvider(
        model="nomic-embed-text", max_retries=3, initial_delay=0.001
    )

    call_count = 0

    def mock_post(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, text="Service Unavailable")
        elif call_count == 2:
            raise httpx.ReadTimeout("Read timed out")
        else:
            return httpx.Response(200, json={"embeddings": [[1.0, 1.0]]})

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        vec = await provider.embed_query("Query text")

    assert call_count == 3
    assert len(vec) == 2
    assert math.isclose(vec[0], 1.0 / math.sqrt(2.0))


@pytest.mark.asyncio
async def test_model_not_found_fails_immediately_without_retries():
    """Verify permanent 404 model-not-found raises ModelNotFoundError immediately."""
    provider = OllamaEmbeddingProvider(
        model="missing-model", max_retries=3, initial_delay=0.001
    )

    call_count = 0

    def mock_post(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, json={"error": "model 'missing-model' not found, try pulling it first"})

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            await provider.embed_query("Query")

    # Should have failed on attempt 1 without redundant retries
    assert call_count == 1


@pytest.mark.asyncio
async def test_embedding_manager_convenience_methods():
    """Verify EmbeddingManager delegates embed_text, embed_query, and embed_documents."""
    mgr = EmbeddingManager()
    # Force use of hash provider
    with patch.object(mgr.ollama_provider, "is_available", AsyncMock(return_value=False)):
        q_vec = await mgr.embed_query("Data structures and algorithms")
        assert len(q_vec) == 256
        assert math.isclose(math.sqrt(sum(x * x for x in q_vec)), 1.0)

        docs_vecs = await mgr.embed_documents(["First doc", "Second doc"])
        assert len(docs_vecs) == 2
        assert len(docs_vecs[0]) == 256


# ---------------------------------------------------------------------------
# 5. Guardrails Against Silent Fallback Pollution (Task P4-02)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offline_provider_in_dev_or_prod_raises_error():
    """Verify that in development/production mode, an unavailable Ollama raises ProviderUnavailableError."""
    # 1. Development mode: Ollama is offline
    dev_settings = Settings(APP_ENV="development", ALLOW_FALLBACK_EMBEDDINGS=False)
    mgr_dev = EmbeddingManager(settings_obj=dev_settings)
    with patch.object(mgr_dev.ollama_provider, "is_available", AsyncMock(return_value=False)):
        with patch.object(mgr_dev, "check_health", AsyncMock(return_value={"status": "offline"})):
            with pytest.raises(ProviderUnavailableError, match="unreachable"):
                await mgr_dev.get_provider()

    # 2. Production mode: Ollama is offline
    prod_settings = Settings(APP_ENV="production", ALLOW_FALLBACK_EMBEDDINGS=False)
    mgr_prod = EmbeddingManager(settings_obj=prod_settings)
    with patch.object(mgr_prod.ollama_provider, "is_available", AsyncMock(return_value=False)):
        with patch.object(mgr_prod, "check_health", AsyncMock(return_value={"status": "offline"})):
            with pytest.raises(ProviderUnavailableError, match="unreachable"):
                await mgr_prod.get_provider()

    # 3. Development mode: Ollama is running but model is missing
    with patch.object(mgr_dev.ollama_provider, "is_available", AsyncMock(return_value=False)):
        with patch.object(mgr_dev, "check_health", AsyncMock(return_value={"status": "missing"})):
            with pytest.raises(ModelNotFoundError, match="ollama pull"):
                await mgr_dev.get_provider()


@pytest.mark.asyncio
async def test_fallback_permitted_in_testing_or_explicit_override():
    """Verify fallback to hash embeddings is allowed in testing mode or with ALLOW_FALLBACK_EMBEDDINGS=True."""
    # 1. Testing environment: fallback is permitted
    test_settings = Settings(APP_ENV="testing")
    mgr_test = EmbeddingManager(settings_obj=test_settings)
    with patch.object(mgr_test.ollama_provider, "is_available", AsyncMock(return_value=False)):
        prov = await mgr_test.get_provider()
        assert isinstance(prov, DeterministicHashEmbeddingProvider)

    # 2. Development environment with explicit ALLOW_FALLBACK_EMBEDDINGS=True
    dev_override_settings = Settings(APP_ENV="development", ALLOW_FALLBACK_EMBEDDINGS=True)
    mgr_dev = EmbeddingManager(settings_obj=dev_override_settings)
    with patch.object(mgr_dev.ollama_provider, "is_available", AsyncMock(return_value=False)):
        prov_dev = await mgr_dev.get_provider()
        assert isinstance(prov_dev, DeterministicHashEmbeddingProvider)


@pytest.mark.asyncio
async def test_dimension_lock_mismatch_raises_model_migration_error(tmp_path):
    """Verify that inserting 1024d vectors into an existing 768d index raises ModelMigrationError."""
    # Initialize VectorStore with a 768d embedding manager
    mgr_768 = EmbeddingManager()
    prov_768 = DeterministicHashEmbeddingProvider(dimension=768)
    store = VectorStore(persist_dir=tmp_path, embedding_mgr=mgr_768)

    # Add initial chunk with 768d
    mgr_768.dimension = 768
    mgr_768.model_name = "nomic-embed-text"
    chunk1 = DocumentChunk(
        chunk_id="chunk-768-1",
        document_id="doc-1",
        filename="lecture1.txt",
        subject="OS",
        page_number=1,
        chunk_index=0,
        text="Introduction to Operating Systems: Concurrency and Scheduling.",
        metadata={"subject": "OS"}
    )
    with patch.object(mgr_768, "get_provider", AsyncMock(return_value=prov_768)):
        await store.add_chunks([chunk1])

    assert store.get_index_status()["embedding_dimension"] == 768
    assert len(store._fallback_chunks) == 1

    # Now attempt to insert a second chunk with a 1024d model without reindexing
    mgr_1024 = EmbeddingManager()
    mgr_1024.dimension = 1024
    mgr_1024.model_name = "mxbai-embed-large"
    prov_1024 = DeterministicHashEmbeddingProvider(dimension=1024)
    store.embedding_manager = mgr_1024

    chunk2 = DocumentChunk(
        chunk_id="chunk-1024-2",
        document_id="doc-2",
        filename="lecture2.txt",
        subject="OS",
        page_number=1,
        chunk_index=0,
        text="Advanced Memory Management: Paging and Page Replacement Algorithms.",
        metadata={"subject": "OS"}
    )
    with patch.object(mgr_1024, "get_provider", AsyncMock(return_value=prov_1024)):
        with pytest.raises(ModelMigrationError, match="dimensionality lock error"):
            await store.add_chunks([chunk2])


@pytest.mark.asyncio
async def test_honest_embedding_health_check_states():
    """Verify check_health() accurately detects ready, missing, and offline Ollama states."""
    mgr = EmbeddingManager()

    # 1. Ready state: Ollama returns 200 with model in tags
    def mock_ready_get(url, *args, **kwargs):
        return httpx.Response(200, json={
            "models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:1.5b"}]
        })

    with patch("httpx.AsyncClient.get", side_effect=mock_ready_get):
        h_ready = await mgr.check_health()
        assert h_ready["status"] == "ready"
        assert h_ready["provider"] == "ollama"
        assert h_ready["is_neural"] is True
        assert h_ready["error"] is None

    # 2. Missing state: Ollama returns 200 but model is absent
    def mock_missing_get(url, *args, **kwargs):
        return httpx.Response(200, json={
            "models": [{"name": "all-minilm:latest"}]
        })

    with patch("httpx.AsyncClient.get", side_effect=mock_missing_get):
        h_missing = await mgr.check_health()
        assert h_missing["status"] == "missing"
        assert "not found" in h_missing["error"]

    # 3. Offline state: Ollama daemon connection fails
    def mock_offline_get(url, *args, **kwargs):
        raise httpx.ConnectError("Connection refused to 127.0.0.1:11434")

    with patch("httpx.AsyncClient.get", side_effect=mock_offline_get):
        h_offline = await mgr.check_health()
        assert h_offline["status"] == "offline"
        assert "Failed to connect" in h_offline["error"]

    # 4. Hash provider: always ready with is_neural=False
    hash_settings = Settings(APP_ENV="testing", EMBEDDING_PROVIDER="deterministic_hash")
    mgr_hash = EmbeddingManager(settings_obj=hash_settings)
    h_hash = await mgr_hash.check_health()
    assert h_hash["status"] == "ready"
    assert h_hash["provider"] == "deterministic_hash"
    assert h_hash["is_neural"] is False

