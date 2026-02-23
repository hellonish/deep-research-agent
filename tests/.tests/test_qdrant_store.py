"""
Integration tests for QdrantStore with in-memory Qdrant.

No server required — uses AsyncQdrantClient(":memory:").
"""

import asyncio
import pytest
import pytest_asyncio

from vector_store import QdrantStore
from models import Document, SearchQuery


# ── Fixtures ───────────────────────────────────────────────────────────

COLLECTION = "test_research"

SAMPLE_DOCS = [
    Document(
        id="doc-1",
        content=(
            "Transformers have revolutionized natural language processing. "
            "The self-attention mechanism allows models to capture long-range "
            "dependencies in text effectively."
        ),
        metadata={"source": "arxiv", "title": "Attention Is All You Need", "year": 2017},
    ),
    Document(
        id="doc-2",
        content=(
            "Retrieval-Augmented Generation combines a retriever with a generator "
            "to produce grounded, factual responses. RAG reduces hallucination in "
            "large language model outputs."
        ),
        metadata={"source": "arxiv", "title": "RAG for Knowledge-Intensive Tasks", "year": 2020},
    ),
    Document(
        id="doc-3",
        content=(
            "Quantum computing leverages quantum mechanical phenomena such as "
            "superposition and entanglement to perform computation. Qubits can "
            "represent both 0 and 1 simultaneously."
        ),
        metadata={"source": "nature", "title": "Intro to Quantum Computing", "year": 2023},
    ),
    Document(
        id="doc-4",
        content=(
            "BM25 is a probabilistic information retrieval model that ranks documents "
            "based on term frequency and inverse document frequency. It remains a "
            "strong baseline for keyword search."
        ),
        metadata={"source": "acm", "title": "BM25 and Beyond", "year": 2009},
    ),
    Document(
        id="doc-5",
        content=(
            "Vector databases store high-dimensional embeddings and enable fast "
            "approximate nearest neighbor search. They are essential infrastructure "
            "for modern AI applications."
        ),
        metadata={"source": "blog", "title": "Vector DB Overview", "year": 2024},
    ),
]


@pytest_asyncio.fixture
async def store():
    """Create an in-memory QdrantStore for testing."""
    s = QdrantStore(in_memory=True)
    yield s


@pytest_asyncio.fixture
async def populated_store(store: QdrantStore):
    """Create a store with a collection and sample documents."""
    await store.create_collection(COLLECTION, recreate=True)
    await store.upsert(COLLECTION, SAMPLE_DOCS)
    yield store
    # Cleanup
    if await store.collection_exists(COLLECTION):
        await store.delete_collection(COLLECTION)


# ── Tests ──────────────────────────────────────────────────────────────


class TestCollectionManagement:

    @pytest.mark.asyncio
    async def test_create_and_exists(self, store: QdrantStore):
        await store.create_collection(COLLECTION, recreate=True)
        assert await store.collection_exists(COLLECTION) is True

    @pytest.mark.asyncio
    async def test_delete_collection(self, store: QdrantStore):
        await store.create_collection(COLLECTION)
        await store.delete_collection(COLLECTION)
        assert await store.collection_exists(COLLECTION) is False

    @pytest.mark.asyncio
    async def test_collection_info(self, populated_store: QdrantStore):
        info = await populated_store.collection_info(COLLECTION)
        assert info.name == COLLECTION
        assert info.points_count == len(SAMPLE_DOCS)

    @pytest.mark.asyncio
    async def test_recreate_collection(self, populated_store: QdrantStore):
        # Recreate should reset the collection
        await populated_store.create_collection(COLLECTION, recreate=True)
        count = await populated_store.count(COLLECTION)
        assert count == 0


class TestUpsertAndRead:

    @pytest.mark.asyncio
    async def test_count(self, populated_store: QdrantStore):
        count = await populated_store.count(COLLECTION)
        assert count == len(SAMPLE_DOCS)

    @pytest.mark.asyncio
    async def test_get_by_id(self, populated_store: QdrantStore):
        docs = await populated_store.get(COLLECTION, ["doc-1"])
        assert len(docs) == 1
        assert "Transformers" in docs[0].content

    @pytest.mark.asyncio
    async def test_get_multiple(self, populated_store: QdrantStore):
        docs = await populated_store.get(COLLECTION, ["doc-1", "doc-3"])
        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_scroll(self, populated_store: QdrantStore):
        docs, next_offset = await populated_store.scroll(COLLECTION, limit=3)
        assert len(docs) == 3
        # There should be more docs to fetch
        assert next_offset is not None

        docs2, next_offset2 = await populated_store.scroll(
            COLLECTION, limit=3, offset=next_offset
        )
        assert len(docs2) == 2  # remaining docs


class TestHybridSearch:

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, populated_store: QdrantStore):
        query = SearchQuery(text="attention mechanism in neural networks", top_k=3)
        results = await populated_store.search(COLLECTION, query)
        assert len(results) > 0
        assert all(r.score > 0 for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_ranks_relevant_first(self, populated_store: QdrantStore):
        query = SearchQuery(text="retrieval augmented generation RAG", top_k=5)
        results = await populated_store.search(COLLECTION, query)
        # The RAG document should be in the top results
        top_ids_content = [r.content for r in results[:2]]
        assert any("RAG" in c or "Retrieval" in c for c in top_ids_content)

    @pytest.mark.asyncio
    async def test_dense_search(self, populated_store: QdrantStore):
        query = SearchQuery(text="semantic similarity in text", top_k=3)
        results = await populated_store.search_dense(COLLECTION, query)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_sparse_search(self, populated_store: QdrantStore):
        query = SearchQuery(text="BM25 term frequency", top_k=3)
        results = await populated_store.search_sparse(COLLECTION, query)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, populated_store: QdrantStore):
        query = SearchQuery(text="machine learning", top_k=2)
        results = await populated_store.search(COLLECTION, query)
        assert len(results) <= 2


class TestSearchByDocument:

    @pytest.mark.asyncio
    async def test_find_similar_documents(self, populated_store: QdrantStore):
        # Use the RAG document and find similar ones
        source_doc = SAMPLE_DOCS[1]  # RAG doc
        results = await populated_store.search_by_document(
            COLLECTION, source_doc, top_k=3
        )
        assert len(results) > 0
        # Should not include the source document itself
        assert all(r.metadata.get("original_id") != source_doc.id for r in results)


class TestDelete:

    @pytest.mark.asyncio
    async def test_delete_reduces_count(self, populated_store: QdrantStore):
        initial_count = await populated_store.count(COLLECTION)
        await populated_store.delete(COLLECTION, ["doc-1"])
        new_count = await populated_store.count(COLLECTION)
        assert new_count == initial_count - 1

    @pytest.mark.asyncio
    async def test_deleted_doc_not_retrievable(self, populated_store: QdrantStore):
        await populated_store.delete(COLLECTION, ["doc-2"])
        docs = await populated_store.get(COLLECTION, ["doc-2"])
        assert len(docs) == 0
