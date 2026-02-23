"""
Qdrant-backed vector store with Dense + Sparse embeddings and RRF hybrid search.

Uses FastEmbed for local embedding (no external API keys required).
All operations are fully async.
"""

import logging
import uuid
from typing import Any, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models

from .base import BaseVectorStore
from models import CollectionInfo, Document, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """
    Production-ready Qdrant vector store with hybrid search.

    Features:
    - Dense embeddings via FastEmbed (sentence-transformers)
    - Sparse embeddings via FastEmbed (BM25)
    - Hybrid search fusing both via Reciprocal Rank Fusion (RRF)
    - Named vectors: "dense" and "sparse" per collection
    - Fully async via AsyncQdrantClient
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        prefetch_limit: int = 20,
        in_memory: bool = False,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            url: Qdrant server URL.
            api_key: Optional API key for Qdrant Cloud.
            dense_model: FastEmbed model name for dense embeddings.
            sparse_model: FastEmbed model name for sparse embeddings.
            prefetch_limit: Number of candidates each sub-query retrieves before RRF fusion.
            in_memory: If True, use an in-memory Qdrant instance (for testing).
        """
        if in_memory:
            self.client = AsyncQdrantClient(location=":memory:")
            self._in_memory = True
            self._url = None
        else:
            self.client = AsyncQdrantClient(url=url, api_key=api_key)
            self._in_memory = False
            self._url = url

        self.prefetch_limit = prefetch_limit

        # Initialize embedding models (runs locally via FastEmbed)
        logger.info("Loading dense embedding model: %s", dense_model)
        self._dense_model = TextEmbedding(model_name=dense_model)
        self._dense_dim = self._get_dense_dim()

        logger.info("Loading sparse embedding model: %s", sparse_model)
        self._sparse_model = SparseTextEmbedding(model_name=sparse_model)

    def _get_dense_dim(self) -> int:
        """Infer dimensionality by encoding a dummy string."""
        dummy = list(self._dense_model.embed(["hello"]))[0]
        return len(dummy)

    # ── Connection / mode ──────────────────────────────────────────────

    def is_remote(self) -> bool:
        """True if this store is connected to a remote Qdrant (URL/cloud), False if in-memory."""
        return not self._in_memory

    @property
    def location(self) -> str:
        """Human-readable location: ':memory:' or the configured URL (e.g. cloud)."""
        return ":memory:" if self._in_memory else (self._url or "remote")

    async def check_connection(self) -> bool:
        """
        Verify connection to Qdrant. For remote clients, calls the server (e.g. get_collections).
        For in-memory, always returns True.
        """
        if self._in_memory:
            return True
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant connection check failed: %s", e)
            return False

    # ── Embedding Helpers ──────────────────────────────────────────────

    def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a list of texts."""
        return [vec.tolist() for vec in self._dense_model.embed(texts)]

    def _embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        """Generate sparse (BM25) embeddings for a list of texts."""
        results = []
        for sparse_vec in self._sparse_model.embed(texts):
            results.append(
                models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                )
            )
        return results

    # ── Filter Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_filters(filters: dict[str, Any] | None) -> models.Filter | None:
        """Convert a simple key-value dict into a Qdrant Filter."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Match any value in the list
                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions)

    # ── Point Conversion Helpers ───────────────────────────────────────

    @staticmethod
    def _point_to_document(point) -> Document:
        """Convert a Qdrant point (ScoredPoint or Record) into a Document."""
        payload = point.payload or {}
        return Document(
            id=str(point.id),
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _point_to_search_result(point) -> SearchResult:
        """Convert a Qdrant ScoredPoint into a SearchResult."""
        payload = point.payload or {}
        return SearchResult(
            id=str(point.id),
            content=payload.get("content", ""),
            score=point.score if hasattr(point, "score") and point.score is not None else 0.0,
            metadata=payload.get("metadata", {}),
        )

    # ── Collection Management ──────────────────────────────────────────

    async def create_collection(
        self,
        name: str,
        dense_dim: int | None = None,
        recreate: bool = False,
    ) -> None:
        dim = dense_dim or self._dense_dim
        exists = await self.collection_exists(name)

        if exists and recreate:
            logger.info("Recreating collection '%s'", name)
            await self.delete_collection(name)
        elif exists:
            logger.info("Collection '%s' already exists, skipping creation", name)
            return

        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )
        logger.info("Created collection '%s' (dense_dim=%d)", name, dim)

    async def delete_collection(self, name: str) -> None:
        await self.client.delete_collection(collection_name=name)
        logger.info("Deleted collection '%s'", name)

    async def collection_exists(self, name: str) -> bool:
        return await self.client.collection_exists(collection_name=name)

    async def collection_info(self, name: str) -> CollectionInfo:
        info = await self.client.get_collection(collection_name=name)
        return CollectionInfo(
            name=name,
            points_count=info.points_count or 0,
            vectors_count=getattr(info, "vectors_count", 0) or 0,
            status=str(info.status),
        )

    # ── Write Operations ───────────────────────────────────────────────

    async def upsert(self, collection: str, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [doc.content for doc in documents]

        # Generate embeddings (runs locally)
        dense_vectors = self._embed_dense(texts)
        sparse_vectors = self._embed_sparse(texts)

        # Build Qdrant points
        points = []
        for doc, dense_vec, sparse_vec in zip(documents, dense_vectors, sparse_vectors):
            # Use the document ID directly, or generate a UUID if it's not a valid UUID
            try:
                point_id = str(uuid.UUID(doc.id))
            except ValueError:
                # Use a deterministic UUID from the string ID for consistency
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_VECTOR_NAME: dense_vec,
                        self.SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "original_id": doc.id,
                    },
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=collection,
                points=batch,
            )

        logger.info("Upserted %d documents into '%s'", len(documents), collection)

    async def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return

        # Convert string IDs to the UUID format used during upsert
        point_ids = []
        for doc_id in ids:
            try:
                point_ids.append(str(uuid.UUID(doc_id)))
            except ValueError:
                point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)))

        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.info("Deleted %d documents from '%s'", len(ids), collection)

    # ── Search Operations ──────────────────────────────────────────────

    async def search(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """Hybrid search: Dense + Sparse fused by Reciprocal Rank Fusion."""
        dense_vec = self._embed_dense([query.text])[0]
        sparse_vec = self._embed_sparse([query.text])[0]
        qdrant_filter = self._build_filters(query.filters)

        results = await self.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=sparse_vec,
                    using=self.SPARSE_VECTOR_NAME,
                    limit=self.prefetch_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using=self.DENSE_VECTOR_NAME,
                    limit=self.prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=query.top_k,
            with_payload=True,
            score_threshold=query.score_threshold,
        )

        return [self._point_to_search_result(pt) for pt in results.points]

    async def search_dense(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """Dense-only (semantic) search."""
        dense_vec = self._embed_dense([query.text])[0]
        qdrant_filter = self._build_filters(query.filters)

        results = await self.client.query_points(
            collection_name=collection,
            query=dense_vec,
            using=self.DENSE_VECTOR_NAME,
            limit=query.top_k,
            with_payload=True,
            query_filter=qdrant_filter,
            score_threshold=query.score_threshold,
        )

        return [self._point_to_search_result(pt) for pt in results.points]

    async def search_sparse(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """Sparse-only (keyword / BM25) search."""
        sparse_vec = self._embed_sparse([query.text])[0]
        qdrant_filter = self._build_filters(query.filters)

        results = await self.client.query_points(
            collection_name=collection,
            query=sparse_vec,
            using=self.SPARSE_VECTOR_NAME,
            limit=query.top_k,
            with_payload=True,
            query_filter=qdrant_filter,
            score_threshold=query.score_threshold,
        )

        return [self._point_to_search_result(pt) for pt in results.points]

    async def search_by_document(
        self,
        collection: str,
        document: Document,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """
        Find documents similar to a given document.
        Uses hybrid search internally.
        """
        query = SearchQuery(text=document.content, top_k=top_k + (1 if exclude_self else 0))
        results = await self.search(collection, query)

        if exclude_self:
            # Filter out the source document by comparing original_id
            results = [r for r in results if r.id != document.id and r.metadata.get("original_id") != document.id]

        return results[:top_k]

    # ── Read Operations ────────────────────────────────────────────────

    async def get(self, collection: str, ids: list[str]) -> list[Document]:
        if not ids:
            return []

        # Convert to UUID format
        point_ids = []
        for doc_id in ids:
            try:
                point_ids.append(str(uuid.UUID(doc_id)))
            except ValueError:
                point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)))

        points = await self.client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_payload=True,
        )

        return [self._point_to_document(pt) for pt in points]

    async def count(self, collection: str) -> int:
        info = await self.client.get_collection(collection_name=collection)
        return info.points_count or 0

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Document], str | None]:
        qdrant_filter = self._build_filters(filters)

        records, next_offset = await self.client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            scroll_filter=qdrant_filter,
            with_payload=True,
        )

        documents = [self._point_to_document(r) for r in records]
        next_page = str(next_offset) if next_offset is not None else None

        return documents, next_page
