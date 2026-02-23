"""
Abstract base class defining the contract for all vector store implementations.

Every method is async to support non-blocking I/O with vector databases.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from models import CollectionInfo, Document, SearchQuery, SearchResult


class BaseVectorStore(ABC):
    """
    Abstract vector store interface for a deep research agent.

    Implementations must support:
    - Dense + Sparse vector storage
    - Hybrid search via Reciprocal Rank Fusion (RRF)
    - Standard CRUD operations
    """

    # ── Collection Management ──────────────────────────────────────────

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dense_dim: int | None = None,
        recreate: bool = False,
    ) -> None:
        """
        Create a collection with dense and sparse vector configurations.

        Args:
            name: Collection name.
            dense_dim: Dimensionality of the dense vectors (inferred from model if None).
            recreate: If True, drops and recreates the collection if it already exists.
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        ...

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Return True if the collection exists."""
        ...

    @abstractmethod
    async def collection_info(self, name: str) -> CollectionInfo:
        """Return stats about the collection (point count, vector count, status)."""
        ...

    # ── Write Operations ───────────────────────────────────────────────

    @abstractmethod
    async def upsert(self, collection: str, documents: list[Document]) -> None:
        """
        Insert or update documents in the collection.

        Each document is embedded (dense + sparse) and stored with its metadata.
        Batches are handled internally.

        Args:
            collection: Target collection name.
            documents: List of Document objects to upsert.
        """
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            collection: Target collection name.
            ids: List of document IDs to remove.
        """
        ...

    # ── Search Operations ──────────────────────────────────────────────

    @abstractmethod
    async def search(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """
        Hybrid search: combines dense (semantic) and sparse (keyword) retrieval
        using Reciprocal Rank Fusion (RRF).

        This is the primary search method — use it by default.

        Args:
            collection: Target collection name.
            query: SearchQuery with text, top_k, optional filters and score_threshold.

        Returns:
            Ranked list of SearchResult objects.
        """
        ...

    @abstractmethod
    async def search_dense(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """
        Dense-only (semantic) search using cosine similarity.

        Best for: "find conceptually similar research", broad exploration.
        """
        ...

    @abstractmethod
    async def search_sparse(
        self, collection: str, query: SearchQuery
    ) -> list[SearchResult]:
        """
        Sparse-only (keyword / BM25) search.

        Best for: precise term matching, acronyms, named entities.
        """
        ...

    @abstractmethod
    async def search_by_document(
        self,
        collection: str,
        document: Document,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """
        Find documents similar to a given document (uses the document's content as the query).

        Useful for "find more research like this" workflows.

        Args:
            collection: Target collection name.
            document: The source document to find neighbors for.
            top_k: Number of results.
            exclude_self: Whether to exclude the source document from results.
        """
        ...

    # ── Read Operations ────────────────────────────────────────────────

    @abstractmethod
    async def get(self, collection: str, ids: list[str]) -> list[Document]:
        """
        Retrieve documents by their IDs.

        Args:
            collection: Target collection name.
            ids: List of document IDs.

        Returns:
            List of Document objects (order matches input IDs where found).
        """
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Return the number of documents in the collection."""
        ...

    @abstractmethod
    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Document], str | None]:
        """
        Paginated retrieval of documents.

        Args:
            collection: Target collection name.
            limit: Max documents per page.
            offset: Pagination cursor from a previous scroll call.
            filters: Optional metadata key-value filters.

        Returns:
            Tuple of (documents, next_offset). next_offset is None when there are no more pages.
        """
        ...
