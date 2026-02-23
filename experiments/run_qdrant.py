"""
Run from repo root: python experiments/run_qdrant.py

Loads all experiment docs (doc1–3, doc_sparse, doc_dense, doc_apple_*) into Qdrant,
then runs three demo queries to show sparse / dense / hybrid superiority.
"""
import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from models import Document, SearchQuery, SearchResult
from vector_store import QdrantStore

COLLECTION = "test_collection"
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_FILES = (
    "doc1.txt",
    "doc2.txt",
    "doc3.txt",
    "doc_sparse.txt",
    "doc_dense.txt",
    "doc_apple_company.txt",
    "doc_apple_fruit.txt",
)


def load_docs() -> list[Document]:
    """Load all experiment .txt files as Document objects."""
    docs = []
    for name in DOC_FILES:
        path = os.path.join(EXPERIMENTS_DIR, name)
        if not os.path.isfile(path):
            print(f"  skip (not found): {name}")
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        docs.append(
            Document(
                id=str(uuid.uuid4()),
                content=content,
                metadata={"source": name},
            )
        )
        print(f"  loaded: {name} ({len(content)} chars)")
    return docs


def print_results(label: str, results: list, top_k: int = 3) -> None:
    for i, r in enumerate(results[:top_k], 1):
        print(f"  {i}. score={r.score:.4f} source={r.metadata.get('source', '?')}")
        print(f"     {r.content[:100].replace(chr(10), ' ')}...")


def merge_hybrid_score_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int = 10,
    dense_weight: float = 0.5,
) -> list[SearchResult]:
    """
    Combine dense + sparse by score fusion (no RRF).
    Normalizes each modality's scores to [0, 1] via min-max, then
    combined_score = dense_weight * norm_dense + (1 - dense_weight) * norm_sparse.
    """
    def _min_max_norm(scores: list[float]) -> list[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        span = hi - lo or 1.0
        return [(s - lo) / span for s in scores]

    by_id: dict[str, SearchResult] = {}
    dense_scores = []
    for r in dense_results:
        by_id[r.id] = SearchResult(id=r.id, content=r.content, score=r.score, metadata=r.metadata)
        dense_scores.append(r.score)
    dense_norm = _min_max_norm(dense_scores)
    for i, r in enumerate(dense_results):
        by_id[r.id].metadata = {**by_id[r.id].metadata, "_norm_dense": dense_norm[i]}

    sparse_scores = []
    for r in sparse_results:
        if r.id not in by_id:
            by_id[r.id] = SearchResult(id=r.id, content=r.content, score=r.score, metadata=r.metadata)
            by_id[r.id].metadata["_norm_dense"] = 0.0
        sparse_scores.append(r.score)
        by_id[r.id].metadata["_norm_sparse"] = 0.0  # placeholder
    sparse_norm = _min_max_norm(sparse_scores)
    for i, r in enumerate(sparse_results):
        by_id[r.id].metadata["_norm_sparse"] = sparse_norm[i]
        if "_norm_dense" not in by_id[r.id].metadata:
            by_id[r.id].metadata["_norm_dense"] = 0.0

    combined = []
    for rid, res in by_id.items():
        nd = res.metadata.get("_norm_dense", 0.0)
        ns = res.metadata.get("_norm_sparse", 0.0)
        combined_score = dense_weight * nd + (1 - dense_weight) * ns
        combined.append(
            SearchResult(
                id=res.id,
                content=res.content,
                score=round(combined_score, 6),
                metadata={k: v for k, v in res.metadata.items() if not k.startswith("_")},
            )
        )
    combined.sort(key=lambda x: -x.score)
    return combined[:top_k]


async def main():
    store = QdrantStore(
        url=os.getenv("QDRANT_LOCATION"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    print("is_remote:", store.is_remote())
    print("location:", store.location)
    print("connected:", await store.check_connection())

    if not await store.collection_exists(COLLECTION):
        await store.create_collection(COLLECTION, dense_dim=384)
        print("created", COLLECTION)
    else:
        print("collection exists:", COLLECTION)

    # Insert documents
    print("\n--- Loading documents ---")
    documents = load_docs()
    if not documents:
        print("No documents to upsert. Add some .txt files to experiments/")
        return
    await store.upsert(COLLECTION, documents)
    print(f"Upserted {len(documents)} documents.")

    # Demo queries: each chosen so one search type should rank the "right" doc #1
    demos = [
        (
            "SPARSE wins (exact terms): 429, RFC 6585",
            "What is error code 429 and RFC 6585?",
            "doc_sparse.txt",
        ),
        (
            "DENSE wins (paraphrase, no keyword 'overfitting')",
            "How do I stop my model from memorizing the training data?",
            "doc_dense.txt",
        ),
        (
            "HYBRID wins (ambiguous: Apple company vs fruit)",
            "Apple latest product launch and stock price",
            "doc_apple_company.txt",
        ),
    ]

    for title, query_text, expected_source in demos:
        print(f"\n{'='*60}")
        print(f"Query: {query_text!r}")
        print(f"Expected #1: {expected_source}")
        print("=" * 60)
        query = SearchQuery(text=query_text, top_k=3)
        query_pool = SearchQuery(text=query_text, top_k=10)  # larger pool for score fusion
        print("\n  Sparse:")
        print_results("sparse", await store.search_sparse(COLLECTION, query))
        print("\n  Dense:")
        print_results("dense", await store.search_dense(COLLECTION, query))
        print("\n  Hybrid (RRF — rank-based fusion):")
        print_results("hybrid", await store.search(COLLECTION, query))
        # Hybrid without RRF: fuse dense + sparse by normalized score (min-max then 0.5*d + 0.5*s)
        dense_pool = await store.search_dense(COLLECTION, query_pool)
        sparse_pool = await store.search_sparse(COLLECTION, query_pool)
        hybrid_no_rrf = merge_hybrid_score_fusion(dense_pool, sparse_pool, top_k=3, dense_weight=0.5)
        print("\n  Hybrid (score fusion, no RRF):")
        print_results("hybrid_no_rrf", hybrid_no_rrf)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
