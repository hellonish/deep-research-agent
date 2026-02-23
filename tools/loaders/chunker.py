"""
Recursive character text splitter for chunking documents before vector storage.

Splits on natural boundaries (paragraphs → newlines → sentences → words)
with configurable chunk size and overlap.
"""
from typing import List
from models.models import Document


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using a hierarchy of separators.

    Args:
        text: Raw text to split.
        chunk_size: Max characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Ordered list of separators to try (default: paragraph → line → sentence → word).

    Returns:
        List of text chunks.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that produces splits
    best_sep = separators[-1]
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    # Split on the best separator
    splits = text.split(best_sep)
    chunks: list[str] = []
    current_chunk = ""

    for piece in splits:
        candidate = (current_chunk + best_sep + piece).strip() if current_chunk else piece.strip()

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Flush current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + best_sep + piece
            else:
                current_chunk = piece

            # If the single piece itself exceeds chunk_size, recurse with finer separators
            if len(current_chunk) > chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:]
                if remaining_seps:
                    sub_chunks = chunk_text(current_chunk, chunk_size, chunk_overlap, remaining_seps)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    # Force split at chunk_size
                    while len(current_chunk) > chunk_size:
                        chunks.append(current_chunk[:chunk_size])
                        current_chunk = current_chunk[chunk_size - chunk_overlap:]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_filename: str = "unknown",
) -> list[Document]:
    """
    Chunk a list of Documents into smaller Documents suitable for vector storage.

    Args:
        documents: Input documents (typically one per file).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        source_filename: Original filename for metadata.

    Returns:
        List of chunked Document objects with metadata including chunk index.
    """
    chunked: list[Document] = []

    for doc in documents:
        chunks = chunk_text(doc.content, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            chunked.append(
                Document(
                    id=f"{doc.id}::chunk-{i}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": source_filename,
                    },
                )
            )

    return chunked
