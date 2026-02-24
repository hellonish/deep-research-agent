"""
ResearcherAgent — Clean modular Populate & Resolve architecture.

Core responsibilities:
  1. _get_tools_map: Maps a query to a list of optimal tool calls (ToolPlan).
  2. populate: Executes mapped tools and stores raw findings in vector memory.
  3. resolve: Uses memory to synthesize answers, judge them, and extract gaps.
"""

import asyncio
import logging
import uuid
from typing import Awaitable, Callable, List, Optional

from llm import BaseLLMClient
from vector_store import BaseVectorStore
from models import Document, Gap, SearchQuery, ToolMap, ToolPair, GapAnalysis
from states import KnowledgeItem
from tools import ToolExecutor
from agents.config import ReportConfig
from prompts import get_tool_map, get_synthesis, get_gaps

logger = logging.getLogger(__name__)

# Max content length per document chunk stored in vector store
_CONTENT_MAX_LEN = 4000


def _tool_pair_to_kwargs(pair: ToolPair) -> dict:
    """Build execution kwargs from a ToolPair."""
    kwargs: dict = {}
    if pair.query:
        kwargs["query"] = pair.query
    if pair.url:
        kwargs["url"] = pair.url
    if pair.tool_name == "arxiv_loader" and pair.query:
        kwargs["source"] = pair.query
    if pair.tool_name == "pdf_loader" and pair.url:
        kwargs["source"] = pair.url
    return kwargs


class ResearcherAgent:
    """
    Independently researches a given topic by dispatching tools and storing
    the learned knowledge in the vector store.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        vector_store: Optional[BaseVectorStore] = None,
        collection_name: str = "research",
        config: Optional[ReportConfig] = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.config = config or ReportConfig.STANDARD()
        max_tavily = getattr(self.config, "max_tavily_calls", 0) or 0
        self.tool_executor = ToolExecutor(max_tavily_calls=max_tavily if max_tavily > 0 else None)

    async def populate(
        self,
        query: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> List[str]:
        """
        Executes tools based on the query, extracts data, and stores it
        in the vector store. Returns list of gathered source URLs.
        """
        logger.debug("Populating memory for: '%s...'", query[:50])

        tool_map = await self._get_tools_map(query)

        gathered_sources: List[str] = []
        for pair in tool_map.tool_pairs:
            try:
                if progress_callback:
                    await progress_callback(
                        "tool_call",
                        {
                            "tool": pair.tool_name,
                            "query": pair.query or pair.url or "",
                        },
                    )
                kwargs = _tool_pair_to_kwargs(pair)
                logger.debug("Tool %s: %s", pair.tool_name, pair.query or pair.url)
                raw_result = await self.tool_executor.execute(
                    pair.tool_name, **kwargs
                )

                docs = self._raw_result_to_docs(
                    raw_result, pair, query, gathered_sources
                )
                if docs and self.vector_store:
                    await self.vector_store.upsert(self.collection_name, docs)
            except Exception as e:
                logger.warning(
                    "Tool execution failed (%s): %s", pair.tool_name, e
                )

        return gathered_sources

    async def resolve(
        self,
        gap_query: str,
        original_goal: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        *,
        compute_gaps: bool = True,
    ) -> tuple[KnowledgeItem, GapAnalysis]:
        """
        Uses embeddings to fetch context, synthesizes an answer, and asks the LLM
        to judge the answer to find remaining gaps.
        """
        logger.debug("Resolving gap: '%s...'", gap_query[:50])

        # 1. Fetch relevant memory context via RAG
        context_data, rag_sources = await self._get_context_for_query(gap_query)

        # 2. On memory miss, populate and re-fetch
        populate_sources: List[str] = []
        if not context_data:
            logger.info("Memory miss. Populating memory directly for gap.")
            populate_sources = await self.populate(gap_query, progress_callback=progress_callback)
            context_data, rag_sources = await self._get_context_for_query(gap_query)
            if not context_data:
                context_data = "No data found."

        # Merge: populate URLs first, then any additional URLs surfaced by RAG metadata
        gathered_sources: List[str] = list(populate_sources)
        for url in rag_sources:
            if url not in gathered_sources:
                gathered_sources.append(url)

        # 3. Synthesize the final answer
        if progress_callback:
            await progress_callback(
                "thinking",
                {"probe": gap_query[:200], "message": "Synthesizing findings..."},
            )

        answer = await asyncio.to_thread(
            get_synthesis,
            query=gap_query,
            evidence=context_data,
            llm_client=self.llm_client,
        )

        item = KnowledgeItem(
            source="Agent Resolution",
            content=answer,
            summary=f"Resolved details for: {gap_query[:50]}",
            sources=gathered_sources,
        )

        # Save synthesized item back to memory
        if self.vector_store:
            try:
                doc = Document(
                    id=str(uuid.uuid4()),
                    content=item.content,
                    metadata={"source": item.source, "summary": item.summary or ""},
                )
                await self.vector_store.upsert(self.collection_name, [doc])
            except Exception:
                pass

        # 4. Judge and extract gaps (optional)
        # If we won't expand the tree further (e.g. leaf depth), skip the extra LLM call.
        if compute_gaps and self.config.max_probes > 0:
            logger.debug("Judging resolution for remaining gaps.")
            gap_response = await asyncio.to_thread(
                get_gaps,
                original_goal=original_goal,
                current_topic=gap_query,
                findings=answer,
                llm_client=self.llm_client,
                max_probes=self.config.max_probes,
            )
        else:
            gap_response = GapAnalysis(gaps=[], is_complete=True)

        return item, gap_response

    def _raw_result_to_docs(
        self,
        raw_result: object,
        pair: ToolPair,
        query: str,
        gathered_sources: List[str],
    ) -> List[Document]:
        """Convert tool raw result into a list of Documents for upsert."""
        
        docs: List[Document] = []
        
        if isinstance(raw_result, list):
            for item in raw_result:
                url = None
                content = getattr(item, "content", str(item))
                if hasattr(item, "metadata") and isinstance(item.metadata, dict):
                    url = item.metadata.get("url") or item.metadata.get("source")
                if not url and hasattr(item, "id") and str(item.id).startswith("http"):
                    url = str(item.id)
                if url:
                    gathered_sources.append(url)
                docs.append(
                    Document(
                        id=str(uuid.uuid4()),
                        content=content[:_CONTENT_MAX_LEN],
                        metadata={"source": url or pair.tool_name, "topic": query},
                    )
                )
        elif isinstance(raw_result, str) and raw_result.strip():
            source_url = pair.url or pair.query or pair.tool_name
            if pair.url:
                gathered_sources.append(pair.url)
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=raw_result[:_CONTENT_MAX_LEN],
                    metadata={"source": source_url, "topic": query},
                )
            )
        return docs

    async def _get_context_for_query(self, query: str) -> tuple[str, List[str]]:
        """
        Fetch relevant memory context for a query via RAG.
        Returns (formatted context string, deduplicated source URLs from result metadata).
        """
        if not self.vector_store:
            return "No data found.", []
        
        try:
            results = await self.vector_store.search(
                self.collection_name,
                SearchQuery(text=query, top_k=self.config.rag_top_k),
            )

            if not results:
                return "No data found.", []

            sources: List[str] = []
            
            for r in results:
                url = r.metadata.get("source", "")
                if url and url.startswith("http") and url not in sources:
                    sources.append(url)

            context = "\n\n".join(
                f"[Source Fragment: {r.score:.2f}]\n{r.content}" for r in results
            )

            return context, sources

        except Exception:
            return "", []

    async def _get_tools_map(self, query: str) -> ToolMap:
        """Determine which tools are best suited to research the given query."""
        logger.debug("Mapping tools for: '%s...'", query[:50])

        tool_map = await asyncio.to_thread(
            get_tool_map, query, self.llm_client, self.config.max_tool_pairs
        )

        logger.debug("Selected %s tool executions.", len(tool_map.tool_pairs))
        return tool_map

    def _threshold_filter(
        self, gaps: List[Gap], drop_bottom_fraction: float = 0.25
    ) -> List[Gap]:
        """
        Drop the lowest-severity 25% of gaps; keep the top 75%.
        """
        if not gaps or drop_bottom_fraction <= 0:
            return list(gaps)
        if drop_bottom_fraction >= 1:
            return []

        sorted_gaps = sorted(gaps, key=lambda g: g.severity, reverse=True)
        keep_count = max(1, int(len(sorted_gaps) * (1 - drop_bottom_fraction)))
        surviving = sorted_gaps[:keep_count]

        logger.debug(
            "Filtered gaps: %s -> %s (dropped bottom %.0f%%)",
            len(gaps),
            len(surviving),
            drop_bottom_fraction * 100,
        )
        return surviving

    async def is_duplicate(self, query: str) -> bool:
        """Check if a query is too similar to existing knowledge (deduplication)."""
        if not self.vector_store:
            return False
        try:
            results = await self.vector_store.search(
                self.collection_name,
                SearchQuery(
                    text=query,
                    top_k=1,
                    score_threshold=self.config.dupe_threshold,
                ),
            )
            return len(results) > 0
        except Exception:
            return False
