"""
WriterAgent — transforms a ResearchNode tree into a structured ResearchReport.

Supports two modes (ReportConfig.writer_mode):
  - "single": one-shot get_write (original behavior).
  - "bottomup": per-node get_write_node, citation merge, get_executive_summary.
"""

import asyncio
import logging
import re
from typing import Awaitable, Callable, List, Optional, Tuple

from llm import BaseLLMClient
from states import KnowledgeItem, NodeDraft, ResearchNode
from models import ResearchReport, ContentBlock, BlockType
from prompts import get_write, get_write_node, get_executive_summary
from agents.config import ReportConfig

logger = logging.getLogger(__name__)


def _flatten_drafts_preorder(draft: NodeDraft, child_trees: List[Tuple[NodeDraft, list]]) -> List[NodeDraft]:
    """Flatten (draft, child_trees) in pre-order to a list of NodeDraft."""
    out: List[NodeDraft] = [draft]
    for _draft, _children in child_trees:
        out.extend(_flatten_drafts_preorder(_draft, _children))
    return out


class WriterAgent:
    """
    Takes the full research tree built by the Orchestrator and produces
    a structured ResearchReport with modular content blocks.
    """

    def __init__(self, llm_client: BaseLLMClient, config: Optional[ReportConfig] = None):
        self.llm_client = llm_client
        self.config = config or ReportConfig.STANDARD()

    def _collect_knowledge(self, nodes: List[ResearchNode]) -> List[KnowledgeItem]:
        """Recursively collect all KnowledgeItems from the tree."""
        items: List[KnowledgeItem] = []
        for node in nodes:
            if node.knowledge:
                items.append(node.knowledge)
            if node.children:
                items.extend(self._collect_knowledge(node.children))
        return items

    def _node_to_digest(
        self,
        node: ResearchNode,
        source_map: dict,
        section_num: int = 1,
        indent: int = 0,
    ) -> str:
        """Convert a ResearchNode and its children into a structured digest string."""
        prefix = "  " * indent
        heading_level = "#" * (indent + 2)
        lines = []
        sev_str = (
            f" [severity={node.severity:.1f}]" if node.severity is not None else ""
        )
        lines.append(f"\n{prefix}{heading_level} Section: {node.topic}{sev_str}\n")
        if node.knowledge:
            lines.append(f"{prefix}**Content:**")
            lines.append(f"{prefix}{node.knowledge.content}")
            if node.knowledge.sources:
                refs = [
                    f"[{source_map.get(url, '?')}]"
                    for url in node.knowledge.sources
                    if url in source_map
                ]
                lines.append(f"{prefix}**Sources:** {', '.join(refs)}")
            lines.append("")
        for i, child in enumerate(node.children, 1):
            lines.append(
                self._node_to_digest(child, source_map, section_num=i, indent=indent + 1)
            )
        return "\n".join(lines)

    async def _write_node_postorder(
        self,
        node: ResearchNode,
        user_query: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> Tuple[NodeDraft, List[Tuple[NodeDraft, list]]]:
        """
        Post-order: write children first, then this node. Returns (draft, child_trees).
        """
        child_trees: List[Tuple[NodeDraft, list]] = []
        for child in node.children:
            ct = await self._write_node_postorder(
                child, user_query, progress_callback=progress_callback
            )
            child_trees.append(ct)
        child_summaries = [t[0].compressed_summary for t in child_trees]
        own_content = node.knowledge.content if node.knowledge else ""
        own_sources = list(node.knowledge.sources) if node.knowledge else []

        if progress_callback:
            await progress_callback("section_start", {"node_topic": node.topic[:200]})
        def _call():
            return get_write_node(
                node_topic=node.topic,
                own_content=own_content,
                own_sources=own_sources,
                child_summaries=child_summaries,
                user_query=user_query,
                llm_client=self.llm_client,
            )
        draft = await asyncio.to_thread(_call)
        if progress_callback:
            await progress_callback(
                "section_complete",
                {"node_topic": node.topic[:200], "blocks_count": len(draft.blocks)},
            )
        return (draft, child_trees)

    def _merge_citations(
        self, drafts: List[NodeDraft]
    ) -> Tuple[List[ContentBlock], List[str]]:
        """
        Build global source list from all drafts (dedup), remap [n] in block text
        to global indices. Returns (merged_blocks, global_sources).
        """
        global_sources: List[str] = []
        url_to_global: dict = {}
        for d in drafts:
            for url in d.local_sources:
                if url not in url_to_global:
                    url_to_global[url] = len(global_sources) + 1
                    global_sources.append(url)

        merged: List[ContentBlock] = []
        for d in drafts:
            local_to_global: dict = {}
            for i, url in enumerate(d.local_sources, 1):
                local_to_global[i] = url_to_global.get(url, i)
            for blk in d.blocks:
                new_blk = blk.model_copy(deep=True)
                if new_blk.markdown:
                    for local_idx in sorted(local_to_global.keys(), reverse=True):
                        global_idx = local_to_global[local_idx]
                        if local_idx != global_idx:
                            new_blk.markdown = re.sub(
                                r"\[" + str(local_idx) + r"\]",
                                f"[{global_idx}]",
                                new_blk.markdown,
                            )
                merged.append(new_blk)
        return merged, global_sources

    async def _write_bottomup(
        self,
        user_query: str,
        research_tree: List[ResearchNode],
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> ResearchReport:
        """Bottom-up pipeline: post-order per node, merge citations, exec summary."""
        root_trees = await asyncio.gather(
            *[
                self._write_node_postorder(r, user_query, progress_callback=progress_callback)
                for r in research_tree
            ]
        )
        all_drafts: List[NodeDraft] = []
        for draft, child_trees in root_trees:
            all_drafts.extend(_flatten_drafts_preorder(draft, child_trees))

        logger.info(
            "Bottom-up write: %s node drafts, merging citations.",
            len(all_drafts),
        )
        merged_blocks, global_sources = self._merge_citations(all_drafts)

        root_drafts = [t[0] for t in root_trees]
        section_titles = [d.node_topic for d in root_drafts]
        section_summaries = [d.compressed_summary for d in root_drafts]

        def _exec_summary():
            return get_executive_summary(
                user_query=user_query,
                section_titles=section_titles,
                section_summaries=section_summaries,
                llm_client=self.llm_client,
            )

        title, summary = await asyncio.to_thread(_exec_summary)

        if global_sources:
            merged_blocks.append(
                ContentBlock(block_type=BlockType.SOURCE_LIST, sources=global_sources)
            )
        return ResearchReport(title=title, summary=summary, blocks=merged_blocks)

    async def write(
        self,
        user_query: str,
        research_tree: List[ResearchNode],
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> ResearchReport:
        """
        Synthesize the research tree into a structured report.
        Uses writer_mode: "single" (one-shot) or "bottomup" (per-node + exec summary).
        """
        if self.config.writer_mode == "bottomup":
            return await self._write_bottomup(
                user_query, research_tree, progress_callback=progress_callback
            )

        all_items = self._collect_knowledge(research_tree)
        logger.info(
            "Synthesizing %s items from %s root topics (single-shot).",
            len(all_items),
            len(research_tree),
        )
        all_sources: List[str] = []
        for item in all_items:
            for url in item.sources:
                if url not in all_sources:
                    all_sources.append(url)
        source_map = {url: i + 1 for i, url in enumerate(all_sources)}
        digest_parts: List[str] = []
        for i, root in enumerate(research_tree, 1):
            digest_parts.append(self._node_to_digest(root, source_map, section_num=i))
        knowledge_digest = "".join(digest_parts)
        sources_str = (
            "\n".join(f"[{i+1}] {url}" for i, url in enumerate(all_sources))
            if all_sources
            else "No sources."
        )

        def call_llm():
            return get_write(
                user_query=user_query,
                knowledge_digest=knowledge_digest,
                sources_str=sources_str,
                llm_client=self.llm_client,
            )

        report = await asyncio.to_thread(call_llm)
        logger.info(
            "Report generated: '%s' with %s blocks",
            report.title,
            len(report.blocks),
        )
        return report
