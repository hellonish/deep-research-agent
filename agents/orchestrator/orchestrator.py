"""
OrchestratorAgent — The Hierarchical Research Manager.

Acts as the entry point to the system. Manages level-based BFS over ResearchNodes.
Initializes and coordinates the Planner, Researcher, and Writer agents.
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable, List

from llm import BaseLLMClient
from vector_store import BaseVectorStore
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.config import ReportConfig
from states import ResearchNode
from models import ResearchReport

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Manages the multi-agent execution pipeline. Runs a level-based BFS:
    process all nodes at each depth in parallel, then build the next level.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        vector_store: BaseVectorStore,
        config: Optional[ReportConfig] = None,
        collection_name: str = "research",
        *,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
    ):
        """
        Initialize the Orchestrator with shared resources and configure sub-agents.

        Args:
            llm_client: LLM client for all agents.
            vector_store: Vector store for RAG and deduplication.
            config: Report depth/breadth config; defaults to STANDARD.
            collection_name: Qdrant collection name (e.g. user-scoped).
            planner: Optional pre-built PlannerAgent; created from llm_client if None.
            researcher: Optional pre-built ResearcherAgent; created if None.
            writer: Optional pre-built WriterAgent; created from llm_client if None.
        """
        self.config = config or ReportConfig.STANDARD()
        self.vector_store = vector_store
        self.collection_name = collection_name

        self.planner = planner or PlannerAgent(llm_client=llm_client)
        self.researcher = researcher or ResearcherAgent(
            llm_client=llm_client,
            vector_store=vector_store,
            collection_name=collection_name,
            config=self.config,
        )
        self.writer = writer or WriterAgent(llm_client=llm_client, config=self.config)

    def _scoped_callback(
        self,
        node: ResearchNode,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]],
    ):
        """Return an async callable that injects node_id into every event."""
        if progress_callback is None:
            async def noop(_event_type: str, _data: dict) -> None:
                pass
            return noop

        async def scoped(event_type: str, data: dict) -> None:
            await progress_callback(event_type, {**data, "node_id": node.node_id})

        return scoped

    async def run(
        self,
        user_query: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> ResearchReport:
        """
        Runs the full research pipeline (Plan -> Level BFS -> Write) for a given query.

        Returns:
            ResearchReport: The synthesized report with citations and content blocks.
        """
        logger.info("Starting pipeline for: '%s'", user_query)

        # 1. Plan
        plan_response = self.planner.create_plan(
            user_query,
            num_plan_steps=self.config.num_plan_steps,
        )
        logger.info("Planner produced %s initial topics.", len(plan_response.plan))

        # 2. Build root nodes with node_id
        root_nodes: List[ResearchNode] = [
            ResearchNode(topic=step.description, depth=0, node_id=str(i))
            for i, step in enumerate(plan_response.plan)
        ]

        if progress_callback:
            await progress_callback("plan_ready", {
                "probes": [{"node_id": n.node_id, "probe": n.topic[:200]} for n in root_nodes],
                "count": len(root_nodes),
            })

        # 3. Level-based BFS
        visited: set = set()
        level: List[ResearchNode] = list(root_nodes)
        next_id = len(root_nodes)

        while level:
            # Filter: skip max_depth, already visited; for depth >= 1 run is_duplicate
            to_process: List[ResearchNode] = []
            if level[0].depth >= 1:
                dupes = await asyncio.gather(
                    *[self.researcher.is_duplicate(n.topic) for n in level]
                )
                for n, is_dup in zip(level, dupes):
                    if n.depth >= self.config.max_depth or n.topic in visited:
                        continue
                    if is_dup:
                        logger.info(
                            "Node '%s...' is a semantic duplicate. Skipping.",
                            n.topic[:30],
                        )
                        visited.add(n.topic)
                        continue
                    visited.add(n.topic)
                    to_process.append(n)
            else:
                for n in level:
                    if n.depth >= self.config.max_depth or n.topic in visited:
                        continue
                    visited.add(n.topic)
                    to_process.append(n)

            if not to_process:
                level = []
                continue

            current_depth = to_process[0].depth
            if progress_callback:
                await progress_callback("level_start", {
                    "depth": current_depth,
                    "probes": [{"node_id": n.node_id, "probe": n.topic[:200]} for n in to_process],
                    "total_in_level": len(to_process),
                })

            await asyncio.gather(*[
                self._process_node(n, user_query, self._scoped_callback(n, progress_callback))
                for n in to_process
            ])

            if progress_callback:
                await progress_callback("level_complete", {
                    "depth": current_depth,
                    "completed": [{"node_id": n.node_id, "probe": n.topic[:200]} for n in to_process],
                })

            # Build next level and assign node_id to children
            next_level: List[ResearchNode] = []
            for node in to_process:
                for child in node.children:
                    child.node_id = str(next_id)
                    next_id += 1
                    next_level.append(child)
            level = next_level

        # 4. Write synthesis
        logger.info("BFS complete. Writing final report.")
        report = await self.writer.write(
            user_query, root_nodes, progress_callback=progress_callback
        )
        logger.info("Report generation complete.")
        return report

    async def _process_node(
        self,
        node: ResearchNode,
        user_query: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> None:
        """
        Populate memory, resolve the topic, attach knowledge to the node,
        and create child gap nodes if not complete. Does not enqueue; caller builds next level.
        """
        logger.info("Exploring node '%s' (depth=%s)", node.topic[:60], node.depth)

        if progress_callback:
            await progress_callback(
                "probe_start",
                {"probe": node.topic[:200], "depth": node.depth},
            )

        await self.researcher.populate(node.topic, progress_callback=progress_callback)

        knowledge_item, gap_response = await self.researcher.resolve(
            node.topic, user_query, progress_callback=progress_callback
        )

        node.knowledge = knowledge_item

        if progress_callback:
            await progress_callback("probe_complete", {"probe": node.topic[:200], "knowledge_items": 1})

        if not gap_response.is_complete and gap_response.gaps:
            surviving_gaps = self.researcher._threshold_filter(gap_response.gaps)
            for gap in surviving_gaps:
                child = ResearchNode(
                    topic=gap.query,
                    depth=node.depth + 1,
                    severity=gap.severity,
                )
                node.children.append(child)
        elif gap_response.is_complete:
            logger.info(
                "Topic '%s...' is marked complete. No children added.",
                node.topic[:30],
            )
