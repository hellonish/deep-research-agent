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
from models import ResearchReport, PlanStep

logger = logging.getLogger(__name__)


def _steps_to_root_nodes(steps: List[PlanStep]) -> List[ResearchNode]:
    """Build root ResearchNodes from a list of PlanStep (normalized at API boundary)."""
    return [
        ResearchNode(topic=step.description[:500], depth=0, node_id=str(i))
        for i, step in enumerate(steps)
    ]


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
        initial_plan: Optional[List[PlanStep]] = None,
        user_context: Optional[str] = None,
    ) -> ResearchReport:
        """
        Runs the full pipeline: Plan -> execute research tree (Researcher) -> Write.

        If initial_plan is provided (e.g. from research scoping), the planner is skipped.
        If user_context is provided, it is prepended to the query for scoping.
        """
        effective_query = (user_context.strip() + "\n\n" + user_query.strip()).strip() if user_context else user_query
        logger.info("Starting pipeline for: '%s'", effective_query[:80] if effective_query else "")

        # 1. Plan: root nodes from initial_plan or from planner
        root_nodes = await self._plan(effective_query, initial_plan, progress_callback)

        # 2. Execute research tree: level-by-level BFS using Researcher
        await self._execute_research_tree(root_nodes, effective_query, progress_callback)

        # 3. Write: synthesize tree into report
        logger.info("BFS complete. Writing final report.")
        report = await self.writer.write(
            effective_query, root_nodes, progress_callback=progress_callback
        )
        logger.info("Report generation complete.")
        return report

    async def _plan(
        self,
        effective_query: str,
        initial_plan: Optional[List[PlanStep]],
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]],
    ) -> List[ResearchNode]:
        """Produce root ResearchNodes: from initial_plan or by calling the planner."""
        if initial_plan:
            root_nodes = _steps_to_root_nodes(initial_plan)
            logger.info("Using provided plan with %s steps (scoping).", len(root_nodes))
        else:
            plan_response = self.planner.create_plan(
                effective_query,
                num_plan_steps=self.config.num_plan_steps,
            )
            logger.info("Planner produced %s initial topics.", len(plan_response.plan))
            root_nodes = [
                ResearchNode(topic=step.description, depth=0, node_id=str(i))
                for i, step in enumerate(plan_response.plan)
            ]
        if progress_callback:
            await progress_callback("plan_ready", {
                "probes": [{"node_id": n.node_id, "probe": n.topic[:200]} for n in root_nodes],
                "count": len(root_nodes),
            })
        return root_nodes

    async def _execute_research_tree(
        self,
        root_nodes: List[ResearchNode],
        user_query: str,
        progress_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> None:
        """
        Run level-by-level BFS over the research tree. For each level: filter (max_depth,
        visited, duplicates), process nodes in parallel via Researcher (populate + resolve),
        then build the next level from gap children. Mutates nodes in place (knowledge, children).
        """
        visited: set = set()
        level: List[ResearchNode] = list(root_nodes)
        next_id = len(root_nodes)

        while level:
            to_process: List[ResearchNode] = []
            # No semantic duplicate check: only max_depth and exact-topic visited filter.
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

            next_level: List[ResearchNode] = []
            for node in to_process:
                for child in node.children:
                    child.node_id = str(next_id)
                    next_id += 1
                    next_level.append(child)
            level = next_level

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

        # Avoid spending an extra LLM call on gap analysis when children would not execute anyway.
        # Children are depth=node.depth+1 and the executor skips nodes where depth >= max_depth.
        can_expand = (node.depth + 1) < self.config.max_depth
        knowledge_item, gap_response = await self.researcher.resolve(
            node.topic,
            user_query,
            progress_callback=progress_callback,
            compute_gaps=can_expand,
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
