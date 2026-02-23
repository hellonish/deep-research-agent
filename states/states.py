from pydantic import BaseModel, Field
from typing import List, Optional

from models import ContentBlock


class KnowledgeItem(BaseModel):
    """A single piece of knowledge gathered during research."""
    source: str = Field(description="Where this knowledge came from (URL)")
    content: str = Field(description="The full content of the knowledge item.")
    summary: Optional[str] = Field(None, description="A short summary of the content.")
    sources: List[str] = Field(default_factory=list, description="Source URLs this knowledge was gathered from.")


class NodeDraft(BaseModel):
    """Per-node write output: content blocks plus a dense summary for parent synthesis."""
    node_topic: str = Field(description="The research topic this draft covers.")
    blocks: List[ContentBlock] = Field(default_factory=list, description="Content blocks for this node.")
    compressed_summary: str = Field(description="Dense 2-3 sentence summary for use by parent section.")
    local_sources: List[str] = Field(default_factory=list, description="Source URLs cited in this node (pre-global remap).")


class ResearchNode(BaseModel):
    """A node in the BFS research tree. Preserves the hierarchy of topics → gaps → sub-gaps."""
    topic: str = Field(description="The research question this node investigated.")
    depth: int = Field(description="Depth in the tree (0 = planner topic, 1+ = gap).")
    knowledge: Optional[KnowledgeItem] = Field(None, description="Resolved knowledge for this node.")
    children: List["ResearchNode"] = Field(default_factory=list, description="Child gap nodes.")
    severity: Optional[float] = Field(None, description="Gap severity score (None for root planner topics).")
    node_id: Optional[str] = Field(None, description="Stable id for progress/UI (e.g. '0', '1').") 

    def to_outline(self, indent: int = 0) -> str:
        """Render the tree as a text outline for the Writer prompt."""
        prefix = "  " * indent
        sev_str = f" [severity={self.severity:.1f}]" if self.severity is not None else ""
        lines = [f"{prefix}{'##' + '#' * indent} {self.topic}{sev_str}"]
        if self.knowledge:
            # Include a truncated version of content for the outline
            content_preview = self.knowledge.content[:500]
            lines.append(f"{prefix}Content: {content_preview}...")
            if self.knowledge.sources:
                lines.append(f"{prefix}Sources: {', '.join(self.knowledge.sources[:5])}")
        for child in self.children:
            lines.append(child.to_outline(indent + 1))
        return "\n".join(lines)
