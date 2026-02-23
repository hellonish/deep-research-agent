"""
Modular content block models for structured research output.

Each ContentBlock is a typed, self-contained unit that maps 1:1
to a React component on the frontend. The WriterAgent produces a
ResearchReport containing an ordered list of these blocks.

NOTE: All fields are explicit (no free-form dict) because Gemini's
structured output does not support additionalProperties.
"""

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Supported content block types."""
    TEXT        = "text"
    TABLE       = "table"
    CHART       = "chart"
    CODE        = "code"
    SOURCE_LIST = "source_list"


class ChartDataset(BaseModel):
    """A single dataset within a chart."""
    label: str = Field(description="Dataset label")
    data: List[float] = Field(description="Data values")


class ContentBlock(BaseModel):
    """
    A single modular content block with all type-specific fields.

    Use the field(s) that match the block_type:
      - text:        markdown
      - table:       headers + rows
      - chart:       chart_type + labels + datasets
      - code:        language + code
      - source_list: sources
    """
    block_type: BlockType = Field(description="The type of content block.")
    title: Optional[str] = Field(None, description="Optional section heading for this block.")

    # text block
    markdown: Optional[str] = Field(None, description="Markdown content (for text blocks).")

    # table block
    headers: Optional[List[str]] = Field(None, description="Table column headers (for table blocks).")
    rows: Optional[List[List[str]]] = Field(None, description="Table rows as lists of strings (for table blocks).")

    # chart block
    chart_type: Optional[str] = Field(None, description="Chart type: bar, line, or pie (for chart blocks).")
    labels: Optional[List[str]] = Field(None, description="Chart axis labels (for chart blocks).")
    datasets: Optional[List[ChartDataset]] = Field(None, description="Chart datasets (for chart blocks).")

    # code block
    language: Optional[str] = Field(None, description="Programming language (for code blocks).")
    code: Optional[str] = Field(None, description="Code content (for code blocks).")

    # source_list block
    sources: Optional[List[str]] = Field(None, description="List of source URLs (for source_list blocks).")


class ResearchReport(BaseModel):
    """
    The final structured output of the research pipeline.
    Serializes directly to JSON for frontend consumption.
    """
    title: str = Field(description="Report title derived from the user query.")
    summary: str = Field(description="A concise executive summary of all findings.")
    blocks: List[ContentBlock] = Field(
        default_factory=list,
        description="Ordered list of content blocks forming the report body."
    )
