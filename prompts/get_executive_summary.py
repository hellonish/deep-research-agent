"""
Tiny LLM call to produce report title and executive summary from section summaries only.

Used at the end of the bottom-up writer pipeline; input is compressed content only.
"""

from typing import Tuple

from pydantic import BaseModel, Field


class ExecutiveSummaryOutput(BaseModel):
    """Title and executive summary for the final report."""
    title: str = Field(description="A compelling report title derived from the user query and findings.")
    summary: str = Field(description="Executive summary: 2-3 paragraphs in markdown covering the main findings.")


def get_executive_summary(
    user_query: str,
    section_titles: list[str],
    section_summaries: list[str],
    llm_client,
) -> Tuple[str, str]:
    """
    Single small LLM call: produce report title and executive summary from
    section titles and compressed summaries only. Returns (title, summary_markdown).
    """
    sections_text = "\n\n".join(
        f"## {title}\n{summary}"
        for title, summary in zip(section_titles, section_summaries)
    )

    system_prompt = """You are an expert Research Writer. Given the original user query and compressed summaries of each major section of a report, produce:
1. title: A short, compelling report title.
2. summary: An executive summary in 2-3 paragraphs (markdown). Synthesize the key findings across sections; do not list section titles. Be concise and high-level."""

    prompt = f"""Original query: {user_query}

Section summaries (each section's key findings):
{sections_text}

Produce a report title and an executive summary (2-3 paragraphs) that synthesize these findings."""

    response = llm_client.generate_structured(
        prompt=prompt,
        system_prompt=system_prompt,
        schema=ExecutiveSummaryOutput,
        temperature=0.4,
    )
    return response.title, response.summary
