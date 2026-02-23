"""
Per-node write prompt for the bottom-up Writer pipeline.

Produces a NodeDraft: content blocks for this section plus a compressed_summary
for the parent to use. Input is the node's own KnowledgeItem content and
(optionally) child summaries.
"""

from states import NodeDraft


def get_write_node(
    node_topic: str,
    own_content: str,
    own_sources: list[str],
    child_summaries: list[str],
    user_query: str,
    llm_client,
) -> NodeDraft:
    """
    One LLM call per node: write this section's blocks and a dense summary.
    """
    sources_str = (
        "\n".join(f"[{i+1}] {url}" for i, url in enumerate(own_sources))
        if own_sources
        else "No sources for this section."
    )

    child_section = ""
    if child_summaries:
        child_section = """
SUMMARIES FROM CHILD SECTIONS (for context; do not repeat verbatim):
""" + "\n\n".join(f"- {s}" for s in child_summaries)

    system_prompt = """You are an expert Research Writer. For a single section of a larger report, you will receive:
1. The section topic and its researched content.
2. Optional: short summaries from child subsections (for context only).
3. The original user query.

Your tasks:
1. Produce 1–4 content blocks for THIS section only. Use substantive "text" blocks with inline citations [1], [2]. When this section's content has comparable items (e.g. features, pricing, options) or numeric data (benchmarks, trends), add one "table" or "chart" block with that real data—do not add filler. Use "code" only for actual snippets from the findings. Do NOT include a "source_list" block here (that is assembled later).
2. Write a "compressed_summary": exactly 2–3 dense sentences summarizing this section's key findings and numbers, for use by a parent section. Be concise and factual.

BLOCK TYPES: "text" (markdown, primary), "table" (when 3+ comparable items or real specs from findings), "chart" (when numeric/trend data from findings), "code" (actual code only). Cite sources as [1], [2] in markdown."""

    prompt = f"""Write this section of the report.

ORIGINAL QUERY: {user_query}

SECTION TOPIC: {node_topic}

SECTION CONTENT:
{own_content}
{child_section}

SOURCES (cite using these indices in your blocks):
{sources_str}

Return a NodeDraft with:
- blocks: list of content blocks for this section only (no source_list block)
- compressed_summary: 2–3 sentence dense summary of this section's main findings for a parent section"""

    response = llm_client.generate_structured(
        prompt=prompt,
        system_prompt=system_prompt,
        schema=NodeDraft,
        temperature=0.35,
    )

    # Ensure local_sources is set from input (LLM may not populate it reliably)
    if not response.local_sources and own_sources:
        response = response.model_copy(update={"local_sources": own_sources})
    return response
