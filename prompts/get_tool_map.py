from models.models import ToolMap

# Backward-compat alias (schema was renamed ToolPlan -> ToolMap in models)
ToolPlan = ToolMap


def get_tool_map(topic: str, llm_client, max_tool_pairs: int = 3) -> ToolMap:
    """
    Asks the LLM to select the best tools and queries to research a topic.

    Args:
        topic: The research topic or gap query to investigate.
        llm_client: The initialized LLM client.
        max_tool_pairs: Maximum number of [tool, query/url] pairs to generate.

    Returns:
        ToolMap: A list of ToolPair objects specifying which tools to call.
    """
    system_prompt = """You are a research strategist. Given a research topic, select the best combination of tools to gather comprehensive information.

AVAILABLE TOOLS:
- `tavily_search`: Best for general web search, recent events, and broad topics. Provide a `query` string.
- `duckduckgo_search`: Alternative web search. Provide a `query` string.
- `serpapi_search`: Best for precise Google results or news. Provide a `query` string.
- `arxiv_loader`: Best for academic papers, deep technical research. Provide a `query` string with search terms.
- `firecrawl_scrape`: Extracts full text from a specific URL. Provide a `url` string. Only use if you have a known URL.
- `bs4_scrape`: Alternative fast scraper for a specific URL. Provide a `url` string. Only use if you have a known URL.

RULES:
1. Select up to {max_tool_pairs} tools maximum.
2. Vary your tool selection — do NOT pick the same tool twice unless with very different queries.
3. For search tools, provide a `query`. For scrape tools, provide a `url`.
4. Pick fewer tools if the topic is narrow. Use more if the topic is broad.
"""

    user_prompt = """Topic to research: {topic}

Select the best tools and queries to gather information on this topic."""

    return llm_client.generate_structured(
        prompt=user_prompt.format(topic=topic),
        system_prompt=system_prompt.format(max_tool_pairs=max_tool_pairs),
        schema=ToolMap,
        temperature=0.2,
    )
