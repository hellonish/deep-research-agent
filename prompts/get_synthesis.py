"""
Prompt for the ResearcherAgent — synthesizes gathered evidence into a detailed resolution.

Extracted from researcher.py to keep all prompts in the prompts/ directory.
"""


def get_synthesis(query: str, evidence: str, llm_client) -> str:
    """
    Single LLM call to synthesize gathered evidence into a comprehensive answer.

    Args:
        query: The research question being resolved.
        evidence: Raw text evidence gathered from tools and vector store.
        llm_client: The initialized LLM client.

    Returns:
        A detailed, citation-rich synthesis string.
    """
    system_prompt = """You are an expert researcher. Synthesize the gathered evidence into a highly detailed, 
comprehensive resolution for the given research question.
1. Provide in-depth explanations with specific details, metrics, and examples.
2. INLINE CITATIONS ARE REQUIRED: cite URLs within the text (e.g., 'According to https://example.com...').
3. Structure your answer with clear sections if the topic warrants it."""

    user_prompt = f"""Research Question: {query}

Gathered Evidence:
{evidence}

Write a detailed, comprehensive resolution based strictly on the evidence above. Cite sources inline."""

    return llm_client.generate_text(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )
