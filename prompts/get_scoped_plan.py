"""
Generate a research plan plus clarifying questions to scope the research
and reduce hallucinations. User can then refine the plan and provide answers
before running the full research.
"""
from models import ScopedPlan


def get_scoped_plan(user_prompt: str, llm_client, num_plan_steps: int = 5) -> ScopedPlan:
    """
    Analyzes the user's research request and returns a structured plan
    plus clarifying questions to narrow scope and clarify intent.

    Args:
        user_prompt: The research task from the user.
        llm_client: The LLM client (e.g. Gemini).
        num_plan_steps: Number of plan steps to generate.

    Returns:
        ScopedPlan: Plan (query_type, steps) and list of clarifying questions.
    """
    system_prompt = (
        "You are an expert Research Planner.\n"
        "\n"
        "CRITICAL: The plan will be executed in PARALLEL by multiple workers.\n"
        "Your plan steps MUST be independent and self-contained:\n"
        "- No step should depend on another step's output.\n"
        "- Do not reference other steps (no 'based on step 1', 'then', 'after').\n"
        "- Each step should cover a distinct slice of the work with minimal overlap.\n"
        "- If some context is needed, include it directly inside the step's description.\n"
        "\n"
        "Your job is to:\n"
        "1. Categorize the request into the most appropriate domain.\n"
        "2. Create a parallelizable research plan with exactly {num_plan_steps} distinct steps.\n"
        "3. For each step provide:\n"
        "   - 'action': a short summary.\n"
        "   - 'description': detailed execution instructions including what to find/verify, where to look (source types),\n"
        "     what to extract, and a completion/success criterion.\n"
        "\n"
        "Clarifying questions:\n"
        "- Generate 0–5 clarifying_questions ONLY IF they are necessary to avoid wrong assumptions\n"
        "  or to choose between materially different scopes.\n"
        "- If the request is sufficiently specific to proceed safely, return an empty list.\n"
        "- Good question types (only when needed): time window/recency, geography, depth, constraints,\n"
        "  inclusion/exclusion, definitions of key terms, decision criteria.\n"
    ).format(num_plan_steps=num_plan_steps)

    response = llm_client.generate_structured(
        prompt=f"Create a scoped research plan for: {user_prompt}",
        system_prompt=system_prompt,
        schema=ScopedPlan,
        temperature=0.2,
    )
    return response
