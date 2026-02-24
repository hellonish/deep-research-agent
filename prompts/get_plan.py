from models import ResearchPlan

def get_plan(user_prompt: str, llm_client, num_plan_steps: int = 5) -> ResearchPlan:
    """
    Analyzes a user's research request, categorizes it, and generates a structured plan.
    
    Args:
        user_prompt (str): The research task from the user.
        llm_client: The initialized LLM client (e.g., OpenAI client).
        
    Returns:
        ResearchPlan: A structured Pydantic object containing the query type and steps.
    """
    
    system_prompt = (
        """You are an expert Research Planner Agent. Your objective is to analyze the user's
        research request and break it down into a highly methodical, actionable plan.

        IMPORTANT: The plan will be executed in PARALLEL by multiple workers.
        Therefore, each step must be INDEPENDENT and SELF-CONTAINED:
        - No step may require outputs from another step.
        - Avoid sequencing language like "first", "then", "after". Write steps as parallel workstreams.
        - Minimize overlap/duplication between steps; each step should cover a distinct slice of the problem.
        - If a dependency is unavoidable, restate the required context inside the step itself (do not reference other steps).

        Instructions:
        1. Categorize the request into the most appropriate domain provided in the schema.
        2. Create a parallelizable research plan.
        3. Break the plan down into exactly {num_plan_steps} distinct steps.
        4. For each step:
           - 'action': A short, simple summary for the user.
           - 'description': Detailed execution instructions including:
             * what to find/verify (specific facts/questions),
             * where to look (source types),
             * what to extract (fields/notes),
             * and a clear success criterion for completion.
        """.format(num_plan_steps=num_plan_steps)
    )

    response = llm_client.generate_structured(
        prompt=f"Create a research plan for the following request: {user_prompt}",
        system_prompt=system_prompt,
        schema=ResearchPlan,
        temperature=0.2
    )
    
    return response