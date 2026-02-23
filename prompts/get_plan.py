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
        research request and break it down into a highly methodical, actionable plan.\n\n
        Instructions:\n
        1. Categorize the request into the most appropriate domain provided in the schema.\n
        2. Create a logical, step-by-step plan to execute the research.\n
        3. Break the plan down into exactly {num_plan_steps} distinct steps.\n
        4. For each step:
           - 'action': A short, simple summary for the user (e.g., "Research Quantum Basics").
           - 'description': The detailed, technical instruction for the agent to execute.
        """.format(num_plan_steps=num_plan_steps)
    )
    response = llm_client.generate_structured(
        prompt=f"Create a research plan for the following request: {user_prompt}",
        system_prompt=system_prompt,
        schema=ResearchPlan,
        temperature=0.2
    )
    return response