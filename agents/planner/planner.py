from llm import BaseLLMClient
from models import ResearchPlan, ScopedPlan
from prompts import get_plan, get_scoped_plan


class PlannerAgent:
    """
    Agent responsible for analyzing a user's research request and generating a structured execution plan.
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the Planner Agent with an LLM dependency.

        Args:
            llm_client (BaseLLMClient): The LLM client for generating the plan.
        """
        self.llm_client = llm_client

    def create_plan(self, user_query: str, num_plan_steps: int = 5) -> ResearchPlan:
        """
        Generates a research plan based on the user's goal.

        Args:
            user_query (str): The high-level research topic or question.

        Returns:
            ResearchPlan: A structured plan containing query type and steps.
        """
        # In the future, we might query the vector store here for existing context
        # before generating the plan. For now, we rely on the LLM's intrinsic knowledge.
        
        plan = get_plan(user_query, self.llm_client, num_plan_steps=num_plan_steps)
        return plan

    def create_scoped_plan(self, user_query: str, num_plan_steps: int = 5) -> ScopedPlan:
        """
        Generates a research plan plus clarifying questions for the user.
        Use this to scope the research before running the full pipeline.
        """
        return get_scoped_plan(user_query, self.llm_client, num_plan_steps=num_plan_steps)
