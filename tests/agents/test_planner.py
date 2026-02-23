import pytest
from agents.planner import PlannerAgent
from tests.agents.mock_llm import MockLLMClient

def test_planner_create_plan():
    llm = MockLLMClient()
    planner = PlannerAgent(llm)
    
    plan = planner.create_plan("dummy query", num_plan_steps=2)
    
    assert plan is not None
    assert len(plan.plan) > 0
    assert plan.plan[0].description != ""
    assert plan.query_type != ""
