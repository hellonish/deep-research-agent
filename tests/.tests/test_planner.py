import pytest
import os
from unittest.mock import MagicMock
from prompts import planner
from models import ResearchPlan, PlanStep, QueryType
from llm import BaseLLMClient
from agents.planner.planner import PlannerAgent
from vector_store.base import BaseVectorStore

class TestPlanner:
    def test_planner_calls_llm_correctly(self):
        # Arrange
        user_prompt = "Research the impact of quantum computing on cryptography"
        mock_llm_client = MagicMock(spec=BaseLLMClient)
        
        expected_plan = ResearchPlan(
            query_type=QueryType.SOFTWARE,
            plan=[
                PlanStep(
                    step_number=1, 
                    action="Research Quantum Basics", 
                    description="Search for fundamental principles of quantum computing including qubits and superposition."
                ),
                PlanStep(
                    step_number=2, 
                    action="Analyze Crypto Impact", 
                    description="Investigate how Shor's algorithm affects RSA and ECC encryption schemes."
                ),
                PlanStep(
                    step_number=3, 
                    action="Synthesize Findings", 
                    description="Summarize the timeline for quantum threats and potential post-quantum solutions."
                )
            ]
        )
        mock_llm_client.generate_structured.return_value = expected_plan

        # Act
        result = planner(user_prompt, mock_llm_client)
        
        print("\n" + "="*50)
        print(f"INPUT PROMPT: {user_prompt}")
        print("-" * 20)
        print(f"OUTPUT PLAN:\n{result.model_dump_json(indent=2)}")
        print("="*50 + "\n")

        # Assert
        assert result == expected_plan
        mock_llm_client.generate_structured.assert_called_once()
        
        # Verify arguments
        call_args = mock_llm_client.generate_structured.call_args
        assert call_args is not None
        _, kwargs = call_args
        
        assert "prompt" in kwargs
        assert user_prompt in kwargs["prompt"]
        assert "system_prompt" in kwargs
        assert "expert Research Planner Agent" in kwargs["system_prompt"]
        assert kwargs["schema"] == ResearchPlan
        assert kwargs["temperature"] == 0.2

    def test_planner_returns_structured_plan(self):
        # Arrange
        user_prompt = "How to bake a cake"
        mock_llm_client = MagicMock(spec=BaseLLMClient)
        
        expected_plan = ResearchPlan(
            query_type=QueryType.GENERAL,
            plan=[
                PlanStep(step_number=1, action="Find Recipe", description="Search for a simple cake recipe."),
            ]
        )
        mock_llm_client.generate_structured.return_value = expected_plan

        # Act
        result = planner(user_prompt, mock_llm_client)

        print("\n" + "="*50)
        print(f"INPUT PROMPT: {user_prompt}")
        print("-" * 20)
        print(f"OUTPUT PLAN:\n{result.model_dump_json(indent=2)}")
        print("="*50 + "\n")

        # Assert
        assert isinstance(result, ResearchPlan)
        assert result.query_type == QueryType.GENERAL
        assert len(result.plan) == 1
        assert result.plan[0].action == "Find Recipe"

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not found")
    def test_planner_integration_real_llm(self):
        """
        Integration test that calls the REAL Gemini API to verify actual output format and quality.
        Requires GOOGLE_API_KEY in environment.
        """
        from llm.gemini import GeminiClient
        
        # Arrange
        client = GeminiClient()
        user_prompt = "Research the impact of quantum computing on cryptography"
        
        print("\n" + "="*50)
        print(f"REAL LLM EXECUTION - Input: {user_prompt}")
        print("-" * 20)
        
        # Act
        result = planner(user_prompt, client)
        
        print(f"OUTPUT PLAN (Query Type: {result.query_type}):")
        print("="*50)
        for step in result.plan:
            print(f"Step {step.step_number}: {step.action} (Summary)")
            print(f"  Detail: {step.description}")
            print("-" * 20)
        print("="*50 + "\n")

        # Assert
        assert isinstance(result, ResearchPlan)
        assert 3 <= len(result.plan) <= 7, f"Plan has {len(result.plan)} steps, expected 3-7"
        assert result.query_type in [QueryType.SOFTWARE, QueryType.MATHEMATICS, QueryType.GENERAL]


class TestPlannerAgent:
    def test_create_plan_delegates_to_planner(self):
        # Arrange
        mock_llm = MagicMock(spec=BaseLLMClient)
        mock_vector_store = MagicMock(spec=BaseVectorStore)
        agent = PlannerAgent(mock_llm, mock_vector_store)
        
        expected_plan = ResearchPlan(query_type=QueryType.GENERAL, plan=[])
        mock_llm.generate_structured.return_value = expected_plan
        
        # Act
        result = agent.create_plan("Test goal")
        
        # Assert
        assert result == expected_plan
        # Verify vector store was stored correctly
        assert agent.vector_store == mock_vector_store
