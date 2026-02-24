from llm import DeepSeekClient
from agents.planner import PlannerAgent


def test_planner_create_scoped_plan(prompt: str):
    llm = DeepSeekClient()
    planner = PlannerAgent(llm)

    plan = planner.create_scoped_plan(prompt)

    print(f"\nQuery Type : {plan.query_type.value}")
    print(f"\nResearch Plan ({len(plan.plan)} steps)")

    print("─" * 60)
    for step in plan.plan:
        print(f"\n  Step {step.step_number}: {step.action}")
        print(f"  {step.description}")

    if plan.clarifying_questions:
        print(f"\nClarifying Questions ({len(plan.clarifying_questions)})")
        print("─" * 60)
        for i, q in enumerate(plan.clarifying_questions, 1):
            print(f"  {i}. {q}")
    else:
        print("\nNo clarifying questions.")

    print()

def test_planner_create_plan(prompt: str):
    llm = DeepSeekClient()
    planner = PlannerAgent(llm)
    
    plan = planner.create_plan(prompt)
    
    # print formatted
    print(f"\nQuery Type : {plan.query_type.value}")
    print(f"\nResearch Plan ({len(plan.plan)} steps)")
    print("─" * 60)
    for step in plan.plan:
        print(f"\n  Step {step.step_number}: {step.action}")
        print(f"  {step.description}")




if __name__ == "__main__":
    TEST_PROMPT = "Research about AegisAI, and what kind of problems they solve!"
    # test_planner_create_scoped_plan(TEST_PROMPT)
    test_planner_create_plan(TEST_PROMPT)