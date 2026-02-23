from typing import List
from pydantic import BaseModel, Field
from models.models import Gap, GapAnalysis

def get_gaps(original_goal: str, current_topic: str, findings: str, llm_client, max_probes: int = 3) -> GapAnalysis:
    """
    Identifies knowledge gaps with severity scores.

    Args:
        original_goal: The top-level research objective.
        current_topic: The topic that was just investigated.
        findings: The findings from investigating the current topic.
        llm_client: The initialized LLM client.
        max_probes: Maximum number of gaps to generate.

    Returns:
        GapAnalysis: List of gaps with severity scores and a completion flag.
    """
    system_prompt = """You are a Research Gap Analyzer. Given the original research goal and current findings, identify what knowledge is still missing.

RULES:
1. Generate up to {max_probes} gaps maximum.
2. Each gap must have a specific, actionable research question.
3. Rate each gap's severity from 1-10:
   - 1-3: Nice-to-have, tangential detail
   - 4-6: Moderately important, adds depth
   - 7-10: Critical missing information, core to the research goal
4. If the original goal is already fully covered, set is_complete to true and return an empty gaps list.
5. Do NOT generate gaps that repeat what has already been found."""

    user_prompt = """Original Research Goal: {original_goal}

Topic Just Investigated: {current_topic}

Current Findings:
{findings}

Identify the remaining knowledge gaps and rate their severity."""

    return llm_client.generate_structured(
        prompt=user_prompt.format(original_goal=original_goal, current_topic=current_topic, findings=findings),
        system_prompt=system_prompt.format(max_probes=max_probes),
        schema=GapAnalysis,
        temperature=0.2,
    )
