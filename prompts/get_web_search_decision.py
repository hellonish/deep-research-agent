"""
Prompt for the chat router: LLM decides whether to run a web search and what query to use.
"""
from app.schemas.chat import WebSearchDecision

    
WEB_SEARCH_DECISION_SYSTEM = """
    You decide whether to run a web search to answer the user. 
    Output use_web=true only when the user needs current/live information (news, recent events, real-time data, or facts you might not have). 
    Output use_web=false for greetings, clarifications, opinions, or when the user is asking about the research report they already have. 
    When use_web=true, set search_query to a short, effective search query (2-8 words) that will find relevant results; do not copy the whole user message.
    """



def get_web_search_decision(user_message: str, has_research_context: bool, llm_client) -> WebSearchDecision:
    """
    Ask the LLM whether to run a web search and what query to use.

    Returns:
        WebSearchDecision with use_web and optional search_query.
    """
    prompt = (
        f"User message: {user_message}\n\n"
        + (
            "Context: This session has a prior research report (use it to decide if fresh web search is still useful).\n"
            if has_research_context
            else "No prior research report in this session.\n"
        )
    )
    return llm_client.generate_structured(
        prompt=prompt,
        system_prompt=WEB_SEARCH_DECISION_SYSTEM,
        schema=WebSearchDecision,
        temperature=0.1,
    )
