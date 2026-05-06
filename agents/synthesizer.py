"""
Synthesizer — merges parallel agent results + logging.
"""

from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage

from config.settings import PRIMARY_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState
from utils.logger import log_synthesis, log_final_answer

_synth_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=MODEL_TEMPERATURE)

SYNTHESIS_PROMPT = """You are the Response Synthesizer for SmartDesk AI.
Combine the following agent outputs into ONE clear, well-organised response.

RULES:
1. Organise logically (e.g., Answer → Ticket Info → Next Steps).
2. Do NOT repeat information.
3. Friendly, professional tone.
4. ID handling — these two are NOT the same and must never be conflated:
   • Knowledge-base article IDs look like HR-001, HR-014, IT-003, IT-013.
     Cite them as "(ref: HR-003)" or "(ref: IT-013)" — NEVER as "Ticket ID".
   • Support ticket IDs look like TKT-A1B2C3 (TKT- prefix + 6 hex chars).
     Only THESE get labelled as "Ticket ID" and highlighted prominently.
   If the agents only cited KB articles, do not invent or mention any ticket ID.
5. End with "Is there anything else I can help with?"

Original Query: {query}

Agent Outputs:
{results}

Write a unified response:"""


def synthesizer_node(state: SmartDeskState) -> dict:
    agent_results = state.get("agent_results", [])
    log_synthesis(len(agent_results))

    if not agent_results:
        fallback = (
            "I'm sorry, I wasn't able to process your request. "
            "Please try again or contact the IT Help Desk at ext. 2020."
        )
        return {"final_answer": fallback, "messages": [AIMessage(content=fallback)]}

    if len(agent_results) == 1:
        result = agent_results[0]["result"]
        log_final_answer()
        return {"final_answer": result, "messages": [AIMessage(content=result)]}

    formatted = "\n\n".join(
        f"--- {r['agent'].upper()} ({r.get('focus', '')}) ---\n{r['result']}"
        for r in agent_results
    )
    response = _synth_llm.invoke([
        HumanMessage(content=SYNTHESIS_PROMPT.format(
            query=state["user_query"], results=formatted,
        ))
    ])
    log_final_answer()
    return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}
