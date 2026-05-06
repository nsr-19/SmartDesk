"""
IT Support Agent — RAG over IT KB with confidence-based escalation + logging.
"""

import json
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import PRIMARY_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState
from tools.kb_search import search_it_knowledge_base
from utils.logger import log_agent_start, log_tool_call, log_tool_result, log_agent_response

IT_PROMPT = """You are an IT Support Specialist at NovaTech Solutions.
You help employees resolve technical issues using the internal IT knowledge base.

WORKFLOW:
1. ALWAYS call the search_it_knowledge_base tool first, using the LAST USER
   MESSAGE you received as the query. The router has already narrowed the
   question to your single subtask — do NOT expand the search to other
   topics that may appear in the conversation history; those are handled
   by sibling agents in parallel.
2. Examine the "confidence" field of the search result:
   • "high" — answer using ONLY the retrieved "answer" text. Cite the article
     ID (e.g., ref: IT-003). Be clear, step-by-step, and concise.
   • "low" or "none" — say you couldn't find a matching article and offer to
     create a ticket FOR THAT TOPIC.
3. CRITICAL anti-hallucination rules — VIOLATING THESE IS A FAILURE:
   • You may use ONLY text returned by the search tool in this turn.
   • You may NOT use prior knowledge, training data, conversation history,
     or "common IT knowledge" to fill in details.
   • You may NOT cite an article ID (IT-001 etc.) unless that exact ID
     appears in the search results you received this turn.
   • If every search came back "low" or "none", you MUST refuse and offer
     a ticket — do NOT compose an answer.
4. NEVER refuse the entire question because part of it is missing from the KB.
   Answer the parts you found at "high" confidence; escalate only the rest.
5. Be polite, professional, and empathetic.
6. End with "Is there anything else I can help with?"

CONVERSATION CONTEXT:
{history}
"""

_it_tools = [search_it_knowledge_base]
_it_tools_by_name = {t.name: t for t in _it_tools}
_it_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=MODEL_TEMPERATURE).bind_tools(_it_tools)


def it_support_node(state: SmartDeskState) -> dict:
    _start = log_agent_start("IT Support")
    it_msgs = state.get("it_messages", [])
    history = "\n".join(
        f"{m.type}: {m.content[:200]}" for m in state.get("messages", [])[-6:]
        if getattr(m, "content", "")
    ) or "(none)"

    messages = [SystemMessage(content=IT_PROMPT.format(history=history))] + it_msgs
    if not it_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))
    response = _it_llm.invoke(messages)

    if not response.tool_calls and response.content:
        log_agent_response("IT Support", response.content, _start)

    return {
        "it_messages": [response],
        "agent_results": [
            {"agent": "IT Support", "focus": "technical support", "result": response.content}
        ] if not response.tool_calls else [],
    }


def it_tool_node(state: SmartDeskState) -> dict:
    it_msgs = state.get("it_messages", [])
    if not it_msgs:
        return {"it_messages": []}
    last = it_msgs[-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"it_messages": []}

    results = []
    for tc in last.tool_calls:
        tool = _it_tools_by_name.get(tc["name"])
        if tool:
            log_tool_call(tc["name"], tc["args"])
            obs = tool.invoke(tc["args"])
            try:
                parsed = json.loads(obs)
                conf = parsed.get("confidence", "")
                score = parsed.get("top_score", "")
                n = len(parsed.get("results", []))
                preview = f"confidence={conf}, top_score={score}, results_count={n}"
            except Exception:
                conf, preview = "", str(obs)[:150]
            log_tool_result(tc["name"], preview, conf)
            results.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
    return {"it_messages": results}


def should_continue_it(state: SmartDeskState) -> str:
    it_msgs = state.get("it_messages", [])
    if it_msgs and hasattr(it_msgs[-1], "tool_calls") and it_msgs[-1].tool_calls:
        return "it_tool_node"
    return "synthesizer"
