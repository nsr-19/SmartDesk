"""
HR Support Agent — RAG over HR KB with confidence-based escalation + logging.
"""

import json
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import PRIMARY_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState
from tools.kb_search import search_hr_knowledge_base
from utils.logger import log_agent_start, log_tool_call, log_tool_result, log_agent_response

HR_PROMPT = """You are an HR Support Specialist at NovaTech Solutions.
You help employees with HR questions using the internal HR knowledge base.

WORKFLOW:
1. ALWAYS call the search_hr_knowledge_base tool first, using the LAST USER
   MESSAGE you received as the query. The router has already narrowed the
   question to your single subtask — do NOT expand the search to other
   topics that may appear in the conversation history; those are handled
   by sibling agents in parallel.
2. Examine the "confidence" field of the search result:
   • "high" — answer using ONLY the retrieved "answer" text. Cite the article
     ID (e.g., ref: HR-004). Be accurate and concise.
   • "low" or "none" — say you couldn't find a matching policy and offer to
     create a ticket FOR THAT TOPIC.
3. CRITICAL anti-hallucination rules — VIOLATING THESE IS A FAILURE:
   • You may use ONLY text returned by the search tool in this turn.
   • You may NOT use prior knowledge, training data, conversation history,
     or "common HR knowledge" to fill in details.
   • You may NOT cite an article ID (HR-001 etc.) unless that exact ID
     appears in the search results you received this turn.
   • If every search came back "low" or "none", you MUST refuse and offer
     a ticket — do NOT compose an answer.
4. NEVER refuse the entire question because part of it is missing from the KB.
   Answer the parts you found at "high" confidence; escalate only the rest.
5. For sensitive topics (harassment, discrimination, ethics violations), always
   direct to the Ethics Hotline (1-800-555-0100) or ethics@novatech.com.
6. Be empathetic, professional, and concise.
7. End with "Is there anything else I can help with?"

CONVERSATION CONTEXT:
{history}
"""

_hr_tools = [search_hr_knowledge_base]
_hr_tools_by_name = {t.name: t for t in _hr_tools}
_hr_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=MODEL_TEMPERATURE).bind_tools(_hr_tools)


def hr_support_node(state: SmartDeskState) -> dict:
    _start = log_agent_start("HR Support")
    hr_msgs = state.get("hr_messages", [])
    history = "\n".join(
        f"{m.type}: {m.content[:200]}" for m in state.get("messages", [])[-6:]
        if getattr(m, "content", "")
    ) or "(none)"

    messages = [SystemMessage(content=HR_PROMPT.format(history=history))] + hr_msgs
    if not hr_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))
    response = _hr_llm.invoke(messages)

    if not response.tool_calls and response.content:
        log_agent_response("HR Support", response.content, _start)

    return {
        "hr_messages": [response],
        "agent_results": [
            {"agent": "HR Support", "focus": "policies and benefits", "result": response.content}
        ] if not response.tool_calls else [],
    }


def hr_tool_node(state: SmartDeskState) -> dict:
    hr_msgs = state.get("hr_messages", [])
    if not hr_msgs:
        return {"hr_messages": []}
    last = hr_msgs[-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"hr_messages": []}

    results = []
    for tc in last.tool_calls:
        tool = _hr_tools_by_name.get(tc["name"])
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
    return {"hr_messages": results}


def should_continue_hr(state: SmartDeskState) -> str:
    hr_msgs = state.get("hr_messages", [])
    if hr_msgs and hasattr(hr_msgs[-1], "tool_calls") and hr_msgs[-1].tool_calls:
        return "hr_tool_node"
    return "synthesizer"
