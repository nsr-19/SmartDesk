"""
Ticket Status Agent — lookup by email + logging.
"""

import json
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import PRIMARY_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState
from tools.ticket_ops import lookup_tickets_by_email
from utils.logger import (
    log_agent_start, log_tool_call, log_tool_result,
    log_agent_response, log_email_captured,
)

TICKET_STATUS_PROMPT = """You are the Ticket Status Assistant for NovaTech Solutions.

WORKFLOW:
1. If the employee's email is NOT known, ask for it politely.
2. Once you have the email, call lookup_tickets_by_email.
3. Present results clearly:
   • Multiple tickets: list them briefly (ID, title, status) and ask which
     one they'd like details on.
   • One ticket: show full details including status and any team comments.
   • No tickets: inform the employee politely and offer to help with something else.
4. Include ticket IDs prominently.

CURRENT STATE:
Employee email: {email}

CONVERSATION HISTORY:
{history}
"""

_status_tools = [lookup_tickets_by_email]
_status_tools_by_name = {t.name: t for t in _status_tools}
_status_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=MODEL_TEMPERATURE).bind_tools(_status_tools)


def ticket_status_node(state: SmartDeskState) -> dict:
    _start = log_agent_start("Ticket Status")
    status_msgs = state.get("status_messages", [])
    email = state.get("employee_email", "") or "not yet provided"
    history = "\n".join(
        f"{m.type}: {m.content[:200]}" for m in state.get("messages", [])[-8:]
        if getattr(m, "content", "")
    ) or "(none)"

    messages = [
        SystemMessage(content=TICKET_STATUS_PROMPT.format(email=email, history=history))
    ] + status_msgs
    if not status_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))

    response = _status_llm.invoke(messages)

    new_email = ""
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "lookup_tickets_by_email":
                new_email = tc["args"].get("employee_email", "")
                if new_email:
                    log_email_captured(new_email)

    if not response.tool_calls and response.content:
        log_agent_response("Ticket Status", response.content, _start)

    result: dict = {
        "status_messages": [response],
        "agent_results": [
            {"agent": "Ticket Status", "focus": "ticket lookup", "result": response.content}
        ] if not response.tool_calls else [],
    }
    if new_email:
        result["employee_email"] = new_email
    return result


def status_tool_node(state: SmartDeskState) -> dict:
    status_msgs = state.get("status_messages", [])
    if not status_msgs:
        return {"status_messages": []}
    last = status_msgs[-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"status_messages": []}

    results = []
    for tc in last.tool_calls:
        tool = _status_tools_by_name.get(tc["name"])
        if tool:
            log_tool_call(tc["name"], tc["args"])
            obs = tool.invoke(tc["args"])
            try:
                parsed = json.loads(obs)
                count = parsed.get("count", 0)
                status = parsed.get("status", "")
                preview = f"status={status}, tickets_found={count}"
            except Exception:
                preview = str(obs)[:150]
            log_tool_result(tc["name"], preview)
            results.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
    return {"status_messages": results}


def should_continue_status(state: SmartDeskState) -> str:
    status_msgs = state.get("status_messages", [])
    if status_msgs and hasattr(status_msgs[-1], "tool_calls") and status_msgs[-1].tool_calls:
        return "status_tool_node"
    return "synthesizer"
