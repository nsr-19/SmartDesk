"""
Ticket Creation Agent — HITL confirmation flow + logging.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import PRIMARY_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState
from tools.ticket_ops import create_support_ticket
from utils.logger import (
    log_agent_start, log_tool_call, log_tool_result,
    log_agent_response, log_email_captured, log_ticket_created,
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

TICKET_CREATE_PROMPT = """You are the Ticket Creation Assistant for NovaTech Solutions.

MANDATORY WORKFLOW (follow IN ORDER — never skip ahead):

Step 1 — CONFIRM THERE IS AN ACTUAL ISSUE TO ESCALATE:
  Decide: does the employee have a clear, specific, UNRESOLVED problem
  they want a ticket for? Use ONLY the most recent turns and the user's
  current message — do NOT assume a ticket is about a prior topic that
  was already answered.

  Ask for clarification (and STOP at Step 1) if ANY of the below is true:
  • The user said "create a ticket" / "raise a ticket" with no specific
    issue described in the same or immediately preceding turn.
  • The prior assistant turn already answered a KB question successfully
    and the user has not raised a new problem.
  • The described issue is too vague to act on (e.g. "something's broken").

  In that case respond ONLY with a clarifying question, e.g.:
    "Sure — could you tell me what issue you'd like the ticket to cover?
     A short description of the problem is all I need."
  Do NOT proceed to Step 2 until you have a concrete issue statement.

Step 2 — COLLECT EMAIL:
  If `Employee email` in CURRENT STATE below shows a real address (contains
  "@"), the email is ALREADY KNOWN — proceed and use that exact address.
  Do NOT re-ask, do NOT re-validate, do NOT re-derive it from history.
  Only ask for the email if CURRENT STATE shows "not yet provided".
  (Email is the primary key used to look up tickets later.)

Step 3 — GATHER DETAILS (only after Steps 1 and 2 are satisfied):
  From the issue the employee just described:
  • Title: concise one-line summary (max 80 chars)
  • Description: detail the issue using their own words; do NOT pad it
    with content from earlier resolved topics
  • Category: IT or HR (infer from the described issue)
  • Priority: Low / Medium / High / Critical (infer from urgency cues)

Step 4 — PRESENT SUMMARY & ASK FOR CONFIRMATION:
  Show the employee exactly what the ticket will contain and ask:
  "Shall I go ahead and create this ticket?"
  ⚠️ Do NOT call create_support_ticket until the employee explicitly confirms.

Step 5 — CREATE:
  Only after confirmation, call create_support_ticket with all fields.
  Then report the ticket ID back to the employee.

CURRENT STATE:
Employee email: {email}

CONVERSATION HISTORY (for context only — DO NOT mine it for the ticket
subject unless the user has explicitly tied the ticket request to a
specific unresolved problem from one of those turns):
{history}
"""

_ticket_tools = [create_support_ticket]
_ticket_tools_by_name = {t.name: t for t in _ticket_tools}
_ticket_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=MODEL_TEMPERATURE).bind_tools(_ticket_tools)


def ticket_create_node(state: SmartDeskState) -> dict:
    _start = log_agent_start("Ticket Creator")
    ticket_msgs = state.get("ticket_messages", [])

    # Deterministic email capture: regex the latest user message FIRST so the
    # LLM doesn't have to re-derive the email from a noisy conversation
    # history every turn. If the user typed multiple emails in one message,
    # take the LAST one (typically the corrected/final one).
    captured_email = ""
    user_query = state.get("user_query", "") or ""
    found = _EMAIL_RE.findall(user_query)
    state_email = state.get("employee_email", "") or ""
    if found:
        candidate = found[-1]
        if candidate != state_email:
            captured_email = candidate
            log_email_captured(candidate)

    effective_email = captured_email or state_email or "not yet provided"

    history = "\n".join(
        f"{m.type}: {m.content[:300]}" for m in state.get("messages", [])[-10:]
        if getattr(m, "content", "")
    ) or "(none)"

    messages = [
        SystemMessage(content=TICKET_CREATE_PROMPT.format(
            email=effective_email, history=history,
        ))
    ] + ticket_msgs
    if not ticket_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))

    response = _ticket_llm.invoke(messages)

    tool_email = ""
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "create_support_ticket":
                tool_email = tc["args"].get("employee_email", "")
                if tool_email and tool_email != effective_email:
                    log_email_captured(tool_email)

    if not response.tool_calls and response.content:
        log_agent_response("Ticket Creator", response.content, _start)

    result: dict = {
        "ticket_messages": [response],
        "agent_results": [
            {"agent": "Ticket Creator", "focus": "ticket creation", "result": response.content}
        ] if not response.tool_calls else [],
    }
    # Persist whichever email we have (prefer fresh capture, then tool args).
    final_email = captured_email or tool_email
    if final_email:
        result["employee_email"] = final_email
    return result


def ticket_tool_node(state: SmartDeskState) -> dict:
    ticket_msgs = state.get("ticket_messages", [])
    if not ticket_msgs:
        return {"ticket_messages": []}
    last = ticket_msgs[-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"ticket_messages": []}

    results = []
    for tc in last.tool_calls:
        tool = _ticket_tools_by_name.get(tc["name"])
        if tool:
            log_tool_call(tc["name"], tc["args"])
            obs = tool.invoke(tc["args"])
            try:
                parsed = json.loads(obs)
                tid = parsed.get("ticket", {}).get("ticket_id", "")
                if tid:
                    log_ticket_created(tid)
                preview = f"ticket_id={tid}, status=created"
            except Exception:
                preview = str(obs)[:150]
            log_tool_result(tc["name"], preview)
            results.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
    return {"ticket_messages": results}


def should_continue_ticket(state: SmartDeskState) -> str:
    ticket_msgs = state.get("ticket_messages", [])
    if ticket_msgs and hasattr(ticket_msgs[-1], "tool_calls") and ticket_msgs[-1].tool_calls:
        return "ticket_tool_node"
    return "synthesizer"
