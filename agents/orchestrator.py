"""
Orchestrator — LLM-driven intent classification and parallel dispatch.
"""

import time
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langgraph.types import Send

from config.settings import ROUTER_MODEL, MODEL_TEMPERATURE
from state.state import SmartDeskState, RoutingDecision
from utils.logger import log_routing

_router_llm = ChatOpenAI(
    model=ROUTER_MODEL,
    temperature=0,  # routing must be deterministic
).with_structured_output(RoutingDecision)

ROUTING_PROMPT = """You are the intent classifier for SmartDesk AI, an internal IT & HR help desk.

Analyse the employee's LATEST message in the context of the conversation history
and decide which agent(s) should handle it.

AVAILABLE AGENTS:
1. it_support     — IT questions: passwords, VPN, software, hardware, email, Wi-Fi,
                     printers, MFA, accounts, remote-work IT setup, AND security
                     topics owned by IT: data classification & handling, data
                     protection, phishing, endpoint security, encryption.
2. hr_support     — HR questions: leave/PTO, benefits, insurance, PF/401k, payroll,
                     reimbursements, onboarding, offboarding, performance reviews,
                     WFH policy, code of conduct, holidays, ESOP, anti-harassment.
3. ticket_create  — Create a support ticket. Use when:
                     • The employee explicitly asks to create/raise/open/log a ticket.
                     • A previous agent response suggested escalation and the employee agreed.
                     • The employee describes an issue that clearly cannot be answered from a KB
                       (e.g., broken hardware, account-specific problems).
4. ticket_status  — Check ticket status. Use when the employee asks about the status,
                     updates, or progress of a previously raised ticket.
5. general_chat   — Greetings, thank-yous, small talk, or questions completely outside
                     IT/HR scope (e.g., "What's the weather?", "Tell me a joke").

ROUTING RULES:

A) SPLIT into MULTIPLE AGENTS (requires_synthesis = true) whenever the message
   contains two or more distinct topics that map to different agents.
   Watch for connectors: "and", "&", "also", "plus", commas, "what about".
   Each topic gets its OWN AgentTask, with a query containing ONLY that topic.

   CRITICAL — verify each task before you emit it:
     1. Read your task.agent.
     2. Read your task.query.
     3. Ask yourself: "Does this query topic match this agent's domain?"
        If NO, you have a swap — fix it before returning.

B) SINGLE AGENT (requires_synthesis = false) — when the entire message stays
   within ONE domain, even if it asks several related sub-questions
   (e.g., "How do casual leaves and sick leaves work?" → hr_support only).

C) Continuation rules:
   • Employee is continuing a ticket-creation conversation (providing email,
     confirming details) → ticket_create.
   • Employee is continuing a ticket-status conversation (providing email to
     look up tickets) → ticket_status.

D) STRICT DOMAIN MAPPING (use this table — do NOT route by intuition):

   TOPIC                                              → AGENT
   ─────────────────────────────────────────────────────────────────
   leave / PTO / vacation / casual / sick / EL / WFH  → hr_support
   benefits / insurance / 401k / PF / payroll / ESOP  → hr_support
   onboarding / offboarding / probation / performance → hr_support
   reimbursement / holidays / wellness / EAP          → hr_support
   anti-harassment / code of conduct                  → hr_support
   ─────────────────────────────────────────────────────────────────
   password / VPN / Wi-Fi / printer / email / MFA     → it_support
   software install / laptop / hardware / equipment   → it_support
   data handling / data classification / data privacy → it_support
   PII / encryption / phishing / endpoint security    → it_support
   account lockout / SSO / certificates               → it_support

E) WORKED EXAMPLES (study the agent ↔ query pairing carefully):

   Input: "what is the leave & data handling policy of organisation?"
   Correct output:
     tasks = [
       AgentTask(agent="hr_support", query="What is the leave policy?",
                 focus="leave policy"),
       AgentTask(agent="it_support",
                 query="What is the data handling / classification policy?",
                 focus="data handling / security policy"),
     ]
     requires_synthesis = True

   Input: "How many leaves do I get and how do I set up VPN?"
   Correct output:
     tasks = [
       AgentTask(agent="hr_support", query="How many leaves do I get?",
                 focus="leave entitlement"),
       AgentTask(agent="it_support", query="How do I set up VPN?",
                 focus="VPN setup"),
     ]
     requires_synthesis = True

   Input: "Tell me about WFH policy & laptop request"
   Correct output:
     tasks = [
       AgentTask(agent="hr_support", query="What is the WFH policy?",
                 focus="WFH policy"),
       AgentTask(agent="it_support", query="How do I request a laptop?",
                 focus="laptop request"),
     ]
     requires_synthesis = True

CONVERSATION HISTORY (for context):
{history}

EMPLOYEE EMAIL (if already known): {email}
"""


def classify_query(state: SmartDeskState) -> dict:
    history_lines = []
    for msg in state.get("messages", [])[-10:]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if content:
            history_lines.append(f"{role}: {content[:300]}")
    history_str = "\n".join(history_lines) if history_lines else "(no prior messages)"
    email = state.get("employee_email", "not yet provided")

    t0 = time.time()
    decision: RoutingDecision = _router_llm.invoke([
        {"role": "system", "content": ROUTING_PROMPT.format(history=history_str, email=email)},
        {"role": "user", "content": state["user_query"]},
    ])
    elapsed = f"{(time.time() - t0) * 1000:.0f}ms"
    log_routing(decision.tasks, decision.requires_synthesis, elapsed)

    return {"tasks": decision.tasks, "requires_synthesis": decision.requires_synthesis}


def dispatch_to_agents(state: SmartDeskState) -> list[Send]:
    sends = []
    for task in state.get("tasks", []):
        worker_state: dict = {
            "messages": state.get("messages", []),
            "user_query": task.query,
            "employee_email": state.get("employee_email", ""),
            "tasks": state.get("tasks", []),
            "requires_synthesis": state.get("requires_synthesis", False),
            "agent_results": [],
            "final_answer": "",
            "it_messages": [],
            "hr_messages": [],
            "ticket_messages": [],
            "status_messages": [],
        }
        sends.append(Send(f"{task.agent}_node", worker_state))
    return sends
