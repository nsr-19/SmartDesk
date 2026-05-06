"""
Graph assembly — wires all agents, tools, and edges into the
compiled LangGraph workflow.

Architecture:

    START
      │
   orchestrator  ──► Send API dispatches to 1+ agents in parallel
      │
      ├──► it_support_node   ⇄  it_tool_node
      ├──► hr_support_node   ⇄  hr_tool_node
      ├──► ticket_create_node ⇄ ticket_tool_node
      ├──► ticket_status_node ⇄ status_tool_node
      └──► general_chat_node
                │
                ▼
           synthesizer  ──►  END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from state.state import SmartDeskState

from agents.orchestrator import classify_query, dispatch_to_agents

from agents.it_support import (
    it_support_node, it_tool_node, should_continue_it,
)
from agents.hr_support import (
    hr_support_node, hr_tool_node, should_continue_hr,
)
from agents.ticket_create import (
    ticket_create_node, ticket_tool_node, should_continue_ticket,
)
from agents.ticket_status import (
    ticket_status_node, status_tool_node, should_continue_status,
)
from agents.general_chat import general_chat_node
from agents.synthesizer import synthesizer_node


def build_graph():
    """Build and compile the SmartDesk AI multi-agent graph."""

    builder = StateGraph(SmartDeskState)

    # ── Nodes ──────────────────────────────────────────────────
    builder.add_node("orchestrator", classify_query)

    builder.add_node("it_support_node", it_support_node)
    builder.add_node("it_tool_node", it_tool_node)

    builder.add_node("hr_support_node", hr_support_node)
    builder.add_node("hr_tool_node", hr_tool_node)

    builder.add_node("ticket_create_node", ticket_create_node)
    builder.add_node("ticket_tool_node", ticket_tool_node)

    builder.add_node("ticket_status_node", ticket_status_node)
    builder.add_node("status_tool_node", status_tool_node)

    builder.add_node("general_chat_node", general_chat_node)

    builder.add_node("synthesizer", synthesizer_node)

    # ── Edges ──────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "orchestrator")

    # Orchestrator → parallel dispatch via Send API
    builder.add_conditional_edges(
        "orchestrator",
        dispatch_to_agents,
        [
            "it_support_node",
            "hr_support_node",
            "ticket_create_node",
            "ticket_status_node",
            "general_chat_node",
        ],
    )

    # IT support tool loop
    builder.add_conditional_edges(
        "it_support_node", should_continue_it,
        ["it_tool_node", "synthesizer"],
    )
    builder.add_edge("it_tool_node", "it_support_node")

    # HR support tool loop
    builder.add_conditional_edges(
        "hr_support_node", should_continue_hr,
        ["hr_tool_node", "synthesizer"],
    )
    builder.add_edge("hr_tool_node", "hr_support_node")

    # Ticket creation tool loop
    builder.add_conditional_edges(
        "ticket_create_node", should_continue_ticket,
        ["ticket_tool_node", "synthesizer"],
    )
    builder.add_edge("ticket_tool_node", "ticket_create_node")

    # Ticket status tool loop
    builder.add_conditional_edges(
        "ticket_status_node", should_continue_status,
        ["status_tool_node", "synthesizer"],
    )
    builder.add_edge("status_tool_node", "ticket_status_node")

    # General chat → synthesizer directly (no tools)
    builder.add_edge("general_chat_node", "synthesizer")

    # Exit
    builder.add_edge("synthesizer", END)

    # ── Compile ────────────────────────────────────────────────
    # Swap InMemorySaver for PostgresSaver in production:
    #   from langgraph.checkpoint.postgres import PostgresSaver
    #   checkpointer = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
