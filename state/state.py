"""
Graph state definitions for SmartDesk AI.

State stores raw data only — prompts are formatted inside nodes.
"""

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain.messages import AnyMessage


def reset_or_add(left: list | None, right) -> list:
    """Append like operator.add, but reset to [] when caller sends None.

    Lets main.py clear per-turn channels (agent_results, per-agent message
    channels) without losing parallel-write semantics during a single turn.
    """
    if right is None:
        return []
    return (left or []) + (right or [])


# ── Orchestrator schemas ───────────────────────────────────────────

class AgentTask(BaseModel):
    """A single task the orchestrator delegates to a worker."""

    agent: Literal[
        "it_support", "hr_support",
        "ticket_create", "ticket_status",
        "general_chat",
    ] = Field(description="Target agent for this task.")

    query: str = Field(description="Focused sub-query for the agent.")

    focus: str = Field(
        default="",
        description="What aspect this agent should focus on.",
    )


class RoutingDecision(BaseModel):
    """LLM-generated routing — supports parallel dispatch."""

    tasks: list[AgentTask] = Field(
        description="Agent tasks to execute (1 for single, 2+ for parallel)."
    )
    requires_synthesis: bool = Field(
        default=False,
        description="True when multiple agents run and results must merge.",
    )


# ── Main graph state ───────────────────────────────────────────────

class SmartDeskState(TypedDict):
    """Shared state accessible by every node."""

    # Current user message
    user_query: str

    # Full conversation history (persisted by checkpointer)
    messages: Annotated[list[AnyMessage], operator.add]

    # Employee email — remembered within the session
    employee_email: str

    # Routing
    tasks: list[AgentTask]
    requires_synthesis: bool

    # Parallel agent outputs — reset each turn, accumulated within a turn
    agent_results: Annotated[list[dict], reset_or_add]

    # Final answer returned to user
    final_answer: str

    # Per-agent message channels — reset each turn, isolated tool-call loops
    it_messages: Annotated[list[AnyMessage], reset_or_add]
    hr_messages: Annotated[list[AnyMessage], reset_or_add]
    ticket_messages: Annotated[list[AnyMessage], reset_or_add]
    status_messages: Annotated[list[AnyMessage], reset_or_add]
