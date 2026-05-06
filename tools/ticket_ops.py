"""
Mock ticket operations — create and look up support tickets.

Replace _tickets list with a real DB or API (Jira, Asana, Notion)
in production.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field


# ── In-memory ticket store ─────────────────────────────────────────
_tickets: list[dict] = []


# ── Create ─────────────────────────────────────────────────────────

class CreateTicketInput(BaseModel):
    title: str = Field(..., description="Short summary of the issue (max 80 chars).")
    description: str = Field(..., description="Detailed description of the issue.")
    category: str = Field(..., description="Category: IT or HR.")
    priority: str = Field(default="Medium", description="Low / Medium / High / Critical.")
    employee_email: str = Field(..., description="Email of the employee raising the ticket.")


@tool(args_schema=CreateTicketInput)
def create_support_ticket(
    title: str,
    description: str,
    category: str,
    priority: str = "Medium",
    employee_email: str = "",
) -> str:
    """Create a support ticket for an issue that cannot be resolved from the KB.

    IMPORTANT: Only call this tool AFTER:
    1. The employee's email has been collected.
    2. A ticket summary has been presented to the employee.
    3. The employee has explicitly confirmed they want to proceed.
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"

    ticket = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "category": category.upper(),
        "priority": priority,
        "employee_email": employee_email.lower().strip(),
        "status": "Open",
        "comments": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _tickets.append(ticket)

    return json.dumps({
        "message": "Ticket created successfully.",
        "ticket": ticket,
    }, indent=2)


# ── Status lookup ──────────────────────────────────────────────────

class LookupTicketsInput(BaseModel):
    employee_email: str = Field(
        ..., description="Email address to look up tickets for."
    )


@tool(args_schema=LookupTicketsInput)
def lookup_tickets_by_email(employee_email: str) -> str:
    """Retrieve all support tickets associated with an employee email.

    Returns a list of tickets with their current status.
    """
    email = employee_email.lower().strip()
    matches = [t for t in _tickets if t["employee_email"] == email]

    if not matches:
        return json.dumps({
            "status": "no_tickets",
            "message": f"No tickets found for {email}.",
        })

    # Return a summary of each ticket
    summaries = []
    for t in matches:
        summaries.append({
            "ticket_id": t["ticket_id"],
            "title": t["title"],
            "category": t["category"],
            "priority": t["priority"],
            "status": t["status"],
            "created_at": t["created_at"],
            "comments": t["comments"],
        })

    return json.dumps({
        "status": "tickets_found",
        "count": len(summaries),
        "tickets": summaries,
    }, indent=2)


# ── Helpers (for CLI / testing) ────────────────────────────────────

def get_all_tickets() -> list[dict]:
    """Return all tickets (for demo inspection)."""
    return list(_tickets)


def seed_demo_tickets() -> None:
    """Pre-populate a few tickets so ticket-status flow is testable."""
    demo = [
        {
            "ticket_id": "TKT-DEMO01",
            "title": "Monitor flickering intermittently",
            "description": "Employee reports 27-inch Dell monitor flickers every few minutes, especially after waking from sleep.",
            "category": "IT",
            "priority": "Medium",
            "employee_email": "jane.doe@novatech.com",
            "status": "In Progress",
            "comments": ["IT team: Replacement monitor ordered, expected delivery by Thursday."],
            "created_at": "2026-04-28T10:30:00+00:00",
        },
        {
            "ticket_id": "TKT-DEMO02",
            "title": "Reimbursement claim pending for 3 weeks",
            "description": "Travel reimbursement submitted on April 5 still shows 'Pending Approval' in Workday.",
            "category": "HR",
            "priority": "High",
            "employee_email": "jane.doe@novatech.com",
            "status": "Open",
            "comments": [],
            "created_at": "2026-04-25T14:00:00+00:00",
        },
        {
            "ticket_id": "TKT-DEMO03",
            "title": "VPN disconnects every 30 minutes",
            "description": "Cisco AnyConnect VPN drops every ~30 minutes when on home Wi-Fi. Tried flushing DNS and switching to backup gateway.",
            "category": "IT",
            "priority": "High",
            "employee_email": "bob.smith@novatech.com",
            "status": "Resolved",
            "comments": [
                "IT team: Identified MTU mismatch with employee's ISP.",
                "Fix applied: Set MTU to 1400 in AnyConnect profile. Employee confirmed stable connection.",
            ],
            "created_at": "2026-04-20T09:15:00+00:00",
        },
    ]
    _tickets.extend(demo)
