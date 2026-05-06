"""
RAG search tools for IT and HR knowledge bases.

Each tool returns results WITH relevance scores so the agent
can assess confidence and decide whether to escalate.
"""

import json
from langchain.tools import tool
from pydantic import BaseModel, Field

from config.settings import CONFIDENCE_THRESHOLD, RAG_TOP_K


class KBSearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query.")


@tool(args_schema=KBSearchInput)
def search_it_knowledge_base(query: str) -> str:
    """Search the internal IT knowledge base.

    Use for questions about: passwords, VPN, software, hardware, email,
    Wi-Fi, network, printers, security, MFA, accounts, or remote-work IT setup.

    Returns relevant articles with relevance scores. If the top score
    is below the confidence threshold, the context may be insufficient
    and the query should be escalated.
    """
    from utils.vector_store import get_it_store

    results = get_it_store().search(query, top_k=RAG_TOP_K)

    if not results:
        return json.dumps({
            "status": "no_results",
            "message": "No relevant IT articles found.",
            "confidence": "none",
        })

    top_score = results[0]["relevance_score"]
    confidence = "high" if top_score >= CONFIDENCE_THRESHOLD else "low"

    return json.dumps({
        "status": "results_found",
        "confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_score": top_score,
        "results": results,
    }, indent=2)


@tool(args_schema=KBSearchInput)
def search_hr_knowledge_base(query: str) -> str:
    """Search the internal HR knowledge base.

    Use for questions about: leave, PTO, benefits, insurance, PF/401k,
    payroll, reimbursements, onboarding, offboarding, performance reviews,
    remote work policy, code of conduct, or company holidays.

    Returns relevant articles with relevance scores. If the top score
    is below the confidence threshold, the context may be insufficient
    and the query should be escalated.
    """
    from utils.vector_store import get_hr_store

    results = get_hr_store().search(query, top_k=RAG_TOP_K)

    if not results:
        return json.dumps({
            "status": "no_results",
            "message": "No relevant HR articles found.",
            "confidence": "none",
        })

    top_score = results[0]["relevance_score"]
    confidence = "high" if top_score >= CONFIDENCE_THRESHOLD else "low"

    return json.dumps({
        "status": "results_found",
        "confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_score": top_score,
        "results": results,
    }, indent=2)
