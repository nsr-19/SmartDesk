"""
General Chat Agent — greetings, small talk, out-of-scope + logging.
"""

from langchain_openai import ChatOpenAI
from config.settings import PRIMARY_MODEL
from state.state import SmartDeskState
from utils.logger import log_agent_start, log_agent_response

_chat_llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=0.5)

GENERAL_PROMPT = """You are SmartDesk AI, the friendly IT & HR support assistant
for NovaTech Solutions.

The employee sent a greeting, thank-you, small talk, or out-of-scope question.

RULES:
• For greetings: "Hi! I'm SmartDesk AI, your IT & HR support assistant at NovaTech.
  I can help with IT issues, HR policies, creating support tickets, or checking
  ticket status. How can I help you today?"
• For thank-yous / goodbyes: "You're welcome! Feel free to reach out anytime."
• For out-of-scope: Politely say you specialise in IT & HR support and redirect.
• Keep responses concise — 2-3 sentences max.
"""


def general_chat_node(state: SmartDeskState) -> dict:
    _start = log_agent_start("General Chat")
    response = _chat_llm.invoke([
        {"role": "system", "content": GENERAL_PROMPT},
        {"role": "user", "content": state.get("user_query", "")},
    ])
    log_agent_response("General Chat", response.content, _start)
    return {
        "agent_results": [
            {"agent": "General Chat", "focus": "conversation", "result": response.content}
        ],
    }
