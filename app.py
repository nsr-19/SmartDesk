"""
SmartDesk AI — Streamlit UI.

Multi-conversation chat interface for the LangGraph agent. Use the sidebar
to create, switch between, or delete chats. Each chat keeps its own thread
in the LangGraph checkpointer, so context (employee email, ticket flow,
ongoing topic) is isolated per conversation.

Run:
    streamlit run app.py

The agent's per-step logs (routing, tool calls, confidence, agent timings)
are printed to the terminal where you launched `streamlit run` — leave that
window visible alongside the browser to watch the graph execute live.
"""

import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain.messages import HumanMessage

# Make project modules importable regardless of where streamlit is launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from graph.workflow import build_graph
from tools.ticket_ops import get_all_tickets, seed_demo_tickets


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartDesk AI — NovaTech",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 920px; }
      [data-testid="stChatMessage"] {
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 0.4rem;
      }
      [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        font-weight: 400;
      }
      .ticket-card {
        background: rgba(127,127,127,0.08);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        line-height: 1.35;
      }
      .small-meta { font-size: 0.78rem; opacity: 0.65; }
      h1 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached graph (one compile shared across browser sessions) ────────

@st.cache_resource(show_spinner="⏳ Loading knowledge bases and building agent graph…")
def get_graph():
    seed_demo_tickets()
    return build_graph()


graph = get_graph()


# ── Session state ────────────────────────────────────────────────────

def _new_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.conversations[chat_id] = {
        "title": "New chat",
        "messages": [],          # [{"role": "user"|"assistant", "content": str}]
        "thread_id": chat_id,    # passed to LangGraph checkpointer
        "email": "",             # remembered across turns within this chat
        "turns": [],             # per-assistant-turn routing metadata
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    _new_chat()


def _current() -> dict:
    return st.session_state.conversations[st.session_state.current_chat_id]


# ── Sidebar: chat switcher + ticket viewer ───────────────────────────

with st.sidebar:
    st.markdown("### 🤖 SmartDesk AI")
    st.caption("NovaTech IT & HR Support")
    st.divider()

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.markdown("##### Conversations")

    for chat_id, chat in reversed(list(st.session_state.conversations.items())):
        is_current = chat_id == st.session_state.current_chat_id
        cols = st.columns([6, 1])
        with cols[0]:
            label = chat["title"] or "New chat"
            prefix = "🟢 " if is_current else "💬 "
            if st.button(
                f"{prefix}{label[:30]}",
                key=f"sel-{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with cols[1]:
            if len(st.session_state.conversations) > 1:
                if st.button("✕", key=f"del-{chat_id}", help="Delete this chat"):
                    del st.session_state.conversations[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = next(
                            iter(st.session_state.conversations)
                        )
                    st.rerun()

    st.divider()

    with st.expander("📋 All tickets", expanded=False):
        tickets = get_all_tickets()
        if not tickets:
            st.caption("No tickets yet.")
        else:
            status_icon = {
                "Open": "🟡", "In Progress": "🔵",
                "Resolved": "🟢", "Closed": "⚫",
            }
            for t in tickets:
                icon = status_icon.get(t["status"], "⚪")
                st.markdown(
                    f"<div class='ticket-card'>{icon} <b>{t['ticket_id']}</b> "
                    f"<i>({t['status']})</i><br/>{t['title']}<br/>"
                    f"<span class='small-meta'>{t['employee_email']}</span></div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(
        "💡 Routing, tool calls, and agent timings stream live to the terminal "
        "where you ran `streamlit run app.py`."
    )


# ── Main pane ────────────────────────────────────────────────────────

current = _current()

st.title("How can I help today?")
st.caption(
    "Ask about IT (passwords, VPN, hardware…) or HR (leave, payroll, benefits…) "
    "— or raise/track a support ticket. Cross-domain questions run IT and HR "
    "agents in parallel."
)


def _render_routing(meta: dict) -> None:
    if not meta or not meta.get("agents"):
        return
    with st.expander("🔍 Routing details", expanded=False):
        st.markdown(f"**Agents used:** {', '.join(meta['agents'])}")
        for t in meta.get("tasks", []):
            st.markdown(f"- `{t['agent']}` — {t['focus'] or t['query']}")


# Replay history
assistant_turn = 0
for msg in current["messages"]:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_turn < len(current["turns"]):
                _render_routing(current["turns"][assistant_turn])
            assistant_turn += 1


# Input
prompt = st.chat_input("Type your question…")

if prompt:
    current["messages"].append({"role": "user", "content": prompt})
    if current["title"] == "New chat":
        current["title"] = prompt[:40]

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke(
                    {
                        "user_query": prompt,
                        "messages": [HumanMessage(content=prompt)],
                        "employee_email": current["email"],
                        # Reset per-turn channels — see state/state.py reset_or_add
                        "agent_results": None,
                        "it_messages": None,
                        "hr_messages": None,
                        "ticket_messages": None,
                        "status_messages": None,
                    },
                    config={"configurable": {"thread_id": current["thread_id"]}},
                )
                answer = result.get(
                    "final_answer",
                    "I'm sorry, I couldn't generate a response. Please try again.",
                )
                if result.get("employee_email"):
                    current["email"] = result["employee_email"]

                tasks = result.get("tasks", []) or []
                meta = {
                    "agents": sorted({
                        (t.agent if hasattr(t, "agent") else t.get("agent", "?"))
                        for t in tasks
                    }),
                    "tasks": [
                        {
                            "agent": t.agent if hasattr(t, "agent") else t.get("agent", "?"),
                            "focus": t.focus if hasattr(t, "focus") else t.get("focus", ""),
                            "query": t.query if hasattr(t, "query") else t.get("query", ""),
                        }
                        for t in tasks
                    ],
                }
            except Exception as e:
                answer = f"⚠️ Something went wrong: `{e}`"
                meta = {"agents": [], "tasks": []}

        st.markdown(answer)
        _render_routing(meta)

    current["messages"].append({"role": "assistant", "content": answer})
    current["turns"].append(meta)
    st.rerun()
