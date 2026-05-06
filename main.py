"""
SmartDesk AI — Interactive CLI with full logging.

Run:  python main.py
"""

import uuid
from langchain.messages import HumanMessage

from graph.workflow import build_graph
from tools.ticket_ops import get_all_tickets, seed_demo_tickets
from utils.logger import log_user_input, log_final_answer, log_error, C


def main():
    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║          SmartDesk AI — NovaTech IT & HR Support        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"  {C.DIM}Commands:{C.RESET}")
    print(f"    Type your question to chat with the agent.")
    print(f"    {C.BOLD}'tickets'{C.RESET}  — View all created tickets.")
    print(f"    {C.BOLD}'new'{C.RESET}      — Start a new conversation session.")
    print(f"    {C.BOLD}'quit'{C.RESET}     — Exit.")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")

    # Seed demo tickets for testing ticket-status flow
    seed_demo_tickets()
    print(f"  {C.DIM}ℹ️  Demo tickets seeded (jane.doe@novatech.com, bob.smith@novatech.com){C.RESET}")

    print(f"\n  ⏳ Loading knowledge bases and building agent graph...")
    graph = build_graph()
    print(f"  {C.GREEN}✅ Ready! ({C.BOLD}44 KB articles indexed{C.RESET}{C.GREEN}){C.RESET}\n")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    employee_email = ""

    while True:
        try:
            user_input = input(f"{C.BOLD}👤 You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{C.CYAN}👋 Goodbye!{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(f"\n{C.CYAN}👋 Goodbye!{C.RESET}")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            employee_email = ""
            print(f"\n{C.GREEN}🔄 New session started.{C.RESET}\n")
            continue

        if user_input.lower() == "tickets":
            tickets = get_all_tickets()
            if not tickets:
                print(f"\n  {C.DIM}📋 No tickets exist yet.{C.RESET}\n")
            else:
                print(f"\n  {C.BOLD}📋 All Tickets ({len(tickets)}):{C.RESET}")
                for t in tickets:
                    status_color = {
                        "Open": C.YELLOW, "In Progress": C.CYAN,
                        "Resolved": C.GREEN, "Closed": C.DIM,
                    }.get(t["status"], C.WHITE)
                    print(
                        f"    [{C.BOLD}{t['ticket_id']}{C.RESET}] "
                        f"{t['priority']:8s} | {t['category']:3s} | "
                        f"{status_color}{t['status']:12s}{C.RESET} | "
                        f"{t['employee_email']:30s} | {t['title']}"
                    )
                print()
            continue

        # ── Log input & invoke graph ───────────────────────────
        log_user_input(user_input)

        try:
            result = graph.invoke(
                {
                    "user_query": user_input,
                    "messages": [HumanMessage(content=user_input)],
                    "employee_email": employee_email,
                    # Reset per-turn channels so previous turn's results
                    # don't bleed into this turn's synthesizer.
                    "agent_results": None,
                    "it_messages": None,
                    "hr_messages": None,
                    "ticket_messages": None,
                    "status_messages": None,
                },
                config=config,
            )

            # Print final answer
            answer = result.get("final_answer", "I'm sorry, I couldn't generate a response.")
            log_final_answer()
            print(f"\n{C.BOLD}🤖 SmartDesk:{C.RESET}\n{answer}\n")

            # Persist email across turns
            if result.get("employee_email"):
                employee_email = result["employee_email"]

            # Show routing summary
            tasks = result.get("tasks", [])
            if tasks:
                agents_used = ", ".join(
                    f"{C.BOLD}{t.agent}{C.RESET}" for t in tasks
                )
                print(f"  {C.DIM}ℹ️  Routed to: {agents_used}{C.RESET}\n")

        except Exception as e:
            log_error(str(e))
            print(f"  {C.DIM}Please try again or rephrase your question.{C.RESET}\n")


if __name__ == "__main__":
    main()
