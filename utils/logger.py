"""
SmartDesk AI — Terminal Logger

Provides color-coded, structured logging for every step of the
agent pipeline so you can visualise exactly what's happening.
"""

import time
from functools import wraps
from typing import Any


# ── ANSI colour codes ──────────────────────────────────────────────

class C:
    """ANSI escape codes for terminal colours."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    BG_BLUE    = "\033[44m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_RED     = "\033[41m"
    BG_MAGENTA = "\033[45m"


# ── Formatting helpers ─────────────────────────────────────────────

_DIVIDER = f"{C.DIM}{'─' * 60}{C.RESET}"
_SECTION = f"{C.DIM}{'═' * 60}{C.RESET}"


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"{C.DIM}... ({len(text)} chars total){C.RESET}"


def _elapsed(start: float) -> str:
    ms = (time.time() - start) * 1000
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ── Public logging functions ───────────────────────────────────────

def log_user_input(query: str) -> None:
    print(f"\n{_SECTION}")
    print(f"{C.BOLD}{C.BLUE}📥 USER INPUT{C.RESET}")
    print(f"{_DIVIDER}")
    print(f"   {C.WHITE}{query}{C.RESET}")
    print(f"{_SECTION}")


def log_routing(tasks: list, requires_synthesis: bool, elapsed: str = "") -> None:
    print(f"\n{C.BOLD}{C.CYAN}🧭 ORCHESTRATOR — Routing Decision{C.RESET} {C.DIM}{elapsed}{C.RESET}")
    print(f"{_DIVIDER}")

    for i, task in enumerate(tasks, 1):
        agent = task.agent if hasattr(task, 'agent') else task.get('agent', '?')
        query = task.query if hasattr(task, 'query') else task.get('query', '?')
        focus = task.focus if hasattr(task, 'focus') else task.get('focus', '')

        colour = {
            "it_support": C.GREEN,
            "hr_support": C.YELLOW,
            "ticket_create": C.RED,
            "ticket_status": C.MAGENTA,
            "general_chat": C.WHITE,
        }.get(agent, C.WHITE)

        print(f"   {C.BOLD}Task {i}:{C.RESET} {colour}{agent}{C.RESET}")
        print(f"          Query: {C.DIM}{_truncate(query, 120)}{C.RESET}")
        if focus:
            print(f"          Focus: {C.DIM}{focus}{C.RESET}")

    synth = f"{C.GREEN}Yes{C.RESET}" if requires_synthesis else f"{C.DIM}No{C.RESET}"
    print(f"   Parallel synthesis: {synth}")
    print(f"{_DIVIDER}")


def log_agent_start(agent_name: str) -> float:
    colour = {
        "IT Support": C.GREEN,
        "HR Support": C.YELLOW,
        "Ticket Creator": C.RED,
        "Ticket Status": C.MAGENTA,
        "General Chat": C.WHITE,
    }.get(agent_name, C.CYAN)

    print(f"\n{colour}{C.BOLD}🤖 {agent_name} Agent — Started{C.RESET}")
    return time.time()


def log_tool_call(tool_name: str, args: dict) -> None:
    print(f"   {C.CYAN}🔧 Tool call:{C.RESET} {C.BOLD}{tool_name}{C.RESET}")
    for k, v in args.items():
        val_str = str(v)
        print(f"      {C.DIM}{k}:{C.RESET} {_truncate(val_str, 100)}")


def log_tool_result(tool_name: str, result_preview: str, confidence: str = "") -> None:
    conf_badge = ""
    if confidence == "high":
        conf_badge = f" {C.BG_GREEN}{C.WHITE} HIGH CONFIDENCE {C.RESET}"
    elif confidence == "low":
        conf_badge = f" {C.BG_YELLOW}{C.WHITE} LOW CONFIDENCE {C.RESET}"
    elif confidence == "none":
        conf_badge = f" {C.BG_RED}{C.WHITE} NO RESULTS {C.RESET}"

    print(f"   {C.CYAN}📋 Tool result:{C.RESET} {tool_name}{conf_badge}")
    print(f"      {C.DIM}{_truncate(result_preview, 200)}{C.RESET}")


def log_agent_response(agent_name: str, response: str, start_time: float) -> None:
    elapsed = _elapsed(start_time)

    colour = {
        "IT Support": C.GREEN,
        "HR Support": C.YELLOW,
        "Ticket Creator": C.RED,
        "Ticket Status": C.MAGENTA,
        "General Chat": C.WHITE,
    }.get(agent_name, C.CYAN)

    print(f"\n{colour}{C.BOLD}✅ {agent_name} Agent — Done{C.RESET} {C.DIM}({elapsed}){C.RESET}")
    print(f"   {C.DIM}{_truncate(response, 250)}{C.RESET}")


def log_synthesis(num_results: int) -> None:
    if num_results > 1:
        print(f"\n{C.BOLD}{C.MAGENTA}🔀 SYNTHESIZER — Merging {num_results} agent outputs{C.RESET}")
    else:
        print(f"\n{C.DIM}🔀 SYNTHESIZER — Single result, passing through{C.RESET}")


def log_final_answer() -> None:
    print(f"\n{_SECTION}")
    print(f"{C.BOLD}{C.GREEN}📤 FINAL RESPONSE{C.RESET}")
    print(f"{_DIVIDER}")


def log_email_captured(email: str) -> None:
    print(f"   {C.MAGENTA}📧 Email captured:{C.RESET} {email}")


def log_ticket_created(ticket_id: str) -> None:
    print(f"   {C.RED}{C.BOLD}🎫 Ticket created:{C.RESET} {ticket_id}")


def log_error(error: str) -> None:
    print(f"\n{C.RED}{C.BOLD}❌ ERROR:{C.RESET} {C.RED}{error}{C.RESET}")


def log_retry(attempt: int, max_attempts: int) -> None:
    print(f"   {C.YELLOW}🔄 Retry {attempt}/{max_attempts}...{C.RESET}")
