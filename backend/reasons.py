"""Reason strings for candidates (§5.5).

Always has a deterministic fallback template (PLAN explicitly requires it).
LLM is opt-in via env var; we never block the demo on a network call.
"""

from __future__ import annotations

from backend.scoring import Ranked


def template_reason(r: Ranked) -> str:
    """`"{days_waiting}d waiting · {window match?} · attendance {x}/5"`,
    plus the learned call record when the patient has offer history."""
    win = "window match" if r.window_match else "outside preferred window"
    attended, total = r.attendance
    parts = [f"{r.days_waiting}d waiting", win, f"attendance {attended}/{total}"]
    offers, answered, _accepted = r.call_history
    if offers:
        parts.append(f"answered {answered}/{offers} past calls")
    return " · ".join(parts)


def reason_for(r: Ranked) -> str:
    # Hook for LLM upgrade (Anthropic/OpenAI). Stays deterministic for the demo.
    return template_reason(r)
