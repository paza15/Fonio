"""Reason strings for candidates (§5.5).

Always has a deterministic fallback template (PLAN explicitly requires it).
LLM is opt-in via env var; we never block the demo on a network call.
"""

from __future__ import annotations

from backend.scoring import Ranked


def template_reason(r: Ranked) -> str:
    """`"{days_waiting}d waiting · {window match?} · attendance {x}/5"`"""
    win = "window match" if r.window_match else "outside preferred window"
    attended, total = r.attendance
    return f"{r.days_waiting}d waiting · {win} · attendance {attended}/{total}"


def reason_for(r: Ranked) -> str:
    # Hook for LLM upgrade (Anthropic/OpenAI). Stays deterministic for the demo.
    return template_reason(r)
