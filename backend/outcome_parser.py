"""Deterministic outcome token parser (§6.3).

The offer agent must end its summary with one of:
    OUTCOME_BOOKED  OUTCOME_DECLINED  OUTCOME_CALLBACK
If the regex misses (drift, bad agent run), we fall back to a keyword classifier.
LLM classification is reserved for a later upgrade — not in this file.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"OUTCOME_(BOOKED|DECLINED|CALLBACK|VOICEMAIL)\b", re.I)


def parse_outcome(summary: str | None, *, disconnect_reason: str | None = None) -> str:
    if not summary:
        if disconnect_reason and "voicemail" in disconnect_reason.lower():
            return "voicemail"
        return "timeout"
    m = TOKEN_RE.search(summary)
    if m:
        return m.group(1).lower()
    s = summary.lower()
    if "voicemail" in s or (disconnect_reason and "voicemail" in disconnect_reason.lower()):
        return "voicemail"
    if any(k in s for k in ("will take", "i'll take", "ill take", "yes", "confirm", "book")):
        return "booked"
    if any(k in s for k in ("call me back", "later", "callback", "call back")):
        return "callback"
    return "declined"
