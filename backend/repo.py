"""Thin DB access layer.

Keeps SQL out of HTTP handlers and the orchestrator. Every write commits
immediately (we run with isolation_level=None).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from backend.db import connect
from backend.scoring import CallStats, Patient, Slot, parse_patient, parse_slot


def all_patients() -> list[Patient]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM patients").fetchall()
    return [parse_patient(r) for r in rows]


def get_patient(pid: int) -> Optional[Patient]:
    with _conn() as c:
        row = c.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    return parse_patient(row) if row else None


def get_slot(sid: int) -> Optional[Slot]:
    with _conn() as c:
        row = c.execute("SELECT * FROM slots WHERE id = ?", (sid,)).fetchone()
    return parse_slot(row) if row else None


def slot_status(sid: int) -> Optional[str]:
    with _conn() as c:
        r = c.execute("SELECT status FROM slots WHERE id = ?", (sid,)).fetchone()
    return r["status"] if r else None


def set_slot_status(sid: int, status: str, booked_patient_id: Optional[int] = None) -> None:
    with _conn() as c:
        if booked_patient_id is None:
            c.execute("UPDATE slots SET status = ? WHERE id = ?", (status, sid))
        else:
            c.execute(
                "UPDATE slots SET status = ?, booked_patient_id = ? WHERE id = ?",
                (status, booked_patient_id, sid),
            )


def cancel_slot(sid: int) -> None:
    set_slot_status(sid, "cancelled")


def offers_this_week_by_pid() -> dict[int, int]:
    """Crude weekly cap counter: count offer-direction calls in the last 7d."""
    since = (datetime.now() - timedelta(days=7)).isoformat()
    with _conn() as c:
        rows = c.execute(
            """SELECT patient_id, COUNT(*) AS n FROM calls
               WHERE direction = 'outbound' AND started_at >= ?
               GROUP BY patient_id""",
            (since,),
        ).fetchall()
    return {r["patient_id"]: r["n"] for r in rows}


def call_stats_by_pid() -> dict[int, CallStats]:
    """Per-patient offer-call history for the learned P(answer)/P(accept) signal.

    answered = picked up (booked/declined/callback); accepted = booked.
    All history (no date filter) — this is lifetime responsiveness, not the
    weekly contact cap.
    """
    with _conn() as c:
        rows = c.execute(
            """SELECT patient_id,
                      COUNT(*) AS offers,
                      SUM(CASE WHEN outcome IN ('booked','declined','callback')
                               THEN 1 ELSE 0 END) AS answered,
                      SUM(CASE WHEN outcome = 'booked' THEN 1 ELSE 0 END) AS accepted
               FROM calls
               WHERE direction = 'outbound' AND outcome IS NOT NULL
               GROUP BY patient_id""",
        ).fetchall()
    return {
        r["patient_id"]: CallStats(
            offers=r["offers"], answered=r["answered"] or 0, accepted=r["accepted"] or 0
        )
        for r in rows
    }


def create_recovery_attempt(sid: int) -> int:
    """Returns recovery_attempt.id. UNIQUE(slot_id) guards duplicates → raises."""
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO recovery_attempts(slot_id, created_at, status)
               VALUES (?, ?, ?)""",
            (sid, datetime.now().isoformat(), "in_progress"),
        )
        return cur.lastrowid


def recovery_attempt_for(sid: int) -> Optional[dict]:
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM recovery_attempts WHERE slot_id = ?", (sid,)
        ).fetchone()
    return dict(r) if r else None


def finish_recovery(sid: int, status: str, filled_pid: Optional[int]) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE recovery_attempts
               SET status = ?, filled_by_patient_id = ?
               WHERE slot_id = ?""",
            (status, filled_pid, sid),
        )


def log_call(
    *,
    fonio_call_id: Optional[str],
    recovery_attempt_id: Optional[int],
    patient_id: int,
    slot_id: int,
    direction: str,
    outcome: Optional[str],
    summary: Optional[str],
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO calls(fonio_call_id, recovery_attempt_id, patient_id,
                                 slot_id, direction, outcome, summary,
                                 started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fonio_call_id) DO UPDATE SET
                 outcome = excluded.outcome,
                 summary = excluded.summary,
                 ended_at = excluded.ended_at""",
            (
                fonio_call_id, recovery_attempt_id, patient_id, slot_id,
                direction, outcome, summary,
                started_at or datetime.now().isoformat(),
                ended_at,
            ),
        )
        return cur.lastrowid


def update_call_outcome(fonio_call_id: str, outcome: str, summary: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE calls SET outcome = ?, summary = ?, ended_at = ?
               WHERE fonio_call_id = ?""",
            (outcome, summary, datetime.now().isoformat(), fonio_call_id),
        )


def find_call_by_fonio_id(fonio_call_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM calls WHERE fonio_call_id = ?", (fonio_call_id,)
        ).fetchone()
    return dict(r) if r else None


def record_offer_called(pid: int) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE patients SET last_offer_called_at = ? WHERE id = ?",
            (datetime.now().isoformat(), pid),
        )


def record_decline(pid: int, slot_type: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE patients
               SET last_decline_at = ?, last_declined_slot_type = ?
               WHERE id = ?""",
            (datetime.now().isoformat(), slot_type, pid),
        )


def push_attendance(pid: int, showed: int) -> None:
    """Maintain last-5 sliding window."""
    p = get_patient(pid)
    if not p:
        return
    hist = p.attendance_history + [showed]
    hist = hist[-5:]
    with _conn() as c:
        c.execute(
            "UPDATE patients SET attendance_history = ? WHERE id = ?",
            (json.dumps(hist), pid),
        )


def schedule(*, days_ahead: int = 7) -> list[dict]:
    until = (datetime.now() + timedelta(days=days_ahead)).isoformat()
    with _conn() as c:
        rows = c.execute(
            """SELECT s.*, p.name AS booked_name
               FROM slots s LEFT JOIN patients p ON p.id = s.booked_patient_id
               WHERE s.start_dt <= ? ORDER BY s.start_dt""",
            (until,),
        ).fetchall()
    return [dict(r) for r in rows]


def outcomes_breakdown() -> dict[str, int]:
    with _conn() as c:
        rows = c.execute(
            "SELECT outcome, COUNT(*) AS n FROM calls "
            "WHERE outcome IS NOT NULL GROUP BY outcome"
        ).fetchall()
    return {r["outcome"]: r["n"] for r in rows}


def refill_stats() -> dict:
    with _conn() as c:
        rec = c.execute(
            "SELECT COUNT(*) AS n FROM recovery_attempts"
        ).fetchone()["n"]
        filled = c.execute(
            "SELECT COUNT(*) AS n FROM recovery_attempts WHERE status = 'filled'"
        ).fetchone()["n"]
        eur = c.execute(
            """SELECT COALESCE(SUM(s.value_eur), 0) AS eur
               FROM recovery_attempts r JOIN slots s ON s.id = r.slot_id
               WHERE r.status = 'filled'"""
        ).fetchone()["eur"]
    return {"attempts": rec, "filled": filled, "eur": eur}


# ---- internal ----
class _conn:
    def __init__(self):
        self.c = None
    def __enter__(self):
        self.c = connect()
        return self.c
    def __exit__(self, *a):
        self.c.close()
