"""Thin DB access layer.

Keeps SQL out of HTTP handlers and the orchestrator. Every write commits
immediately (we run with isolation_level=None).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from backend.db import connect
from backend.scoring import (
    TREATMENT_MINUTES, TREATMENT_VALUE, CallStats, Patient, Slot,
    parse_patient, parse_slot,
)


def all_patients() -> list[Patient]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM patients").fetchall()
    return [parse_patient(r) for r in rows]


def get_patient(pid: int) -> Optional[Patient]:
    with _conn() as c:
        row = c.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    return parse_patient(row) if row else None


def waitlist_patients() -> list[Patient]:
    """Patients with NO active appointment — the true waitlist. Excludes anyone
    currently booked (they're handled by pull-forward, not cold-offered a slot)."""
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM patients WHERE id NOT IN (
                 SELECT booked_patient_id FROM slots
                 WHERE booked_patient_id IS NOT NULL
                   AND status IN ('booked', 'filled', 'recovering')
               )"""
        ).fetchall()
    return [parse_patient(r) for r in rows]


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


# --- slot state machine (guarded transitions) -------------------------------
#
# A slot moves only along legal edges. Each edge is an ATOMIC compare-and-set on
# the current status, so a stale/duplicate/concurrent caller that names the wrong
# `from` state affects 0 rows and is rejected — it can never clobber a confirmed
# booking (e.g. trigger-recovery can't overwrite a 'filled' slot, cancel can't
# evict a slot mid-recovery). This is the documented contract; the WHERE-clause
# is what enforces it.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "cancelled":     {"booked", "filled", "open"},   # a real future appt is cancelled
    "recovering":    {"cancelled", "open"},           # a freed slot enters recovery
    "filled":        {"recovering"},                  # see fill_slot / try_fill_slot
    "escalated":     {"recovering"},                  # handed to a human
    "unrecoverable": {"recovering"},                  # too late to refill
}


def transition_slot(sid: int, allowed_from: set[str], to: str,
                    *, clear_patient: bool = False) -> bool:
    """Atomic CAS status transition. Returns True iff the slot was in one of
    `allowed_from` (rowcount == 1). False = the slot was in some other state and
    nothing was written — the guard held."""
    inq = ",".join("?" for _ in allowed_from)
    extra = ", booked_patient_id = NULL" if clear_patient else ""
    with _conn() as c:
        cur = c.execute(
            f"UPDATE slots SET status = ?{extra} WHERE id = ? AND status IN ({inq})",
            (to, sid, *allowed_from),
        )
    return cur.rowcount == 1


def cancel_slot(sid: int) -> bool:
    """Cancel a real appointment (booked/filled/open). NULLs booked_patient_id —
    the patient is no longer attached to this slot (matches free_slot /
    pull_forward_commit). REFUSES to cancel a slot that is 'recovering' (don't yank
    it mid-refill) or already terminal. Returns True iff it actually cancelled."""
    return transition_slot(sid, ALLOWED_TRANSITIONS["cancelled"], "cancelled",
                           clear_patient=True)


def begin_recovery(sid: int) -> bool:
    """Move a freed slot (cancelled/open) into 'recovering'. Atomic CAS, so two
    concurrent cancellations for the same slot can't both start a recovery, and a
    recovery can never be triggered on an already-filled/booked slot. Returns True
    iff THIS caller started the recovery."""
    return transition_slot(sid, ALLOWED_TRANSITIONS["recovering"], "recovering")


def escalate_if_recovering(sid: int) -> bool:
    """Terminal escalate, but ONLY if the slot is still 'recovering'. Used by the
    crash handler so a half-finished recovery is surfaced to a human instead of
    being left stuck — and so it can NEVER overwrite a slot a concurrent winner
    already filled (that slot is no longer 'recovering' → 0 rows → no-op)."""
    return transition_slot(sid, {"recovering"}, "escalated")


def upcoming_booked(start, until) -> list[dict]:
    """Booked appointments with a patient in the [start, until] window — the pool
    the proactive no-show sweep confirmation-calls."""
    s = start.isoformat() if hasattr(start, "isoformat") else start
    u = until.isoformat() if hasattr(until, "isoformat") else until
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM slots
               WHERE status = 'booked' AND booked_patient_id IS NOT NULL
                 AND start_dt > ? AND start_dt <= ? ORDER BY start_dt""",
            (s, u),
        ).fetchall()
    return [dict(r) for r in rows]


def set_confirmation(sid: int, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE slots SET confirmation_status = ? WHERE id = ?", (status, sid))


def fill_slot(sid: int, patient_id: int, treatment: str, minutes: int, value: int) -> bool:
    """Book a patient into a freed slot for a specific treatment, adopting that
    treatment's type/duration/value (the freed time becomes their appointment).

    ATOMIC compare-and-set: only fills a slot still in 'recovering'. Returns True
    if THIS writer won the slot (rowcount == 1), False if it was already filled by
    a concurrent winner / a duplicate webhook (rowcount == 0). The caller MUST
    honour the bool and skip booking/attendance on False — no double-book."""
    with _conn() as c:
        cur = c.execute(
            """UPDATE slots SET status = 'filled', booked_patient_id = ?,
                                type = ?, duration_min = ?, value_eur = ?
               WHERE id = ? AND status = 'recovering'""",
            (patient_id, treatment, minutes, value, sid),
        )
        return cur.rowcount == 1


def try_fill_slot(sid: int, pid: int) -> bool:
    """Atomic conditional commit (compare-and-set). Only succeeds if the slot is
    still 'recovering'. The race winner gets True; any second/duplicate writer
    gets rowcount 0 -> False, so the slot can never be double-booked. This is the
    booked-commit guard for the two-patients-one-slot race AND for duplicate /
    out-of-order post-call deliveries.

    NB: the slot is set to 'recovering' at trigger time (orchestrator.trigger_recovery)
    and stays 'recovering' through the entire dial loop, so this precondition never
    false-rejects a legitimate first booking.
    """
    with _conn() as c:
        cur = c.execute(
            "UPDATE slots SET status = 'filled', booked_patient_id = :pid "
            "WHERE id = :sid AND status = 'recovering'",
            {"pid": pid, "sid": sid},
        )
        return cur.rowcount == 1


def mark_event_processed(event_key: str, kind: str) -> bool:
    """Idempotency dedup. INSERT OR IGNORE into processed_events. Returns True if
    newly inserted (first delivery -> caller should proceed), False if the key was
    already present (duplicate delivery -> caller should no-op)."""
    with _conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO processed_events(event_key, kind, received_at) "
            "VALUES (?, ?, ?)",
            (event_key, kind, datetime.now().isoformat()),
        )
        return cur.rowcount == 1


def pull_forward_commit(from_slot_id: int, to_slot_id: int, patient_id: int,
                        treatment: str, minutes: int, value: int) -> bool:
    """ATOMIC pull-forward: fill the earlier slot (to_slot_id) with the patient and
    free their later slot (from_slot_id) in ONE transaction, so the move can never
    leave the patient double-booked (filled into A while still booked in B) and two
    concurrent recoveries can never both pull the same patient.

    Two compare-and-sets, both of which must succeed inside the SAME txn:
      1. fill the earlier slot — CAS on status='recovering'. Loses if another
         recovery already filled A / A is no longer recovering.
      2. free the later slot — CAS on booked_patient_id=patient_id. Loses if a
         CONCURRENT pull-forward already moved this patient out of B into a
         DIFFERENT earlier slot.
    If EITHER CAS affects 0 rows, the whole move ROLLs BACK and returns False, so
    the patient can land in at most ONE earlier slot and B is freed at most once.
    This is what stops the cross-slot double-book when two recoveries race the
    same booked patient into two different open slots (the to_slot CAS alone does
    not serialize them — they target different rows; the from_slot CAS does)."""
    c = connect()  # isolation_level=None -> we control the txn explicitly
    try:
        c.execute("BEGIN IMMEDIATE")
        fill = c.execute(
            """UPDATE slots SET status = 'filled', booked_patient_id = ?,
                                type = ?, duration_min = ?, value_eur = ?
               WHERE id = ? AND status = 'recovering'""",
            (patient_id, treatment, minutes, value, to_slot_id),
        )
        if fill.rowcount != 1:
            c.execute("ROLLBACK")
            return False
        free = c.execute(
            "UPDATE slots SET status = 'cancelled', booked_patient_id = NULL "
            "WHERE id = ? AND booked_patient_id = ?",
            (from_slot_id, patient_id),
        )
        if free.rowcount != 1:
            # the patient was already pulled out of their later slot by a concurrent
            # move — abort so we don't double-book them into two earlier slots.
            c.execute("ROLLBACK")
            return False
        c.execute("COMMIT")
        return True
    except Exception:
        try:
            c.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        c.close()


def free_slot(sid: int) -> None:
    """Vacate a slot (a patient was pulled forward out of it) so it can recover."""
    with _conn() as c:
        c.execute(
            "UPDATE slots SET status = 'cancelled', booked_patient_id = NULL WHERE id = ?",
            (sid,),
        )


def create_open_slot(start_dt, duration_min: int, slot_type: str, value_eur: int) -> int:
    """Create a new open (cancelled→to-recover) slot — used for leftover capacity
    after packing. Returns its id."""
    start = start_dt.isoformat() if hasattr(start_dt, "isoformat") else start_dt
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO slots(start_dt, duration_min, type, value_eur, status,
                                 booked_patient_id)
               VALUES (?, ?, ?, ?, 'cancelled', NULL)""",
            (start, duration_min, slot_type, value_eur),
        )
        return cur.lastrowid


def pull_forward_candidates(slot: Slot, capacity_min: int, *, now=None,
                            exclude_pids: Optional[set[int]] = None) -> list[tuple]:
    """Patients booked LATER than `slot` who could be pulled forward into it.

    Nearby slots (≤14d) may swap any treatment that fits; weeks-ahead slots only
    the SAME treatment. Respects consent + short-notice. Returns
    [(booked_slot, patient, treatment, minutes, value), ...] sorted by how much
    earlier they'd be seen (fairness) — money is never used here.
    """
    now = now or datetime.now()
    exclude = exclude_pids or set()
    near_cutoff = slot.start + timedelta(days=14)
    short_notice_slot = (slot.start - now).total_seconds() < 24 * 3600
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM slots
               WHERE status = 'booked' AND booked_patient_id IS NOT NULL
                 AND start_dt > ? ORDER BY start_dt""",
            (slot.start.isoformat(),),
        ).fetchall()
    out = []
    for r in rows:
        bslot = parse_slot(r)
        patient = get_patient(r["booked_patient_id"])
        if not patient or patient.id in exclude or not patient.consent_outbound:
            continue
        if short_notice_slot and not patient.short_notice_ok:
            continue
        minutes = TREATMENT_MINUTES.get(bslot.type, bslot.duration_min)
        value = TREATMENT_VALUE.get(bslot.type, bslot.value_eur)
        if minutes > capacity_min:
            continue
        if bslot.start > near_cutoff and bslot.type != slot.type:
            continue
        days_earlier = (bslot.start - slot.start).days
        out.append((days_earlier, bslot, patient, bslot.type, minutes, value))
    out.sort(key=lambda x: x[0], reverse=True)
    return [(b, p, t, m, v) for (_d, b, p, t, m, v) in out]


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
    """Open (or RE-open) a recovery attempt for a slot. Returns the attempt id.

    Idempotency / re-recovery contract, enforced atomically under BEGIN IMMEDIATE:
      * no attempt row yet                 → INSERT a fresh 'in_progress' attempt.
      * an attempt row exists, terminal    → RESET it to 'in_progress' (a slot that
        was filled/escalated earlier and is now being cancelled + recovered AGAIN —
        a slot must be recoverable more than once, not bricked for life by the
        UNIQUE(slot_id) row).
      * an attempt row exists, 'in_progress' → RAISE IntegrityError. This is the
        duplicate-cancellation-event guard (two cancels for the same slot while a
        recovery is already running) — same semantics the old UNIQUE insert gave.
    """
    c = connect()  # isolation_level=None → explicit txn
    try:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT id, status FROM recovery_attempts WHERE slot_id = ?", (sid,)
        ).fetchone()
        now = datetime.now().isoformat()
        if row is None:
            cur = c.execute(
                "INSERT INTO recovery_attempts(slot_id, created_at, status) VALUES (?,?,?)",
                (sid, now, "in_progress"),
            )
            c.execute("COMMIT")
            return cur.lastrowid
        if row["status"] == "in_progress":
            c.execute("ROLLBACK")
            raise sqlite3.IntegrityError(
                f"recovery already in progress for slot {sid} (duplicate cancellation)")
        # terminal attempt → this is a NEW recovery cycle for the same slot.
        c.execute(
            "UPDATE recovery_attempts SET status='in_progress', created_at=?, "
            "filled_by_patient_id=NULL, leased_until=NULL, redrive_count=0 WHERE slot_id=?",
            (now, sid),
        )
        c.execute("COMMIT")
        return row["id"]
    except Exception:
        try:
            c.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        c.close()


def delete_recovery_attempt(sid: int) -> None:
    """Remove an attempt row (used to roll back when a recovery aborts before it
    really started, e.g. the slot turned out not to be recoverable)."""
    with _conn() as c:
        c.execute("DELETE FROM recovery_attempts WHERE slot_id = ?", (sid,))


# --- reaper lease / orphan detection ----------------------------------------

def set_recovery_lease(sid: int, leased_until_iso: str) -> None:
    """A live worker stamps/renews its lease. While leased_until is in the future
    the watchdog leaves it alone."""
    with _conn() as c:
        c.execute(
            "UPDATE recovery_attempts SET leased_until = ? WHERE slot_id = ?",
            (leased_until_iso, sid),
        )


def orphaned_recovery_slot_ids(now_iso: str) -> list[int]:
    """Slots stuck mid-recovery whose worker is gone: attempt still 'in_progress',
    slot still 'recovering', and the lease is absent or expired. These are the
    bricked slots the watchdog re-drives."""
    with _conn() as c:
        rows = c.execute(
            """SELECT ra.slot_id AS sid
               FROM recovery_attempts ra JOIN slots s ON s.id = ra.slot_id
               WHERE ra.status = 'in_progress' AND s.status = 'recovering'
                 AND (ra.leased_until IS NULL OR ra.leased_until < ?)""",
            (now_iso,),
        ).fetchall()
    return [r["sid"] for r in rows]


def claim_orphan(sid: int, now_iso: str, new_lease_iso: str) -> bool:
    """Atomically claim an orphaned recovery: set a fresh lease AND bump
    redrive_count IFF it is still an expired 'in_progress' attempt. Returns True for
    the single claimer; concurrent watchdog ticks / multiple workers get False
    (rowcount 0) — exactly one re-drive. The redrive_count bump bounds re-drives so
    a never-answering number can't be re-called forever (see recovery_redrive_count)."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE recovery_attempts SET leased_until = ?, redrive_count = redrive_count + 1 "
            "WHERE slot_id = ? AND status = 'in_progress' "
            "AND (leased_until IS NULL OR leased_until < ?)",
            (new_lease_iso, sid, now_iso),
        )
    return cur.rowcount == 1


def recovery_redrive_count(sid: int) -> int:
    """How many times the reaper has re-driven this recovery (0 if no row)."""
    with _conn() as c:
        r = c.execute(
            "SELECT redrive_count FROM recovery_attempts WHERE slot_id = ?", (sid,)
        ).fetchone()
    return r["redrive_count"] if r else 0


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
    to_number: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO calls(fonio_call_id, recovery_attempt_id, patient_id,
                                 slot_id, to_number, direction, outcome, summary,
                                 started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fonio_call_id) DO UPDATE SET
                 outcome = excluded.outcome,
                 summary = excluded.summary,
                 ended_at = excluded.ended_at""",
            (
                fonio_call_id, recovery_attempt_id, patient_id, slot_id,
                to_number, direction, outcome, summary,
                started_at or datetime.now().isoformat(),
                ended_at,
            ),
        )
        return cur.lastrowid


def find_inflight_call_id(*, slot_id=None, patient_id=None, to_number=None) -> Optional[str]:
    """PERSISTED correlation fallback for the post-call webhook (survives a process
    restart, unlike in-memory STATE.pending_calls). Find the most recent
    not-yet-resolved (outcome IS NULL) outbound/confirmation call matching the
    given keys, and return its fonio_call_id. Used only when the webhook can't read
    our call_attempt_id straight out of the round-tripped `context`."""
    clauses = ["outcome IS NULL"]
    params: list = []
    if slot_id is not None:
        clauses.append("slot_id = ?"); params.append(slot_id)
    if patient_id is not None:
        clauses.append("patient_id = ?"); params.append(patient_id)
    if to_number is not None:
        clauses.append("to_number = ?"); params.append(to_number)
    if not params:
        return None
    with _conn() as c:
        r = c.execute(
            f"SELECT fonio_call_id FROM calls WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC LIMIT 1",
            params,
        ).fetchone()
    return r["fonio_call_id"] if r and r["fonio_call_id"] else None


def record_dead_letter(payload: str, reason: str) -> int:
    """Park an uncorrelatable webhook payload instead of dropping it."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO dead_letters(received_at, reason, payload) VALUES (?,?,?)",
            (datetime.now().isoformat(), reason, payload),
        )
        return cur.lastrowid


def count_dead_letters() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) AS n FROM dead_letters").fetchone()["n"]


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
