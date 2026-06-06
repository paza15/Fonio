"""Unit tests for the scoring engine (§5.2-5.4): filters, accept score,
deadline-aware priority, phase, and ranking."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from backend.scoring import (
    CallStats, _bayes, accept_score, apply_hard_filters, deadline_priority,
    learned_accept, learned_answer, phase_of, rank,
)
from conftest import make_patient, make_slot

NOW = datetime(2026, 6, 6, 12, 0)


# --- hard filters (§5.3) ---

def test_skip_no_consent():
    p = make_patient(consent_outbound=False)
    slot = make_slot(start=NOW + timedelta(hours=48))
    assert apply_hard_filters(p, slot, NOW) == "No consent for outbound calls"


def test_skip_treatment_mismatch():
    p = make_patient(needed_treatments=["crown"])
    slot = make_slot(type="cleaning", start=NOW + timedelta(hours=48))
    assert apply_hard_filters(p, slot, NOW) == "Treatment mismatch"


def test_skip_short_notice():
    p = make_patient(short_notice_ok=False)
    slot = make_slot(start=NOW + timedelta(hours=3))  # < 24h
    assert apply_hard_filters(p, slot, NOW) == "Cannot make it on short notice"


def test_short_notice_ok_passes_when_flagged():
    p = make_patient(short_notice_ok=True)
    slot = make_slot(start=NOW + timedelta(hours=3))
    assert apply_hard_filters(p, slot, NOW) is None


def test_skip_cooldown_unless_only_feasible():
    p = make_patient(last_offer_called_at=NOW - timedelta(hours=10))  # < 72h
    slot = make_slot(start=NOW + timedelta(hours=48))
    assert apply_hard_filters(p, slot, NOW) == "Cooldown (called recently)"
    # the rescue path: only feasible candidate ⇒ cooldown is waived
    assert apply_hard_filters(p, slot, NOW, only_feasible=True) is None


def test_skip_recently_declined_similar():
    p = make_patient(last_decline_at=NOW - timedelta(days=3), last_declined_slot_type="cleaning")
    slot = make_slot(type="cleaning", start=NOW + timedelta(hours=48))
    assert apply_hard_filters(p, slot, NOW) == "Recently declined similar"


def test_skip_weekly_cap():
    p = make_patient()
    slot = make_slot(start=NOW + timedelta(hours=48))
    assert apply_hard_filters(p, slot, NOW, offers_this_week=2) == "Weekly contact cap"


# --- accept score (§5.2) ---

def test_accept_score_window_and_treatment_bonus():
    p = make_patient(preferred_window_start=time(8, 0), preferred_window_end=time(19, 0),
                     needed_treatments=["cleaning"], days_waiting=0)
    slot = make_slot(type="cleaning", start=datetime(2026, 6, 8, 10, 0))
    # 0.5 base + 0.25 window + 0.15 treatment = 0.9
    assert abs(accept_score(p, slot) - 0.9) < 1e-9


def test_accept_score_clipped():
    p = make_patient(preferred_window_start=time(0, 0), preferred_window_end=time(23, 59),
                     needed_treatments=["cleaning"], days_waiting=200)  # huge wait
    slot = make_slot(type="cleaning", start=datetime(2026, 6, 8, 10, 0))
    assert accept_score(p, slot) <= 0.95


# --- deadline priority (§5.4) ---

def test_starvation_guard_boost():
    base, _ = deadline_priority(0.8, 0.8, 0.5, hours_left=48, days_waiting=10)
    boosted, _ = deadline_priority(0.8, 0.8, 0.5, hours_left=48, days_waiting=31)
    assert abs(boosted - base * 1.5) < 1e-9


def test_urgency_mode_bounds():
    _, u_relaxed = deadline_priority(0.8, 0.8, 0.5, hours_left=100, days_waiting=0)
    _, u_panic = deadline_priority(0.8, 0.8, 0.5, hours_left=0, days_waiting=0)
    assert u_relaxed == 0.0 and u_panic == 1.0


# --- phase (§5.4) ---

def test_phase_boundaries():
    assert phase_of(make_slot(start=NOW + timedelta(hours=48)), NOW) == "RELAXED"
    assert phase_of(make_slot(start=NOW + timedelta(hours=5)), NOW) == "URGENT"
    assert phase_of(make_slot(start=NOW + timedelta(minutes=60)), NOW) == "CRITICAL"
    assert phase_of(make_slot(start=NOW + timedelta(minutes=10)), NOW) == "UNRECOVERABLE"


# --- rank (§5.4) ---

def _const_reliability(p, lead_days=0.0):
    return 0.8


def test_rank_sorts_desc_and_skips():
    good = make_patient(id=1, days_waiting=40)
    no_consent = make_patient(id=2, consent_outbound=False)
    mismatch = make_patient(id=3, needed_treatments=["crown"])
    slot = make_slot(type="cleaning", start=NOW + timedelta(hours=48))
    ranked, skipped, phase = rank(slot, [good, no_consent, mismatch], _const_reliability, now=NOW)
    assert [r.patient_id for r in ranked] == [1]
    reasons = {s.patient_id: s.reason for s in skipped}
    assert reasons[2] == "No consent for outbound calls"
    assert reasons[3] == "Treatment mismatch"
    assert phase == "RELAXED"


def test_rank_unrecoverable_returns_empty():
    slot = make_slot(start=NOW + timedelta(minutes=10))
    ranked, skipped, phase = rank(slot, [make_patient()], _const_reliability, now=NOW)
    assert ranked == [] and phase == "UNRECOVERABLE"


def test_rank_excludes_ids_and_caps_top_k():
    patients = [make_patient(id=i, days_waiting=i) for i in range(1, 9)]
    slot = make_slot(type="cleaning", start=NOW + timedelta(hours=48))
    ranked, _, _ = rank(slot, patients, _const_reliability, exclude_ids={1, 2}, now=NOW)
    ids = {r.patient_id for r in ranked}
    assert 1 not in ids and 2 not in ids
    assert len(ranked) <= 5


# --- learned signal from the call log (§5.2 upgrade) ---

def test_bayes_cold_start_returns_prior():
    assert _bayes(0.8, 4, 0, 0) == 0.8


def test_learned_accept_pulls_down_on_declines():
    # prior 0.8, but the patient answered 4 offers and accepted none
    assert learned_accept(0.8, CallStats(offers=4, answered=4, accepted=0)) < 0.5


def test_learned_accept_cold_start_is_prior():
    assert learned_accept(0.8, None) == 0.8
    assert learned_accept(0.8, CallStats()) == 0.8  # no answered calls yet


def test_learned_answer_pulls_down_on_voicemails():
    assert learned_answer(0.85, CallStats(offers=4, answered=0, accepted=0)) < 0.6


def test_rank_demotes_chronic_decliner():
    a = make_patient(id=1)
    b = make_patient(id=2)  # identical patient, no call history
    slot = make_slot(type="cleaning", start=NOW + timedelta(hours=48))
    stats = {1: CallStats(offers=5, answered=5, accepted=0)}  # #1 always declines
    ranked, _, _ = rank(slot, [a, b], _const_reliability, call_stats_by_pid=stats, now=NOW)
    assert [r.patient_id for r in ranked] == [2, 1]
    assert ranked[1].call_history == (5, 5, 0)
