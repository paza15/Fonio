"""Unit tests for the capacity-aware recovery planner."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from backend.scheduling import plan_recovery
from backend.scoring import Patient, Slot

NOW = datetime(2026, 6, 6, 9, 0)
VALUES = {"cleaning": 80, "checkup": 60, "filling": 150, "crown": 600}
REL = lambda p, lead=0.0: 0.85


def P(pid, name, treatments, days=10, consent=True, short_notice=True):
    return Patient(
        id=pid, name=name, phone=f"+43000{pid:05d}", age=35, sms_opt_in=True,
        hypertension=False, diabetes=False, consent_outbound=consent,
        short_notice_ok=short_notice, preferred_window_start=time(8, 0),
        preferred_window_end=time(19, 0), needed_treatments=list(treatments),
        days_waiting=days, attendance_history=[1, 1, 1, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None)


def Sl(sid, ttype, minutes, when):
    return Slot(id=sid, start=when, duration_min=minutes, type=ttype, value_eur=VALUES[ttype])


def test_packing_fills_a_long_slot_with_two_short_treatments():
    slot = Sl(1, "crown", 60, NOW + timedelta(hours=5))
    wl = [P(1, "A", ["cleaning"], days=35), P(2, "B", ["checkup"], days=22)]
    plan = plan_recovery(slot, wl, [], REL, now=NOW)
    assert len(plan.bookings) == 2
    assert plan.filled_min == 60 and plan.utilization == 1.0
    assert {b.treatment for b in plan.bookings} == {"cleaning", "checkup"}


def test_pull_forward_when_waitlist_cannot_fill():
    slot = Sl(1, "crown", 60, NOW + timedelta(hours=5))   # short notice today
    wl = [P(1, "Greta", ["crown"], short_notice=False)]   # can't do today
    booked = [(Sl(99, "crown", 60, NOW + timedelta(days=21)), P(2, "Frank", ["crown"]))]
    plan = plan_recovery(slot, wl, booked, REL, now=NOW)
    kinds = [b.kind for b in plan.bookings]
    assert "pull_forward" in kinds
    assert plan.bookings[0].patient_id == 2


def test_cascade_fills_the_vacated_slot():
    slot = Sl(1, "crown", 60, NOW + timedelta(hours=5))
    wl = [P(1, "Greta", ["crown"], short_notice=False)]   # only fits the far slot
    booked = [(Sl(99, "crown", 60, NOW + timedelta(days=21)), P(2, "Frank", ["crown"]))]
    plan = plan_recovery(slot, wl, booked, REL, now=NOW)
    assert plan.cascade_count == 1
    assert plan.patients_helped == 2          # Frank pulled up + Greta into the freed slot


def test_money_is_not_prioritised_over_waiting():
    slot = Sl(1, "crown", 60, NOW + timedelta(hours=5))
    wl = [P(1, "Hans", ["crown"], days=3), P(2, "Ida", ["cleaning"], days=40)]
    plan = plan_recovery(slot, wl, [], REL, now=NOW)
    # the 40-day waiter (cheap cleaning) must be called before the fresh €600 crown
    assert plan.bookings[0].patient_id == 2


def test_weeks_ahead_pull_forward_is_same_type_only():
    slot = Sl(1, "cleaning", 30, NOW + timedelta(hours=5))
    # a filling booked 21 days out would fit the 30-min slot? no (45>30); and a
    # cleaning booked far out is same-type → eligible.
    booked = [(Sl(98, "cleaning", 30, NOW + timedelta(days=25)), P(3, "Pat", ["cleaning"]))]
    plan = plan_recovery(slot, [], booked, REL, now=NOW)
    assert plan.bookings and plan.bookings[0].patient_id == 3
