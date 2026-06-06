"""Demo: capacity-aware recovery — packing, pull-forward, cascade — with scores.

Pure planner, synthetic inputs, no DB. Run: python -m scripts.demo_scheduling
"""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.scheduling import plan_recovery
from backend.scoring import Patient, Slot

NOW = datetime.now().replace(microsecond=0)
VALUES = {"cleaning": 80, "checkup": 60, "filling": 150, "crown": 600}
REL = lambda p, lead=0.0: 0.85   # flat reliability — this demo is about scheduling


def P(pid, name, treatments, days=10, consent=True, short_notice=True):
    return Patient(
        id=pid, name=name, phone=f"+43000{pid:05d}", age=35, sms_opt_in=True,
        hypertension=False, diabetes=False, consent_outbound=consent,
        short_notice_ok=short_notice, preferred_window_start=time(8, 0),
        preferred_window_end=time(19, 0), needed_treatments=list(treatments),
        days_waiting=days, attendance_history=[1, 1, 1, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None)


def S(sid, ttype, minutes, when):
    return Slot(id=sid, start=when, duration_min=minutes, type=ttype, value_eur=VALUES[ttype])


def show(plan, indent="  "):
    for b in plan.bookings:
        tag = "PULL-FORWARD" if b.kind == "pull_forward" else "waitlist"
        extra = f"  → frees slot {b.from_slot_id}, cascading" if b.from_slot_id else ""
        print(f"{indent}+ {b.patient_name:8s} {b.treatment:8s} {b.minutes:>2}min  €{b.value_eur:<4} [{tag}]{extra}")
    for c in plan.cascades:
        print(f"{indent}  └─ cascade fills the vacated slot {c.slot_id}:")
        show(c, indent + "       ")


def scores(plan):
    print(f"  → SCORES: utilization {plan.utilization:.0%} ({plan.filled_min}/{plan.capacity_min} min) · "
          f"€{plan.recovered_eur} recovered · {plan.patients_helped} patients helped · "
          f"{plan.cascade_count} cascade(s)\n")


print("=" * 72)
print(" SCENARIO 1 — duration packing  (a 60-min crown slot, no crown taker)")
print("=" * 72)
slot1 = S(1, "crown", 60, NOW + timedelta(hours=5))
wl1 = [P(11, "Anna", ["cleaning"], days=35), P(12, "Ben", ["checkup"], days=22),
       P(13, "Carla", ["crown"], consent=False)]
print("  freed: 60-min crown slot · waitlist: Anna(cleaning), Ben(checkup), Carla(crown,no-consent)")
p1 = plan_recovery(slot1, wl1, [], REL, now=NOW)
show(p1); scores(p1)

print("=" * 72)
print(" SCENARIO 2 — pull-forward + cascade")
print("=" * 72)
slot2 = S(1, "crown", 60, NOW + timedelta(hours=5))   # today ⇒ short notice
wl2 = [P(21, "Greta", ["crown"], days=30, short_notice=False)]   # can't do today
booked2 = [(S(99, "crown", 60, NOW + timedelta(days=21)), P(22, "Frank", ["crown"]))]
print("  freed: 60-min crown today · Greta wants a crown but can't do short notice ·")
print("  Frank is booked for a crown 21 days out")
p2 = plan_recovery(slot2, wl2, booked2, REL, now=NOW)
show(p2); scores(p2)

print("=" * 72)
print(" SCENARIO 3 — fairness: long-waiter beats fresh high-value patient")
print("=" * 72)
slot3 = S(1, "crown", 60, NOW + timedelta(hours=5))
wl3 = [P(31, "Hans", ["crown"], days=3), P(32, "Ida", ["cleaning"], days=40)]
print("  freed: 60-min crown · Hans(crown €600, 3d waiting) vs Ida(cleaning €80, 40d waiting)")
p3 = plan_recovery(slot3, wl3, [], REL, now=NOW)
show(p3)
first = p3.bookings[0].patient_name if p3.bookings else "—"
print(f"  → first call goes to: {first}  (money is a ±10% tiebreaker, not a driver)\n")
