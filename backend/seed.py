"""Synthetic data generator: ~30 waitlist patients + 1 week of slots.

Top-5 are hand-tuned for the demo narrative (§3 of PLAN.md):
- one obvious winner (perfect window match, high attendance, short-notice ok)
- one consent=false (visible hard skip)
- one in cooldown (called yesterday)
- one long-waiter (>30 days, starvation boost)
- one with treatment mismatch (visible hard skip)

Run: `python -m backend.seed`
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, time

from backend.db import connect, init_db, reset_db

TREATMENTS = [
    ("cleaning", 80),
    ("checkup", 60),
    ("filling", 150),
    ("crown", 600),
]

# Hand-tuned narrative patients (ids 1..5). Phones use the +43 (AT) prefix
# but are deliberately not real numbers — wired to mock fonio by default.
NARRATIVE_PATIENTS = [
    # 1: hero candidate — should win the demo
    dict(
        id=1, name="Maria Huber", phone="+4366012340001", age=34,
        sms_opt_in=1, hypertension=0, diabetes=0,
        consent_outbound=1, short_notice_ok=1,
        preferred_window_start="08:00", preferred_window_end="19:00",
        needed_treatments=["cleaning", "checkup"],
        days_waiting=22, attendance_history=[1, 1, 1, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None,
    ),
    # 2: consent=false — must be hard-skipped, reason visible
    dict(
        id=2, name="Lukas Berger", phone="+4366012340002", age=41,
        sms_opt_in=0, hypertension=1, diabetes=0,
        consent_outbound=0, short_notice_ok=1,
        preferred_window_start="08:00", preferred_window_end="19:00",
        needed_treatments=["cleaning"],
        days_waiting=6, attendance_history=[1, 1, 1, 0, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None,
    ),
    # 3: in cooldown — called yesterday
    dict(
        id=3, name="Sophie Wagner", phone="+4366012340003", age=29,
        sms_opt_in=1, hypertension=0, diabetes=0,
        consent_outbound=1, short_notice_ok=1,
        preferred_window_start="09:00", preferred_window_end="17:00",
        needed_treatments=["cleaning", "filling"],
        days_waiting=18, attendance_history=[1, 1, 1, 1, 0],
        last_offer_called_at=None,  # filled at runtime to "yesterday"
        last_decline_at=None, last_declined_slot_type=None,
    ),
    # 4: long-waiter — starvation guard should boost
    dict(
        id=4, name="Johann Steiner", phone="+4366012340004", age=58,
        sms_opt_in=1, hypertension=1, diabetes=1,
        consent_outbound=1, short_notice_ok=1,
        preferred_window_start="08:00", preferred_window_end="19:00",
        needed_treatments=["crown", "filling"],
        days_waiting=42, attendance_history=[1, 1, 0, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None,
    ),
    # 5: treatment mismatch — hard skip "Treatment mismatch"
    dict(
        id=5, name="Anna Fischer", phone="+4366012340005", age=46,
        sms_opt_in=1, hypertension=0, diabetes=0,
        consent_outbound=1, short_notice_ok=1,
        preferred_window_start="08:00", preferred_window_end="19:00",
        needed_treatments=["crown"],  # mismatches the demo cleaning slot
        days_waiting=8, attendance_history=[1, 1, 1, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None,
    ),
]

FIRST_NAMES = [
    "Felix", "Hanna", "Paul", "Lea", "Jonas", "Mia", "Elias", "Sara",
    "Noah", "Klara", "Tobias", "Emma", "Niklas", "Lara", "David",
    "Julia", "Simon", "Nora", "Florian", "Theresa", "Patrick", "Magdalena",
    "Manuel", "Katharina", "Stefan",
]
LAST_NAMES = [
    "Bauer", "Schmid", "Gruber", "Mayer", "Hofer", "Wimmer", "Pichler",
    "Auer", "Reiter", "Eder", "Lechner", "Brunner", "Moser", "Lang",
    "Zimmermann", "Schneider", "Weber", "Fuchs", "Lehner", "Holzer",
    "Riedl", "Aigner", "Strasser", "Köhler", "Kainz",
]


def _isoz(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def make_random_patients(start_id: int, n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        treatments = rng.sample(
            [t for t, _ in TREATMENTS], rng.choice([1, 1, 2])
        )
        history = [rng.choices([1, 0], weights=[0.85, 0.15])[0] for _ in range(5)]
        last_decline_at = None
        last_decline_type = None
        if rng.random() < 0.15:
            last_decline_at = _isoz(datetime.now() - timedelta(days=rng.randint(8, 30)))
            last_decline_type = rng.choice([t for t, _ in TREATMENTS])
        out.append(dict(
            id=start_id + i,
            name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            phone=f"+436601234{6000 + start_id + i:04d}",
            age=rng.randint(18, 78),
            sms_opt_in=rng.choices([1, 0], weights=[0.8, 0.2])[0],
            hypertension=rng.choices([0, 1], weights=[0.75, 0.25])[0],
            diabetes=rng.choices([0, 1], weights=[0.88, 0.12])[0],
            consent_outbound=rng.choices([1, 0], weights=[0.88, 0.12])[0],
            short_notice_ok=rng.choices([1, 0], weights=[0.7, 0.3])[0],
            preferred_window_start=rng.choice(["08:00", "09:00", "10:00"]),
            preferred_window_end=rng.choice(["16:00", "17:00", "18:00", "19:00"]),
            needed_treatments=treatments,
            days_waiting=rng.randint(2, 28),
            attendance_history=history,
            last_offer_called_at=None,
            last_decline_at=last_decline_at,
            last_declined_slot_type=last_decline_type,
        ))
    return out


def make_slots(now: datetime, rng: random.Random) -> list[dict]:
    """One week of slots: today's middle is the demo slot (cancel target)."""
    slots = []
    sid = 1
    # Today: a cleaning at +90 minutes (URGENT/CRITICAL boundary), booked,
    # ready to be cancelled by the demo button.
    today_demo = now.replace(hour=14, minute=30, second=0, microsecond=0)
    if today_demo <= now + timedelta(minutes=20):
        today_demo = now + timedelta(minutes=90)
    slots.append(dict(
        id=sid, start_dt=_isoz(today_demo), duration_min=30,
        type="cleaning", value_eur=80, status="booked", booked_patient_id=None,
        lead_days=6,  # booked ~a week ahead → modest no-show risk
    ))
    sid += 1
    # Today: a long-lead appointment → high no-show risk (the "flagged red" slot
    # in the demo) and the ESCALATED target after its cancellation.
    later = now.replace(hour=17, minute=0, second=0, microsecond=0)
    if later <= now:
        later = now + timedelta(hours=3)
    slots.append(dict(
        id=sid, start_dt=_isoz(later), duration_min=30,
        type="crown", value_eur=600, status="booked", booked_patient_id=None,
        lead_days=64,  # booked ~2 months ahead → high predicted no-show risk
    ))
    sid += 1
    # Remaining week, 6 slots/day, all booked, with a realistic spread of booking
    # horizons so the no-show risk badges vary across the schedule.
    for d in range(1, 6):
        day = (now + timedelta(days=d)).replace(hour=8, minute=0, second=0, microsecond=0)
        for h in range(6):
            tt, val = rng.choice(TREATMENTS)
            slots.append(dict(
                id=sid,
                start_dt=_isoz(day + timedelta(hours=h * 1, minutes=rng.choice([0, 30]))),
                duration_min=rng.choice([30, 45, 60]),
                type=tt, value_eur=val,
                status="booked", booked_patient_id=None,
                lead_days=rng.choice([1, 2, 3, 5, 8, 14, 21, 30, 45, 60]),
            ))
            sid += 1
    return slots


def seed(rng_seed: int = 42, n_random_patients: int = 25) -> None:
    reset_db()
    rng = random.Random(rng_seed)
    now = datetime.now()

    patients = list(NARRATIVE_PATIENTS)
    # Mark patient 3 as "called yesterday" so the cooldown filter fires.
    for p in patients:
        if p["id"] == 3:
            p["last_offer_called_at"] = _isoz(now - timedelta(hours=20))
    patients += make_random_patients(start_id=6, n=n_random_patients, rng=rng)
    slots = make_slots(now, rng)

    conn = connect()
    try:
        conn.executemany(
            "INSERT INTO treatments(type, value_eur) VALUES (?, ?)",
            TREATMENTS,
        )
        conn.executemany(
            """INSERT INTO patients(
                id, name, phone, age, sms_opt_in, hypertension, diabetes,
                consent_outbound, short_notice_ok,
                preferred_window_start, preferred_window_end,
                needed_treatments, days_waiting, attendance_history,
                last_offer_called_at, last_decline_at, last_declined_slot_type
            ) VALUES (
                :id, :name, :phone, :age, :sms_opt_in, :hypertension, :diabetes,
                :consent_outbound, :short_notice_ok,
                :preferred_window_start, :preferred_window_end,
                :needed_treatments_json, :days_waiting, :attendance_history_json,
                :last_offer_called_at, :last_decline_at, :last_declined_slot_type
            )""",
            [
                {
                    **p,
                    "needed_treatments_json": json.dumps(p["needed_treatments"]),
                    "attendance_history_json": json.dumps(p["attendance_history"]),
                }
                for p in patients
            ],
        )
        conn.executemany(
            """INSERT INTO slots(id, start_dt, duration_min, type, value_eur,
                                 status, booked_patient_id, lead_days)
               VALUES (:id, :start_dt, :duration_min, :type, :value_eur,
                       :status, :booked_patient_id, :lead_days)""",
            slots,
        )
        # Book some random patients into the rest of the week's slots so they
        # look like "patients we'd be calling about cancellations".
        booked_pids = rng.sample(range(6, 6 + n_random_patients), k=min(8, n_random_patients))
        for i, sid in enumerate(s["id"] for s in slots[2:]):
            if i < len(booked_pids):
                conn.execute(
                    "UPDATE slots SET booked_patient_id = ? WHERE id = ?",
                    (booked_pids[i], sid),
                )
    finally:
        conn.close()

    print(f"Seeded {len(patients)} patients, {len(slots)} slots.")


if __name__ == "__main__":
    init_db()
    seed()
