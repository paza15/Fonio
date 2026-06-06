"""Make the repo root importable (so `import backend...` works) regardless of
how pytest is invoked, and provide shared fixtures/builders."""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.scoring import Patient, Slot  # noqa: E402


def make_patient(**over) -> Patient:
    """A reliable, eager patient by default; override fields per test."""
    base = dict(
        id=1, name="Test Patient", phone="+430000000000", age=35,
        sms_opt_in=True, hypertension=False, diabetes=False,
        consent_outbound=True, short_notice_ok=True,
        preferred_window_start=time(8, 0), preferred_window_end=time(19, 0),
        needed_treatments=["cleaning"], days_waiting=10,
        attendance_history=[1, 1, 1, 1, 1],
        last_offer_called_at=None, last_decline_at=None, last_declined_slot_type=None,
    )
    base.update(over)
    return Patient(**base)


def make_slot(**over) -> Slot:
    base = dict(
        id=1, start=datetime.now() + timedelta(hours=48),
        duration_min=30, type="cleaning", value_eur=80, lead_days=0,
    )
    base.update(over)
    return Slot(**base)
