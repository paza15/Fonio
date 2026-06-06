"""Tests for the reliability serving layer — bounds, the heuristic fallback,
and that attendance history moves the prediction in the right direction
(validates the train→serve feature wiring)."""

from __future__ import annotations

from backend import reliability
from conftest import make_patient


def test_predict_bounds():
    p = make_patient()
    assert 0.05 <= reliability.predict(p) <= 0.95


def test_attendance_history_is_monotonic():
    """A patient who always shows must score higher than one who never shows.
    Holds for both the trained model and the heuristic fallback."""
    always = make_patient(attendance_history=[1, 1, 1, 1, 1])
    never = make_patient(attendance_history=[0, 0, 0, 0, 0])
    assert reliability.predict(always) > reliability.predict(never)


def test_heuristic_used_without_model(monkeypatch):
    monkeypatch.setattr(reliability, "_model", None)
    p_healthy = make_patient(attendance_history=[1, 1, 1, 1, 1], diabetes=False, hypertension=False)
    p_sick = make_patient(attendance_history=[1, 1, 1, 1, 1], diabetes=True, hypertension=True)
    assert reliability.predict(p_healthy) > reliability.predict(p_sick)
    assert 0.05 <= reliability.predict(p_sick) <= 0.95


def test_lead_days_increases_no_show_risk():
    """Longer booking horizon ⇒ lower show probability (model's top feature).
    Only meaningful with the trained model loaded; skip on the heuristic."""
    if reliability._model is None:
        return
    p = make_patient()
    same_day = reliability.predict(p, lead_days=0)
    far_out = reliability.predict(p, lead_days=60)
    assert same_day > far_out
