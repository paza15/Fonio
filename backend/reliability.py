"""Reliability model wrapper.

If `ml/reliability_model.pkl` exists (LightGBM + CalibratedClassifierCV from
ml/train_reliability.py), use it. Otherwise fall back to a deterministic
heuristic so the demo still works:

    P(answer & follow through) ≈ base_attendance × age_smoothing
                                  − 0.05 × diabetes − 0.05 × hypertension

This fallback is documented in the README's real-vs-mocked table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from backend.scoring import Patient

_MODEL_PATH = Path("ml/reliability_model.pkl")
_model = None
_feature_order: list[str] = []


def load() -> None:
    global _model, _feature_order
    if not _MODEL_PATH.exists():
        _model = None
        _feature_order = []
        return
    bundle = joblib.load(_MODEL_PATH)
    _model = bundle["model"]
    _feature_order = bundle["feature_order"]


def _heuristic(p: Patient) -> float:
    history = p.attendance_history or [1, 1, 1, 1, 1]
    base = sum(history) / max(len(history), 1)
    # gentle age smoothing — kids and very elderly miss more
    age_pen = 0.0
    if p.age < 25:
        age_pen = 0.05
    if p.age > 70:
        age_pen = 0.05
    pen = 0.05 * int(p.diabetes) + 0.05 * int(p.hypertension) + age_pen
    return max(0.05, min(0.95, base - pen))


def predict(p: Patient, lead_days: float = 0.0) -> float:
    """Returns P(showed_up) in [0.05, 0.95]."""
    if _model is None:
        return _heuristic(p)
    # Map our Patient to Kaggle features (§5.1).
    row = {
        "Age": p.age,
        "lead_days": max(lead_days, 0.0),
        "SMS_received": int(p.sms_opt_in),
        "Hipertension": int(p.hypertension),
        "Diabetes": int(p.diabetes),
        "Scholarship": 0,
    }
    x = np.array([[row[k] for k in _feature_order]])
    # Kaggle's "No-show" target is 1=no-show. We predict P(showed) = P(class=0).
    proba = _model.predict_proba(x)[0]
    classes = list(_model.classes_)
    if 0 in classes:
        show = proba[classes.index(0)]
    else:
        show = 1.0 - proba[classes.index(1)]
    return float(max(0.05, min(0.95, show)))


load()
