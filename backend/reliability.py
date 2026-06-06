"""Reliability model wrapper — P(patient shows & follows through).

If `ml/reliability_model.pkl` exists (LightGBM + isotonic calibration from
ml/train_reliability.py) we use it; otherwise a deterministic heuristic keeps
the demo working.

The model expects engineered patient-history features (prior_no_show_rate,
prior_visits, …). We derive those at serving time from the patient's stored
`attendance_history` (last-5 show/no-show flags) using the SAME smoothing
constants the model was trained with — those constants travel in the model
bundle (`base_rate`, `alpha`, `window`) so training and serving never drift.

Older bundles without the history features still work: we build every feature
into a dict and select only the ones the bundle's `feature_order` asks for.
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
_base_rate = 0.20      # global no-show base rate; overwritten from the bundle
_alpha = 2.0           # Bayesian smoothing pseudo-count; overwritten from bundle
_window = 5            # trailing attendance window; overwritten from the bundle


def load() -> None:
    global _model, _feature_order, _base_rate, _alpha, _window
    if not _MODEL_PATH.exists():
        _model = None
        _feature_order = []
        return
    bundle = joblib.load(_MODEL_PATH)
    _model = bundle["model"]
    _feature_order = bundle["feature_order"]
    _base_rate = float(bundle.get("base_rate", 0.20))
    _alpha = float(bundle.get("alpha", 2.0))
    _window = int(bundle.get("window", 5))


def _heuristic(p: Patient) -> float:
    history = p.attendance_history or [1, 1, 1, 1, 1]
    base = sum(history) / max(len(history), 1)
    age_pen = 0.05 if (p.age < 25 or p.age > 70) else 0.0
    pen = 0.05 * int(p.diabetes) + 0.05 * int(p.hypertension) + age_pen
    return max(0.05, min(0.95, base - pen))


def _features(p: Patient, lead_days: float) -> dict:
    """Build every model feature from a Patient. attendance_history holds
    show flags (1 = showed); we need the no-show rate over the trailing window."""
    window = (p.attendance_history or [])[-_window:]
    visits = len(window)
    noshow_in_window = visits - sum(window)            # 0 = always showed
    prior_no_show_rate = (noshow_in_window + _alpha * _base_rate) / (visits + _alpha)
    return {
        "Age": p.age,
        "lead_days": max(lead_days, 0.0),
        "same_day": 1 if lead_days < 1 else 0,         # same calendar day as booking
        "SMS_received": int(p.sms_opt_in),
        "Hipertension": int(p.hypertension),
        "Diabetes": int(p.diabetes),
        "Scholarship": 0,                              # not tracked locally
        "prior_visits": float(visits),
        "prior_no_show_rate": prior_no_show_rate,
        "is_first_visit": 1 if visits == 0 else 0,
    }


def predict(p: Patient, lead_days: float = 0.0) -> float:
    """Returns P(showed_up) in [0.05, 0.95]."""
    if _model is None:
        return _heuristic(p)
    feats = _features(p, lead_days)
    x = np.array([[feats[k] for k in _feature_order]])
    # Kaggle target is 1 = no-show; we want P(showed) = P(class 0).
    proba = _model.predict_proba(x)[0]
    classes = list(_model.classes_)
    show = proba[classes.index(0)] if 0 in classes else 1.0 - proba[classes.index(1)]
    return float(max(0.05, min(0.95, show)))


load()
