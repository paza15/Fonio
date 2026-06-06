# Model Card — No-Show Reliability Model

`reliability_model.pkl` predicts **P(a patient shows up & follows through)** for a
given appointment. It powers (a) the no-show risk badge on booked slots and
(b) the reliability term in the waitlist ranking (PLAN §5.1 / §5.4).

## Intended use
Rank waitlisted patients for last-minute slot refills and flag at-risk booked
appointments in a single dental practice. **Decision-support only** — a human
receptionist stays in the loop; the model never auto-cancels or auto-books.

## Data
- **Source:** Kaggle "Medical Appointment No Shows" (`joniarroba/noshowappointments`,
  `KaggleV2-May-2016.csv`), Brazil public health, 2016.
- **Rows:** 110,527 → **110,521** after dropping `Age < 0` or `Age > 110`.
- **Target:** `No-show` = "Yes" → `1` (no-show). Base rate **20.2%**.
- **Repeat patients:** 62,299 unique patients across the appointments; 24,378
  patients have >1 visit. 44% of appointments have a prior history to learn from.

## Features (all reproducible at serving time — no train/serve skew)
| Feature | Source | Notes |
|---|---|---|
| `Age` | patient | |
| `lead_days` | AppointmentDay − ScheduledDay | clipped ≥ 0; **top feature (~85% gain)** |
| `same_day` | lead_days == 0 | derived |
| `SMS_received` | patient `sms_opt_in` | |
| `Hipertension`, `Diabetes` | patient | |
| `Scholarship` | 0 at serving (not tracked locally) | ~0 importance |
| `prior_no_show_rate` | **reconstructed** from repeat `PatientId`s | trailing-5, past-only, Bayesian-smoothed |
| `prior_visits` | reconstructed | trailing-5 window (matches the app's last-5) |
| `is_first_visit` | reconstructed | cold-start flag |

**Reconstruction (leakage-safe):** for each appointment, ordered by patient then
schedule time, we use **only that patient's strictly earlier appointments**. The
no-show rate is smoothed toward the global base:
`rate = (past_no_shows + α·base) / (prior_visits + α)`, `α = 2`, `base = 0.202`.
`base` and `α` are saved in the model bundle so `backend/reliability.py` maps our
stored `attendance_history` into the identical feature.

## Training
- LightGBM (`max_depth=4`, `lr=0.05`, `class_weight="balanced"`) + isotonic
  calibration (`CalibratedClassifierCV`, prefit on validation).
- **Split: 60/20/20 stratified** — train grows the trees, **validation** drives
  early stopping (best_iteration ≈ 213) + calibration, **test** is held out.

## Results
| Metric | Value |
|---|---|
| **Test ROC-AUC** | **0.742** |
| Validation ROC-AUC | 0.740 |
| Train ROC-AUC | 0.755 |

Tight train/val/test spread ⇒ not overfit; well under the 0.9 leakage tripwire.

**Why ROC-AUC, not accuracy:** 80% of patients show up, so "always predict show"
scores 80% accuracy while being useless. AUC measures ranking skill.

### Feature ablation (5-fold CV, LightGBM)
| Features | ROC-AUC |
|---|---|
| baseline (6) | 0.729 |
| + reconstructed attendance history (10) | **0.744** |
| **lift** | **+0.015** |

### Model comparison (5-fold CV, full features)
| Model | ROC-AUC |
|---|---|
| XGBoost | 0.7446 |
| LightGBM (production) | 0.7441 |
| RandomForest | 0.7371 |
| AdaBoost | 0.7314 |

XGBoost and LightGBM tie within fold noise (±0.003). The ~0.74 ceiling is set by
the **features, not the algorithm** — we ship LightGBM (faster, native class
weights, already calibrated).

## Learned scoring priors (`ml/tune_priors.py`)
Beyond the model, `rank()` shrinks each patient's accept/answer prior toward
their **observed** call-log rate: `p = (prior·k + successes) / (k + trials)`.
The strength `k` was tuned, not guessed — by predictive log-loss on 48,225
leakage-safe Kaggle repeat-patient sequences (predict each appointment's outcome
from the patient's trailing-5 prior outcomes):

| k | log-loss | Brier |
|---|---|---|
| 0 (raw empirical) | 2.876 | 0.243 |
| 4 | 0.4777 | 0.1517 |
| **6 (chosen)** | **0.4755** | **0.1507** |
| 8 | 0.4758 | 0.1508 |
| ∞ (ignore patient) | 0.4917 | 0.1562 |

Flat optimum at **k=6**: raw rates overfit tiny samples (log-loss 2.88),
ignoring the patient signal costs +0.016 log-loss. Used for both
`ANSWER_PRIOR_STRENGTH` and `ACCEPT_PRIOR_STRENGTH` (same repeated-binary
structure; the accept side awaits real call data to retune directly).

## Limitations & honesty
- **Domain transfer:** trained on Brazilian public-health data; it *demonstrates
  the method*. A real deployment retrains on the clinic's own history, which
  natively provides attendance + neighbourhood signals.
- **First-visit cold start:** 56% of rows have no prior history → fall back to the
  smoothed base rate. The attendance lift is concentrated on returning patients
  (which is most of a dental waitlist, so it matters more in production than the
  44% coverage here implies).
- **Dropped signals with no serving analog** (Neighbourhood, Gender, weekday) were
  intentionally excluded to avoid train/serve skew; they're the next lift if the
  production schema carries them.
- **License:** the dataset is third-party Kaggle data — see its Kaggle page for
  terms before redistribution.

## Reproduce
```bash
python -m ml.train_reliability    # → reliability_model.pkl + metrics.json
python -m ml.benchmark 5          # → benchmark.json (models + feature ablation)
python -m pytest tests/ -q        # scoring + serving unit tests
```
