# Slot Refill Engine — Status Report

**Project:** when a dental appointment is cancelled, rank the waitlist, have an AI
voice agent (fonio) call candidates one by one, and book the slot — closing the
loop in under a minute, every decision explainable and fair.

**Hackathon:** fonio.ai track, START Hack Vienna '26. This report = what's built,
what's measured, and what's left (fonio + dashboard).

---

## 1. What's done ✅

### Backend engine (FastAPI + SQLite)
- **API:** `/rank`, `/outcome`, `/state`, `/simulate/{cancel,no_show,confirmation}`,
  `/webhook/fonio`, `/healthz`. Verified running.
- **Recovery state machine** ([orchestrator.py](backend/orchestrator.py)) with hard
  guarantees: **idempotent** (`UNIQUE(slot_id)` + in-flight set), **strictly
  sequential** dialing, **per-patient lock** (no double-booking), **call window**
  (08:00–19:00), **90 s webhook timeout → voicemail**, **<20 min → unrecoverable**,
  **waitlist exhausted → escalate to a human**.
- **Persistence:** patients, slots, recovery_attempts, calls — survives restart.
- **Tests:** **31 tests** (incl. live-orchestrator packing + pull-forward/cascade
  integration tests) + an end-to-end smoke test, all green.

### ML — no-show / reliability model
- Trained on the real Kaggle "Medical Appointment No Shows" data (110,521 rows),
  **60/20/20 train/val/test**, LightGBM + isotonic calibration.
- **Held-out test ROC-AUC = 0.742.** (Never accuracy — 20% base rate.)
- **Reconstructed attendance history** from repeat patients (leakage-safe,
  trailing-5): **+0.015 AUC** over baseline (ablation, 5-fold). The raw file has
  no attendance column; we rebuilt it.
- **Model benchmark:** XGBoost 0.745 ≈ LightGBM 0.744 > RandomForest 0.737 >
  AdaBoost 0.731 — model choice is noise; features are the lever. Ship LightGBM.
- Artifacts: `reliability_model.pkl`, `metrics.json`, `benchmark.json`,
  [MODEL_CARD.md](ml/MODEL_CARD.md).

### Scoring & ranking (the brains)
- **Hard filters with visible reasons** (consent, cooldown, recently-declined,
  weekly cap, fit) — the fairness/consent story.
- **deadline-aware priority** with urgency + starvation guard (long-waiters boosted).
- **Self-learning signal:** P(answer) / P(accept) are shrunk toward each patient's
  real **call-log** history (Bayesian). The engine learns who actually responds
  from its own outcomes. Prior strength **tuned to k=6** by predictive log-loss on
  48k sequences ([tune_priors.py](ml/tune_priors.py)) — not guessed.
- **Ethics — money is the least factor:** treatment value swings the score by at
  most **±10%**; a 40-day-waiter on a cheap cleaning beats a fresh €600 crown.
- **Capacity-aware recovery — now LIVE in the orchestrator** (pure planner in
  [scheduling.py](backend/scheduling.py) + wired into [orchestrator.py](backend/orchestrator.py)):
  - **Duration packing** — fill one long freed slot with several short treatments
    (leftover time spins off as its own recovery).
  - **Pull-forward** — if the waitlist can't fill it, pull a patient booked *later*
    into the earlier slot (nearby: any fitting treatment; weeks-ahead: same type only).
  - **Cascade** — the slot they vacate is recovered again (bounded by MAX_RECOVERY_DEPTH).
  - **No double-booking:** Tier 1 ranks only true waitlist patients (no current
    appointment); pull-forward targets booked patients.
  - Proven end-to-end by live integration tests; planner also returns scores
    (utilization, € recovered, patients helped, cascades).

### Demonstrations (`python -m scripts.<name>`)
- `smoke_test` — end-to-end cancel → rank → call → book.
- `demo_learning` — a chronic decliner / never-answers patient drop out of the top-5.
- `demo_scheduling` — packing (100% util, 2 patients), pull-forward+cascade
  (2 patients, 1 cascade), fairness (long-waiter beats high-value).

---

## 2. Will fonio and the dashboard be done?

**Both are still TODO.** Honest status:

### fonio integration — *coming (you're driving this)*
- The **verified** public API is documented (PLAN §6): `POST /public/v1/outbound_call`
  with `{fromNumber, toNumber, context}`; **no call id in the response** → correlate
  via `context`; post-call webhook delivers the agent's defined variables.
- **What's left:** update `backend/fonio_client.py` (`RealFonioClient`) + `.env` to
  that exact shape (it currently has the old assumed shape), then one real
  end-to-end call. This is **code-ready** — no creds needed to write it; needs an
  imported number + Teams plan + minutes to *run*. The whole orchestrator already
  works against the mock client, so this is a swap, not a rebuild.
- **Risk:** highest-weighted judging item (working real call = 30%) — do it first
  once credentials land.

### Dashboard — *not started (backend is ready)*
- `/state` already serves everything a dashboard needs: today's schedule with
  status + no-show risk badges, the live recovery panel (current call, ranked list
  with breakdowns, **skipped-with-reasons**), and metric tiles (refill %, €
  recovered, time-to-fill, outcomes).
- **What's left:** a single polling page (2 s) — React or plain HTML+fetch.
- **Weight:** 15% of judging; it's the biggest *unbuilt* scoring item.

### Other open items
- ~~Wire the planner into the live orchestrator~~ — **done** (packing + pull-forward
  + cascade run in the live recovery loop; covered by live integration tests).
- **README** (setup/run + real-vs-mocked table), 3-min video, Tally form (§11).

---

## 3. How to run
```bash
pip install -r requirements.txt
python -m backend.seed                 # seed the demo DB
python -m scripts.smoke_test           # end-to-end (mock fonio)
python -m scripts.demo_scheduling      # packing / pull-forward / cascade scores
python -m ml.train_reliability         # retrain (needs data/kaggle/*.csv)
python -m pytest tests/ -q             # 29 tests
uvicorn backend.main:app               # the API
```

## 4. Scoreboard
| Area | Result |
|---|---|
| No-show model (held-out test AUC) | **0.742** |
| Attendance-feature lift | +0.015 AUC |
| Prior-strength tuning | k=6 (log-loss optimum) |
| Unit tests | 29 passing |
| Packing demo | 100% utilization, 2 patients / slot |
| Pull-forward demo | 2 patients helped per cancellation (1 cascade) |
| Ethics | money ≤ ±10% of score |
