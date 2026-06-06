# PLAN.md — Slot Refill Engine (fonio.ai Track, START Hack Vienna '26)

**One-liner:** When an appointment is cancelled (or predicted to no-show), the system ranks the waitlist, has fonio call candidates one by one, and books the slot — closing the loop in under a minute, with every decision explainable on a live dashboard.

**Context:** Dental practice. Language: English (calls + UI). One practice, sandbox/mocked integrations (labeled). Code freeze: **Sunday June 7, 14:00**.

**Judging weights to optimize for:** working end-to-end demo with a real call (30%), solid engineering — persistence, error handling, idempotency (25%), problem fit (20%), usable dashboard (15%), pitch (10%).

---

## 1. Team Split

- **Track A (Models pair):** Kaggle model, synthetic data generator, FastAPI scoring backend (`/rank`, `/outcome`), then dashboard.
- **Track B (fonio pair):** fonio agents (offer call + confirmation call), outbound trigger, post-call webhook, orchestrator state machine.
- **Interface between tracks:** the JSON contract in §4. Do not change it without telling the other pair.

---

## 2. Architecture

```
Cancellation sources                    ┌──────────────────────┐
 • inbound cancel (fonio / simulated)   │  ORCHESTRATOR (B)    │
 • confirmation call says "not coming"──►  state machine per   │
 • dashboard "cancel/no-show" button    │  slot recovery       │
                                        └──────┬───────────────┘
                                               │ POST /rank
                                        ┌──────▼───────────────┐
                                        │  SCORING BACKEND (A) │
                                        │  filters → score →   │
                                        │  ranked top-5 + skips│
                                        └──────┬───────────────┘
                                               │ candidate
                                        ┌──────▼───────────────┐
                                        │  fonio OUTBOUND CALL │
                                        │  post-call webhook → │
                                        │  outcome classified  │
                                        └──────┬───────────────┘
                                               │ POST /outcome
                            booked → slot filled, dashboard green
                            declined/voicemail → next candidate
                            list empty → ESCALATED to receptionist
        All state changes → DASHBOARD (live panel + metrics tiles)
```

---

## 3. Data Model (SQLite — persistence is a judging criterion)

```
patients(id, name, phone, age, sms_opt_in, hypertension, diabetes,
         consent_outbound BOOL, short_notice_ok BOOL,
         preferred_window_start, preferred_window_end,
         needed_treatments JSON,            -- ["cleaning","crown",...]
         days_waiting INT,
         attendance_history JSON,           -- [1,1,0,1,1] last 5, 1=showed
         last_offer_called_at, last_decline_at)

slots(id, start_dt, duration_min, type, value_eur,
      status: booked|cancelled|recovering|filled|escalated|unrecoverable,
      booked_patient_id)

recovery_attempts(id, slot_id UNIQUE,      -- UNIQUE = idempotency guard
      created_at, status, filled_by_patient_id)

calls(id, fonio_call_id, recovery_attempt_id, patient_id,
      direction, outcome: booked|declined|voicemail|callback|timeout,
      summary, started_at, ended_at)

treatments(type, value_eur)                -- cleaning 80, checkup 60,
                                           -- filling 150, crown 600
```

Synthetic data: ~30 waitlist patients, 1 week of slots. Hand-tune the top 5 patients for the demo narrative. Include: 2–3 with `consent_outbound=false`, one in cooldown (`last_offer_called_at` = yesterday), one long-waiter (>30 days), realistic age spread matching Kaggle distribution.

---

## 4. API Contract (Track A serves, Track B consumes)

```
POST /rank
{ "slot": {"id","start","duration_min","type","value_eur"},
  "exclude_patient_ids": [] }
→ { "candidates": [ {"patient_id","name","phone","score",
       "breakdown": {"answer_prob","accept_score","value_norm","phase"},
       "reason": "<one sentence>"} ],          // max 5, sorted desc
    "skipped": [ {"patient_id","reason"} ] }   // shown on dashboard

POST /outcome
{ "slot_id","patient_id","outcome" }   // booked|declined|voicemail|callback
→ 200; logs call, updates attendance_history + cooldowns,
       on "booked" marks slot filled
```

---

## 5. ML & Scoring (Track A)

### 5.1 Reliability model — P(answer & follow through)  ✅ implemented
- **Dataset:** Kaggle "Medical Appointment No Shows" (joniarroba/noshowappointments, `KaggleV2-May-2016.csv`, 110,521 rows after dropping Age<0/>110, Brazil 2016). Committed under `data/kaggle/`.
- **Baseline features:** Age, lead_days (AppointmentDay − ScheduledDay, clipped ≥0), SMS_received, Hipertension, Diabetes, Scholarship.
- **Reconstructed history features (the real lift):** the raw file has no attendance column, but repeat `PatientId`s let us rebuild each patient's past behaviour — **leakage-safe** (strictly earlier appointments), trailing-5 to mirror our live last-5 `attendance_history`: `prior_no_show_rate` (Bayesian-smoothed toward the base rate), `prior_visits`, `is_first_visit`, plus `same_day`. `base_rate`+`alpha` travel in the model bundle so serving reproduces the feature exactly (no train/serve skew).
- **Model:** LightGBM (max_depth=4, lr=0.05, class_weight="balanced", early-stopping on val) + isotonic calibration. **Split: 60/20/20 train/val/test, stratified.**
- **Result:** held-out **test ROC-AUC = 0.742** (train 0.755 / val 0.740). Ablation: baseline-6 = 0.729 → +attendance = 0.744 (**+0.015 AUC**, 5-fold CV). NEVER accuracy (20% base rate makes it meaningless). AUC>0.9 ⇒ leakage — we're well under.
- **Benchmark (`ml/benchmark.py`):** LightGBM ≈ XGBoost (0.744) > RandomForest (0.737) > AdaBoost (0.731) — model choice is within noise; the features are the lever. (TabPFN v2 bonus still optional.)
- **Artifacts:** `ml/reliability_model.pkl`, `ml/metrics.json`, `ml/benchmark.json`, model card `ml/MODEL_CARD.md`.
- **Honesty caveat for README:** Brazilian public-health data demonstrates the method; a real deployment retrains on the practice's own history — which *natively* supplies the attendance (and neighbourhood) signals we reconstruct/approximate here.

### 5.2 Accept score — P(accepts | answered) (rules, no data exists publicly)
```
s = 0.5
+0.25 slot time inside preferred_window
+0.15 slot type in needed_treatments
+min(days_waiting/60, 0.15)
−0.30 declined similar slot in last 7 days
clip to [0.05, 0.95]
```

### 5.3 Hard filters (run BEFORE scoring; each skip emits a reason)
1. `consent_outbound == false` → skip "No consent for outbound calls"
2. slot type not in needed_treatments → skip "Treatment mismatch"
3. time_left < patient's feasible notice (if !short_notice_ok and time_left < 24h) → skip "Cannot make it on short notice"
4. offered a slot in last 72h → skip "Cooldown (called recently)" — UNLESS only feasible candidate (then allow, flag on dashboard)
5. declined similar slot < 7 days ago → skip "Recently declined similar"
6. max 2 offer calls per patient per week → skip "Weekly contact cap"

### 5.4 Final priority (deadline-aware)
```
hours_left   = slot.start − now
urgency_mode = clamp((24 − hours_left)/24, 0, 1)     # 0 relaxed → 1 panic
answer = reliability_model(patient)                   # §5.1
accept = accept_score(patient, slot)                  # §5.2
value  = treatment_value(patient, slot) / max_value   # normalized
score  = answer**(1+urgency_mode) * accept * value**(1−0.7*urgency_mode)
```
Starvation guard: waiting > 30 days ⇒ score ×1.5.
Phase label for dashboard: RELAXED (>24h) / URGENT (2–24h) / CRITICAL (<2h) / UNRECOVERABLE (<20 min or list empty).
Name in pitch: **"deadline-aware dispatch."**

### 5.5 `reason` strings
LLM one-liner per top-5 candidate (Anthropic/OpenAI API). **Fallback template** (must exist): `"{days_waiting}d waiting · {window match?} · attendance {x}/5"`.

---

## 6. fonio Integration (Track B)

### 6.1 Facts from fonio docs (verified against the official API spec)
Docs: help center [fonio.info](https://fonio.info) · Outbound API guide `fonio.info/articles/outbound-calls` · Swagger `app.fonio.ai/api/docs` (raw spec `app.fonio.ai/api/docs-json`).

- **Prerequisites (gating):** outbound needs an **imported phone number** (or SIP/ZIPN) connected to the assistant **+ the Teams plan**, and outbound minutes incur carrier cost. → confirm hackathon credits (§6.2 Q4) before relying on a real call.
- **Trigger endpoint:** `POST https://app.fonio.ai/api/public/v1/outbound_call`. Auth = fonio API key as `Authorization: Bearer <key>` (or `apiKey` in body). Body (`OutboundCallPayloadDto`):
```json
{ "fromNumber": "+43...",   // required — our imported number, connected to the agent
  "toNumber":   "+43...",   // required — patient, must match ^\+\d+$
  "context":    { "slot_id": 12, "patient_id": 7, "patient_name": "Maria",
                  "slot_time": "14:30", "treatment": "cleaning" } }  // required object: our IDs + agent variables
```
  Response 200: `{ "status": "success"|"error", "message": "..." }` — **no call id is returned.** (Sanity-check a key with `POST /api/public/v1/test-api-key`.)
- **Post-call data:** there is **no** post-call REST endpoint in the public API. Outcomes arrive via the per-assistant **post-call webhook** (Assistant → Webhooks, POST), which sends the agent's **defined variables** + `from_number` — NOT a fixed `{id,summary,transcript,formattedTranscript,disconnectReason,audioLink,...}` payload (that earlier field list was wrong — it's Retell's schema, not fonio's). So the outcome must be a **defined variable** (§6.3), and `slot_id`/`patient_id` must travel in `context` to round-trip.
- **Correlation strategy (revised — the response carries no id):** (a) **primary** — embed `slot_id`+`patient_id` in `context` and read them back from the post-call webhook; (b) **fallback** — match `toNumber` + the single in-flight call (safe: dialing is strictly sequential).

### 6.2 Open questions for Kim (most now answered by the docs)
1. ~~Does the outbound API return the call `id` synchronously?~~ **No** — response is `{status, message}` only. (Resolved.)
2. Does `context` **echo back** into the post-call webhook payload? Docs imply yes (dedicated required `context` field) — **confirm**, our primary correlation depends on it. If not, fall back to `toNumber` + in-flight.
3. ~~Post-call webhook URL config?~~ Per-assistant **Webhooks** tab, POST method; expose ours via ngrok/cloudflared. (Resolved.)
4. Do hackathon accounts have an **imported number + Teams plan + outbound-minute/SMS credit**? (Hard blocker — outbound is gated on all three.)

### 6.3 Agents to configure
- **Offer agent (hero):** variables: patient_name, slot_time, treatment, practice_name. Personal opening ("Hi Maria, this is the assistant at Smile Dental — a cleaning slot just opened today at 14:30 and you're first on our list"). Handles: yes → confirm + state booking; no → polite, "you stay on the list"; callback → capture time; price/duration questions → short FAQ answer or "the practice will confirm details". **Must set a defined `outcome` variable to exactly one of OUTCOME_BOOKED / OUTCOME_DECLINED / OUTCOME_CALLBACK** so it arrives deterministically in the post-call webhook (don't depend on regexing a `summary` field — fonio's post-call payload is the variables you define, not a fixed schema). LLM-classify the transcript only as a fallback.
- **Confirmation agent:** "confirming your 14:30 appointment today — are you coming?" → yes = confirmed; no = fires a cancellation into the orchestrator. Triggered by dashboard button for the demo.

### 6.4 Orchestrator state machine
```
on_cancellation(slot_id):
  if recovery_attempts.exists(slot_id): return        # IDEMPOTENT
  insert attempt; slot.status = recovering
  tried = set()
  loop:
    r = POST backend /rank (slot, exclude=tried)
    if no candidates: slot.status=escalated; notify dashboard; break
    c = r.candidates[0]
    fonio.trigger(from=OUR_NUMBER, to=c.phone,            # POST /public/v1/outbound_call
                  context={slot_id, patient_id:c.id, ...vars})   # response = {status,message}, no id
    outcome = wait_webhook(slot_id, patient_id, timeout=90s)     # correlate via context; timeout ⇒ voicemail
                                                          # fallback correlation: toNumber + in-flight
    POST backend /outcome
    if booked: slot.status=filled; break
    tried.add(c.patient_id)
```
Rules: strictly sequential dialing (no parallel — double-booking + brand-damage risk). No calls before 08:00 / after 19:00 (config constant). Phase CRITICAL after 2 failed calls ⇒ (stretch) SMS broadcast to top 5 remaining, first reply claims slot atomically; if not built, escalate instead.

---

## 7. Dashboard (single page, no auth, no routing)

1. **Today's schedule strip** — slots with status colors + no-show risk badge (green/yellow/red from §5.1) on booked slots.
2. **Live recovery panel** — current call (patient, elapsed), ranked list with score breakdowns, **skipped list with reasons** (this is the fairness/consent story — do not cut).
3. **Metric tiles** — refill rate %, € recovered, avg time-to-fill, outcomes donut.
4. **Buttons:** "Simulate cancellation", "Trigger confirmation call", "Mark no-show" (→ partial-refill check: ≥20 min remaining + short_notice candidates ⇒ refill, else unrecoverable with reason).
Updates: 2s polling is fine. React (see frontend-design conventions) or plain HTML+fetch — whatever is fastest.

---

## 8. Edge Cases (≥1 must be VISIBLE in demo)

| Case | Handling |
|---|---|
| No consent | Hard skip + dashboard reason ← **show this one** |
| Waitlist exhausted | slot ESCALATED + receptionist notice ← or this one |
| Webhook never arrives | 90s timeout ⇒ voicemail ⇒ next candidate |
| Duplicate cancellation event | idempotency: UNIQUE(slot_id) on recovery_attempts |
| Two slots cancelled at once | lock patient while a call for them is in flight |
| Cancellation <20 min before slot | UNRECOVERABLE, clean message, no zombie calls |
| Booked-then-recancelled slot | allow max 2 recovery rounds per slot, then escalate |
| Confirmation call: no answer ×2 | never auto-cancel; mark "at risk", notify human |

---

## 9. Timeline

**Saturday night (parallel):**
- A: train model ✅pkl + AUC; data generator ✅practice.json/SQLite
- B: fonio account in; **one real hardcoded outbound call completed end-to-end** (highest-risk item — do first); Kim's 4 answers
- Both: repo skeleton, this file committed

**Sunday:**
- 08:00–10:00 — A: `/rank`+`/outcome` done · B: orchestrator loop done
- 10:00–11:00 — **full integration run during the mentor window** (fonio staff on hand)
- 11:00–12:00 — dashboard + visible edge cases
- 12:00–13:00 — **record 3-min demo video** (before any further polish; it's required + insurance)
- 13:00–13:45 — README (honest real-vs-mocked table), MIT LICENSE at root, secrets scrub incl. `git log`, push to org/fonio/<team>, Tally form
- 13:45 — STOP.

**Cut order if behind:** SMS broadcast → TabPFN benchmark → sentiment chip → LLM reasons (keep template) → weekly metrics (keep 4 tiles) → confirmation agent (keep simulate button).
**Never cut:** live ranked call loop · idempotency · skipped-with-reasons display · README honesty.

---

## 10. Demo Script (3 min)

1. Dashboard: today's schedule, one slot flagged red (no-show risk 67%). (15s)
2. Confirmation call fires to teammate's phone → "can't make it" → slot flips cancelled, recovery starts automatically. (40s)
3. Ranked list appears: top-5 with breakdowns, 2 skips with reasons (consent, cooldown). (20s)
4. Live call #1 → teammate declines → list advances automatically. (40s)
5. Call #2 → "yes" → slot green, € counter ticks, time-to-fill shown. (40s)
6. 10s on a second slot in ESCALATED state ("waitlist exhausted → human"). Close: refill-rate tile. (25s)

**Pitch framing:** "Deadline-aware dispatch. We don't just react to cancellations — we predict no-shows an hour ahead (model trained on 110k real appointments), convert them into recoverable slots, and fill them with the right patient, explainably and fairly."

---

## 11. Submission Checklist

- [ ] Public repo in START Hack Vienna '26 org → `fonio/<team>/`
- [ ] `LICENSE` (MIT) at root · [ ] TabPFN v2 attribution if used
- [ ] README: setup/run, **real-vs-mocked table**, AUC, dataset citation
- [ ] No secrets anywhere incl. git history
- [ ] 3-min video: cancellation → detect → rank → call → booked
- [ ] Tally form: title, one-liner, team, problem, solution, stack, links
- [ ] Optional: live demo link, deck PDF, REPORT.md
