"""FastAPI app — /rank, /outcome, dashboard endpoints, fonio webhook.

Run: `uvicorn backend.main:app --reload`
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from backend import orchestrator, reliability, repo
from backend.db import init_db
from backend.reasons import reason_for
from backend.schemas import (
    Candidate, DashboardSlot, DashboardState, Metrics, OutcomeRequest,
    OutcomeResponse, RankRequest, RankResponse, RecoveryState, ScoreBreakdown,
    Skipped,
)
from backend.scoring import Slot, rank
from backend.state import STATE

LOG = logging.getLogger("api")
app = FastAPI(title="fonio Slot Refill Engine")


@app.on_event("startup")
def _startup():
    init_db()
    reliability.load()


# ---- Track A↔B contract (§4) ----

@app.post("/rank", response_model=RankResponse)
def post_rank(req: RankRequest) -> RankResponse:
    slot = Slot(
        id=req.slot.id,
        start=datetime.fromisoformat(req.slot.start),
        duration_min=req.slot.duration_min,
        type=req.slot.type,
        value_eur=req.slot.value_eur,
    )
    patients = repo.all_patients()
    offers = repo.offers_this_week_by_pid()
    ranked, skipped, phase = rank(
        slot, patients, reliability.predict,
        exclude_ids=set(req.exclude_patient_ids),
        offers_this_week_by_pid=offers,
    )
    candidates = [
        Candidate(
            patient_id=r.patient_id,
            name=r.name,
            phone=r.phone,
            score=r.score,
            breakdown=ScoreBreakdown(
                answer_prob=r.answer_prob, accept_score=r.accept_score,
                value_norm=r.value_norm, phase=r.phase,
            ),
            reason=reason_for(r),
        )
        for r in ranked
    ]
    skips = [Skipped(patient_id=s.patient_id, name=s.name, reason=s.reason) for s in skipped]
    return RankResponse(candidates=candidates, skipped=skips, phase=phase)


@app.post("/outcome", response_model=OutcomeResponse)
def post_outcome(req: OutcomeRequest) -> OutcomeResponse:
    if req.outcome == "booked":
        repo.set_slot_status(req.slot_id, "filled", booked_patient_id=req.patient_id)
        repo.finish_recovery(req.slot_id, "filled", req.patient_id)
        repo.push_attendance(req.patient_id, 1)
    elif req.outcome == "declined":
        slot = repo.get_slot(req.slot_id)
        if slot:
            repo.record_decline(req.patient_id, slot.type)
    # log a call row even if the orchestrator didn't (manual /outcome path)
    if req.fonio_call_id and not repo.find_call_by_fonio_id(req.fonio_call_id):
        repo.log_call(
            fonio_call_id=req.fonio_call_id, recovery_attempt_id=None,
            patient_id=req.patient_id, slot_id=req.slot_id,
            direction="outbound", outcome=req.outcome, summary=req.summary,
        )
    return OutcomeResponse(ok=True, slot_status=repo.slot_status(req.slot_id) or "unknown")


# ---- demo / dashboard buttons ----

@app.post("/simulate/cancel/{slot_id}")
def simulate_cancel(slot_id: int):
    if not repo.get_slot(slot_id):
        raise HTTPException(404, "slot not found")
    repo.cancel_slot(slot_id)
    return orchestrator.trigger_recovery(slot_id)


@app.post("/simulate/no_show/{slot_id}")
def simulate_no_show(slot_id: int):
    """Per §7: ≥20 min remaining + short_notice candidates ⇒ refill, else
    unrecoverable with reason. We just delegate to the recovery engine — it
    already detects UNRECOVERABLE via phase."""
    if not repo.get_slot(slot_id):
        raise HTTPException(404, "slot not found")
    repo.cancel_slot(slot_id)
    return orchestrator.trigger_recovery(slot_id)


@app.post("/simulate/confirmation/{slot_id}")
def simulate_confirmation_call(slot_id: int):
    """Demo: fire a confirmation call. For the mock client, this is just a
    cancellation trigger so the recovery loop kicks off visibly."""
    if not repo.get_slot(slot_id):
        raise HTTPException(404, "slot not found")
    repo.cancel_slot(slot_id)
    return orchestrator.trigger_recovery(slot_id)


# ---- fonio post-call webhook (§6.1) ----

@app.post("/webhook/fonio")
async def fonio_webhook(req: Request):
    payload = await req.json()
    cid = payload.get("id") or payload.get("call_id") or ""
    if not cid:
        raise HTTPException(400, "missing call id")
    STATE.webhook_events[cid] = payload
    LOG.info("webhook received for %s", cid)
    return {"ok": True}


# ---- dashboard read endpoints ----

@app.get("/state", response_model=DashboardState)
def dashboard_state() -> DashboardState:
    rows = repo.schedule(days_ahead=2)
    schedule_out: list[DashboardSlot] = []
    for r in rows:
        risk = None
        if r["booked_patient_id"]:
            p = repo.get_patient(r["booked_patient_id"])
            if p:
                # no-show risk for this booked appt uses its real booking horizon
                risk = 1.0 - reliability.predict(p, lead_days=float(r.get("lead_days") or 0))
        schedule_out.append(DashboardSlot(
            id=r["id"], start=r["start_dt"], duration_min=r["duration_min"],
            type=r["type"], value_eur=r["value_eur"], status=r["status"],
            booked_patient_name=r.get("booked_name"),
            no_show_risk=risk,
        ))

    rec = STATE.recovery
    if rec is None:
        recovery = RecoveryState()
    else:
        with STATE.lock:
            recovery = RecoveryState(
                slot_id=rec.slot_id,
                phase=rec.phase,
                current_patient_id=rec.current_patient_id,
                current_patient_name=rec.current_patient_name,
                current_started_at=(rec.current_started_at.isoformat()
                                    if rec.current_started_at else None),
                candidates=[Candidate(**c) for c in rec.candidates],
                skipped=[Skipped(**s) for s in rec.skipped],
                tried_patient_ids=list(rec.tried_patient_ids),
            )

    stats = repo.refill_stats()
    rate = (stats["filled"] / stats["attempts"] * 100) if stats["attempts"] else 0.0
    avg = (sum(STATE.time_to_fill_seconds) / len(STATE.time_to_fill_seconds)
           if STATE.time_to_fill_seconds else None)
    metrics = Metrics(
        refill_rate_pct=round(rate, 1),
        eur_recovered=int(stats["eur"]),
        avg_time_to_fill_seconds=avg,
        outcomes=repo.outcomes_breakdown(),
    )
    return DashboardState(
        now=datetime.now().isoformat(),
        schedule=schedule_out,
        recovery=recovery,
        metrics=metrics,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}
