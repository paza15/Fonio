"""fonio client — real + mock.

The verified fonio outbound API returns NO call id, so we generate our own
`call_attempt_id` per trigger and round-trip it (plus slot_id, patient_id) inside
the request `context`. It comes back in the post-call webhook's extracted
variables, which is how we correlate the async outcome to the in-flight call. We
also store `call_attempt_id -> (slot, patient, to_number)` in STATE.pending_calls
at trigger time as a correlation fallback (matched by slot+patient or by toNumber
against the single in-flight call — dialing is strictly sequential).

The mock client simulates calls deterministically (no fonio credentials): it
fires a webhook event after a short ring delay, keyed by the same call_attempt_id,
with outcomes scripted per patient id (patient 1 -> BOOKED, others -> DECLINED).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

from backend.state import STATE


@dataclass
class TriggerResult:
    fonio_call_id: str
    accepted: bool
    error: Optional[str] = None


class FonioClient:
    def trigger_offer(self, *, slot_id: int, patient_id: int, phone: str,
                      variables: dict) -> TriggerResult: ...
    def trigger_confirmation(self, *, slot_id: int, patient_id: int, phone: str,
                             variables: dict) -> TriggerResult: ...


# ---- real ----

class RealFonioClient(FonioClient):
    """VERIFIED fonio outbound contract.

    POST https://app.fonio.ai/api/public/v1/outbound_call
      header: Authorization: Bearer <FONIO_API_KEY>
      body  : { fromNumber (E.164, our imported line — it SELECTS the assistant,
                            there is NO agent_id),
                toNumber   (^\\+\\d+$),
                context    (object — round-trips into the post-call webhook's
                            extracted variables) }
      resp  : { status: "success"|"error", message } — NO call id is returned.

    Since fonio returns no call id, we generate our OWN call_attempt_id and embed
    it (+ slot_id, patient_id) in `context` so it round-trips in the post-call
    webhook for correlation (see main.fonio_webhook).
    """

    def __init__(self):
        self.api_key = os.environ.get("FONIO_API_KEY", "")
        self.from_number = os.environ.get("FONIO_FROM_NUMBER", "")
        self.url = os.environ.get(
            "FONIO_OUTBOUND_API_URL",
            "https://app.fonio.ai/api/public/v1/outbound_call",
        )

    def _trigger(self, *, slot_id: int, patient_id: int, phone: str,
                 variables: dict) -> TriggerResult:
        call_attempt_id = f"att-{slot_id}-{patient_id}-{uuid.uuid4().hex[:8]}"
        context = {
            **variables,
            "slot_id": slot_id,
            "patient_id": patient_id,
            "call_attempt_id": call_attempt_id,
        }
        body = {
            "fromNumber": self.from_number,
            "toNumber": phone,
            "context": context,
        }
        try:
            r = httpx.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body, timeout=15.0,
            )
            data = r.json()
            status = data.get("status")
            message = data.get("message")
            accepted = status == "success"
        except Exception as e:
            # No call was placed -> do NOT register pending_calls. Return our
            # generated id so logging/_wait_for_outcome keys still line up.
            return TriggerResult(fonio_call_id=call_attempt_id, accepted=False, error=str(e))
        if accepted:
            STATE.pending_calls[call_attempt_id] = {
                "slot_id": slot_id, "patient_id": patient_id, "to_number": phone,
            }
        return TriggerResult(
            fonio_call_id=call_attempt_id,
            accepted=accepted,
            error=None if accepted else message,
        )

    def trigger_offer(self, **kw):
        return self._trigger(**kw)

    def trigger_confirmation(self, **kw):
        return self._trigger(**kw)


# ---- mock ----

class MockFonioClient(FonioClient):
    """Deterministic mock: outcomes scripted per patient id for the demo.

    Patient 1 (Maria Huber) → BOOKED on the demo cleaning slot.
    All others → DECLINED after a 2-second "ring" so the loop advances visibly.
    """

    OUTCOME_SCRIPT = {
        1: ("OUTCOME_BOOKED", "Patient confirmed they will take the slot. OUTCOME_BOOKED"),
        4: ("OUTCOME_DECLINED", "Patient cannot make the time on short notice. OUTCOME_DECLINED"),
    }
    DEFAULT = ("OUTCOME_DECLINED", "Patient politely declined the offered slot. OUTCOME_DECLINED")
    # confirmation sweep: patient_id → outcome; default everyone confirms they're coming
    CONFIRMATION = {
        "OUTCOME_CONFIRMED": "Patient confirmed they are coming. OUTCOME_CONFIRMED",
        "OUTCOME_CANCEL": "Patient says they can no longer make it. OUTCOME_CANCEL",
    }

    def __init__(self, *, ring_seconds: float = 1.0, confirmation_script: dict | None = None):
        self.ring_seconds = ring_seconds
        self.confirmation_script = confirmation_script or {}

    def _trigger(self, *, slot_id: int, patient_id: int, phone: str,
                 variables: dict, agent: str) -> TriggerResult:
        # Mirror the real client's correlation: our own call_attempt_id is the key
        # the orchestrator waits on, and it is what we carry in `context`.
        call_attempt_id = f"att-{slot_id}-{patient_id}-{uuid.uuid4().hex[:8]}"
        STATE.pending_calls[call_attempt_id] = {
            "slot_id": slot_id, "patient_id": patient_id, "to_number": phone,
        }
        threading.Thread(
            target=self._fire_webhook,
            args=(call_attempt_id, slot_id, patient_id, agent),
            daemon=True,
        ).start()
        return TriggerResult(fonio_call_id=call_attempt_id, accepted=True)

    def trigger_offer(self, **kw):
        return self._trigger(agent="offer", **kw)

    def trigger_confirmation(self, **kw):
        return self._trigger(agent="confirm", **kw)

    def _fire_webhook(self, cid: str, slot_id: int, patient_id: int, agent: str):
        time.sleep(self.ring_seconds)
        # Confirmation sweep (prevention) vs offer call. Fire DIRECTLY into
        # STATE.webhook_events keyed by our call_attempt_id, carrying the same
        # `context` shape the real webhook round-trips, so _wait_for_outcome resolves.
        if agent == "confirm":
            token = self.confirmation_script.get(patient_id, "OUTCOME_CONFIRMED")
            summary = self.CONFIRMATION.get(token, token)
        else:
            token, summary = self.OUTCOME_SCRIPT.get(patient_id, self.DEFAULT)
        STATE.webhook_events[cid] = {
            "summary": summary,
            "transcript": f"[mock transcript] {summary}",
            "context": {
                "slot_id": slot_id,
                "patient_id": patient_id,
                "call_attempt_id": cid,
            },
            "received_at": datetime.now().isoformat(),
        }


def build_client() -> FonioClient:
    use_mock = os.environ.get("ORCHESTRATOR_USE_MOCK", "true").lower() == "true"
    return MockFonioClient() if use_mock else RealFonioClient()
