"""fonio client — real + mock.

Real client posts to the outbound API and stores `fonio_call_id → (slot, patient)`
at trigger time (the baseline correlation strategy from §6.1). The mock client
simulates calls deterministically so the orchestrator can be tested without
fonio credentials.

The mock fires a webhook event after a short delay; outcomes are driven by
hand-tuned narrative rules on patient id (see _mock_outcome).
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
    def __init__(self):
        self.api_key = os.environ.get("FONIO_API_KEY", "")
        self.base = os.environ.get("FONIO_BASE_URL", "https://api.fonio.ai")
        self.offer_agent = os.environ.get("FONIO_OUTBOUND_AGENT_ID", "")
        self.confirm_agent = os.environ.get("FONIO_CONFIRMATION_AGENT_ID", "")

    def _trigger(self, *, agent_id: str, slot_id: int, patient_id: int,
                 phone: str, variables: dict) -> TriggerResult:
        # NOTE: exact endpoint/payload TBD — adjust once Kim confirms the
        # outbound API shape (Saturday-night todo). Until then the contract is:
        #   POST {base}/v1/outbound { agent_id, to, variables, context }
        # We always set `context` with our (slot_id, patient_id) — §6.1.
        payload = {
            "agent_id": agent_id,
            "to": phone,
            "variables": variables,
            "context": {"slot_id": slot_id, "patient_id": patient_id},
        }
        try:
            r = httpx.post(
                f"{self.base}/v1/outbound",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=15.0,
            )
            r.raise_for_status()
            cid = r.json().get("id") or r.json().get("call_id") or ""
        except Exception as e:
            return TriggerResult(fonio_call_id="", accepted=False, error=str(e))
        STATE.pending_calls[cid] = {"slot_id": slot_id, "patient_id": patient_id}
        return TriggerResult(fonio_call_id=cid, accepted=True)

    def trigger_offer(self, **kw): return self._trigger(agent_id=self.offer_agent, **kw)
    def trigger_confirmation(self, **kw): return self._trigger(agent_id=self.confirm_agent, **kw)


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

    def __init__(self, *, ring_seconds: float = 2.5):
        self.ring_seconds = ring_seconds

    def _trigger(self, *, slot_id: int, patient_id: int, phone: str,
                 variables: dict, agent: str) -> TriggerResult:
        cid = f"mock_{uuid.uuid4().hex[:10]}"
        STATE.pending_calls[cid] = {"slot_id": slot_id, "patient_id": patient_id}
        threading.Thread(
            target=self._fire_webhook,
            args=(cid, slot_id, patient_id),
            daemon=True,
        ).start()
        return TriggerResult(fonio_call_id=cid, accepted=True)

    def trigger_offer(self, **kw):
        return self._trigger(agent="offer", **kw)

    def trigger_confirmation(self, **kw):
        return self._trigger(agent="confirm", **kw)

    def _fire_webhook(self, cid: str, slot_id: int, patient_id: int):
        time.sleep(self.ring_seconds)
        token, summary = self.OUTCOME_SCRIPT.get(patient_id, self.DEFAULT)
        STATE.webhook_events[cid] = {
            "id": cid,
            "summary": summary,
            "transcript": f"[mock transcript] {summary}",
            "context": {"slot_id": slot_id, "patient_id": patient_id},
            "received_at": datetime.now().isoformat(),
        }


def build_client() -> FonioClient:
    use_mock = os.environ.get("ORCHESTRATOR_USE_MOCK", "true").lower() == "true"
    return MockFonioClient() if use_mock else RealFonioClient()
