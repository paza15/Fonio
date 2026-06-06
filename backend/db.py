"""SQLite setup + schema.

Persistence is a judging criterion (§3 of PLAN.md). One file, no migrations:
the demo runs against a fresh seeded DB. UNIQUE(slot_id) on recovery_attempts
is the idempotency guard against duplicate cancellation events.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("FONIO_DB_PATH", "data/fonio.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS treatments (
    type TEXT PRIMARY KEY,
    value_eur INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    age INTEGER NOT NULL,
    sms_opt_in INTEGER NOT NULL DEFAULT 1,
    hypertension INTEGER NOT NULL DEFAULT 0,
    diabetes INTEGER NOT NULL DEFAULT 0,
    consent_outbound INTEGER NOT NULL DEFAULT 1,
    short_notice_ok INTEGER NOT NULL DEFAULT 1,
    preferred_window_start TEXT NOT NULL DEFAULT '08:00',
    preferred_window_end TEXT NOT NULL DEFAULT '19:00',
    needed_treatments TEXT NOT NULL DEFAULT '[]',
    days_waiting INTEGER NOT NULL DEFAULT 0,
    attendance_history TEXT NOT NULL DEFAULT '[]',
    last_offer_called_at TEXT,
    last_decline_at TEXT,
    last_declined_slot_type TEXT
);

CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY,
    start_dt TEXT NOT NULL,
    duration_min INTEGER NOT NULL,
    type TEXT NOT NULL,
    value_eur INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked',
    booked_patient_id INTEGER,
    lead_days INTEGER NOT NULL DEFAULT 0,   -- booking horizon: days the appt was booked ahead (model's top feature)
    confirmation_status TEXT,               -- proactive sweep: confirmed | at_risk | cancelled
    FOREIGN KEY(booked_patient_id) REFERENCES patients(id)
);

-- UNIQUE(slot_id) = idempotency guard against duplicate cancel events.
CREATE TABLE IF NOT EXISTS recovery_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id INTEGER NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_by_patient_id INTEGER,
    FOREIGN KEY(slot_id) REFERENCES slots(id),
    FOREIGN KEY(filled_by_patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fonio_call_id TEXT UNIQUE,
    recovery_attempt_id INTEGER,
    patient_id INTEGER,
    slot_id INTEGER,
    direction TEXT NOT NULL,
    outcome TEXT,
    summary TEXT,
    started_at TEXT,
    ended_at TEXT,
    FOREIGN KEY(recovery_attempt_id) REFERENCES recovery_attempts(id),
    FOREIGN KEY(patient_id) REFERENCES patients(id),
    FOREIGN KEY(slot_id) REFERENCES slots(id)
);

CREATE INDEX IF NOT EXISTS idx_slots_status ON slots(status);
CREATE INDEX IF NOT EXISTS idx_calls_fonio ON calls(fonio_call_id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


@contextmanager
def cursor():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def reset_db() -> None:
    # On Windows a background recovery thread may briefly hold the file open;
    # retry the unlink so a fresh seed/test reset doesn't race it.
    for path in (DB_PATH, DB_PATH.with_suffix(".sqlite-wal"), DB_PATH.with_suffix(".sqlite-shm")):
        for _ in range(50):
            if not path.exists():
                break
            try:
                path.unlink()
                break
            except PermissionError:
                time.sleep(0.1)
    init_db()
