"""Append-only JSONL logs: clinician feedback on alerts and the Pull/raw-window audit trail.

Feedback is stored now because the future federated loop will learn from it. Files live
in logs/ (gitignored); MADHUTWIN_LOG_DIR overrides the location (tests use a tmp dir).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

FeedbackLabel = Literal["useful", "false_alarm", "missed_event"]
FEEDBACK_LABELS: tuple[str, ...] = ("useful", "false_alarm", "missed_event")


def log_dir() -> Path:
    d = Path(os.environ.get("MADHUTWIN_LOG_DIR") or Path(__file__).resolve().parents[2] / "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append(name: str, record: dict[str, Any]) -> dict[str, Any]:
    record = {"logged_at": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with (log_dir() / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return record


def feedback(alert_id: str, label: str, patient_id: str, model_version: str | None,
             alert_time: datetime | None = None, note: str = "") -> dict[str, Any]:
    if label not in FEEDBACK_LABELS:
        raise ValueError(f"label must be one of {FEEDBACK_LABELS}")
    return _append("feedback.jsonl", {
        "alert_id": alert_id, "label": label, "patient_id": patient_id, "model_version": model_version,
        "alert_time": alert_time.isoformat() if alert_time else None, "note": note,
    })


def audit(action: str, patient_id: str, detail: dict[str, Any] | None = None, consent: bool = False) -> dict[str, Any]:
    return _append("audit.jsonl", {"action": action, "patient_id": patient_id, "consent_flag": consent,
                                   "detail": detail or {}, "stub": True})


def read(name: str, limit: int = 50) -> list[dict[str, Any]]:
    path = log_dir() / name
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(x) for x in lines if x.strip()]
