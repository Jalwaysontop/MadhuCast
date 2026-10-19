"""Data & system health panel, the Pull ("Request raw window") stub, and alert feedback buttons."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import common
from src.contracts import Forecast, PatientState
from src.serving import alerts as al
from src.serving import backend, logbook
from src.serving import timeseries as ts


def health_panel(state: PatientState, fc: Forecast, key: str = "health") -> None:
    st.markdown("#### Data & system health")
    rows = []
    for stream, label in (("cgm", "CGM"), ("wearable", "Wearable (HR / METs)")):
        seen = (state.stream_last_seen or {}).get(stream)
        if seen is None:
            rows.append({"Stream": label, "Last sample": "never", "Age": "n/a", "Status": "? UNKNOWN"})
            continue
        age = (state.now - seen).total_seconds() / 60
        status = "✓ FRESH" if age <= ts.FRESH_MIN else "? STALE: risk UNKNOWN"
        rows.append({"Stream": label, "Last sample": ts.hhmm(seen), "Age": f"{age:.0f} min", "Status": status})
    gaps = [(a, b) for a, b in ts.missing_runs(state.history) if (state.now - b).total_seconds() <= 4 * 3600]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Gaps in last 4 h: "
            + (", ".join(f"{ts.hhmm(a)}–{ts.hhmm(b)}" for a, b in gaps) or "none")
            + " · sleep is a proxy derived from night METs and heart rate."
        )
    with c2:
        dq, cls = common.dq_label(state.data_quality)
        b = backend.describe()
        common.md(
            f"<b>Data quality</b> {common.badge(dq, cls)}<br>"
            f"<b>Uplink mode</b> {state.uplink_mode or 'not provided'}<br>"
            f"<b>Model</b> <code>{fc.model_version or 'n/a'}</code> · "
            f"<b>features</b> <code>{fc.feature_lib_version or 'n/a'}</code><br>"
            f"<b>Backend</b> data {b['data']} / model {b['model']}"
        )
        consent = st.checkbox("Patient consent recorded for raw-window pull", key=f"{key}-consent")
        if st.button(
            "Request raw window",
            key=f"{key}-pull",
            icon=":material/download:",
            help="PoC stub: logs an audit entry only. No raw data leaves the device.",
        ):
            logbook.audit(
                "request_raw_window",
                state.profile.patient_id,
                {"now": state.now.isoformat(), "window_min": 60},
                consent=consent,
            )
            st.toast("Raw-window request logged to logs/audit.jsonl (stub, no data pulled).", icon="📝")
        st.caption("Pull is a stub in this PoC: it writes an audit line and nothing else.")


def feedback_buttons(alert: al.Alert, key: str) -> None:
    cols = st.columns(3)
    labels = {
        "useful": ("Useful", ":material/thumb_up:"),
        "false_alarm": ("False alarm", ":material/thumb_down:"),
        "missed_event": ("Missed event", ":material/report:"),
    }
    for col, (lab, (txt, icon)) in zip(cols, labels.items()):
        if col.button(txt, key=f"{key}-{lab}", icon=icon, width="stretch"):
            logbook.feedback(alert.id, lab, alert.patient_id, alert.model_version, alert.opened_at)
            st.toast(f"Feedback '{txt}' saved for {alert.id}.", icon="✅")
