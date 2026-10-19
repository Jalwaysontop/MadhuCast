"""Data & system health strip, the Pull ("Request raw window") stub, and alert feedback buttons.

The strip stays quiet (muted ink) while everything is fresh and turns loud, with an
UNKNOWN badge, only when a stream is stale.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import kit
from src.contracts import Forecast, PatientState
from src.serving import alerts as al
from src.serving import backend, logbook
from src.serving import timeseries as ts


def _streams(state: PatientState) -> tuple[list[tuple[str, str, bool]], bool]:
    out, stale_any = [], False
    for stream, label in (("cgm", "CGM"), ("wearable", "Wearable")):
        seen = (state.stream_last_seen or {}).get(stream)
        if seen is None:
            out.append((label, "no data", True))
            stale_any = True
            continue
        age = (state.now - seen).total_seconds() / 60
        stale = age > ts.FRESH_MIN
        stale_any |= stale
        out.append((label, f"{'stale' if stale else 'fresh'}, {age:.0f} min", stale))
    return out, stale_any


def health_panel(state: PatientState, fc: Forecast, key: str = "health") -> None:
    streams, stale = _streams(state)
    gaps = [(a, b) for a, b in ts.missing_runs(state.history) if (state.now - b).total_seconds() <= 4 * 3600]
    sleep = ts.prior_sleep_hours(state.history, state.now)
    items = [(lab, val, bad) for lab, val, bad in streams]
    items += [
        ("Gaps, last 4 h", ", ".join(f"{ts.hhmm(a)}–{ts.hhmm(b)}" for a, b in gaps) or "none", bool(gaps)),
        ("Sleep", f"{sleep:.1f} h (proxy)" if sleep is not None else "unknown (proxy)", False),
        (
            "Data quality",
            "unknown" if state.data_quality is None else f"{state.data_quality:.0%}",
            (state.data_quality or 0) < 0.9,
        ),
        ("Model", fc.model_version or "n/a", False),
    ]
    cells = "".join(
        f'<span style="margin-right:22px"><span class="mt-caption">{kit.esc(k)}</span> '
        f'<span style="font-size:12px;font-weight:600;color:var(--mt-{"warning" if bad else "ink"})">{kit.esc(v)}</span></span>'
        for k, v, bad in items
    )
    loud = f'{kit.tier_badge("UNKNOWN")} <b>A stream is stale: risk cannot be assessed.</b> ' if stale else ""
    kit.md(
        f'<div role="status" aria-label="Data and system health" style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 0">'
        f"{loud}{cells}</div>"
    )
    with st.expander("Data and system details, raw-window pull", expanded=stale):
        b = backend.describe()
        kit.md(
            f'<div class="mt-muted">Uplink <b>{kit.esc(state.uplink_mode or "not provided")}</b>. '
            f"Model <b>{kit.esc(fc.model_version or 'n/a')}</b>, features <b>{kit.esc(fc.feature_lib_version or 'n/a')}</b>. "
            f"Backend: data {b['data'].lower()}, model {b['model'].lower()}.</div>"
        )
        c1, c2 = st.columns([2, 3])
        consent = c1.checkbox("Patient consent recorded for raw-window pull", key=f"{key}-consent")
        if c2.button(
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
            st.toast("Raw-window request logged to logs/audit.jsonl. Stub: no data was pulled.")
        kit.caption("Pull is a stub in this PoC: it writes an audit line and nothing else.")


def feedback_buttons(alert: al.Alert, key: str) -> None:
    cols = st.columns(3)
    labels = {
        "useful": ("Useful", ":material/thumb_up:"),
        "false_alarm": ("False alarm", ":material/thumb_down:"),
        "missed_event": ("Missed event", ":material/report:"),
    }
    for col, (lab, (txt, icon)) in zip(cols, labels.items()):
        if col.button(txt, key=f"{key}-{lab}", icon=icon, width="stretch", type="secondary"):
            logbook.feedback(alert.id, lab, alert.patient_id, alert.model_version, alert.opened_at)
            st.toast(f"Feedback saved for {alert.id}: {txt.lower()}.")
