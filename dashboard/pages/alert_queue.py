"""F7 Alert queue: triage landing page. The alert engine (src/serving/alerts.py) does the logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import common, health
from src.serving import alerts as al
from src.serving import timeseries as ts

common.init_session()


def render() -> None:
    st.title("Alert queue")
    thr, calibrated = common.thresholds()
    snap = common.snapshot()
    if snap is None:
        st.info("Move the replay time forward to start the alert engine.")
        return

    cal_badge = (common.badge("✓ CALIBRATED thresholds", "tier-none") if calibrated else
                 common.badge("⚠ UNCALIBRATED placeholder thresholds", "tier-warning",
                              "Placeholder thresholds until P2's thresholds.json (chosen on validation) lands"))
    common.md(f"Replay at <b>{ts.hhmm(snap.now)} IST</b> · evaluated every 15 min since "
              f"the start of the day · {cal_badge}")

    b = snap.burden
    cfg = al.AlertConfig()
    eval_summary, ev_label = common.report("eval_summary")
    c = st.columns(4)
    c[0].metric("Alerts / patient-day", f"{b.per_patient_day:.1f}",
                delta=f"budget {cfg.budget_per_patient_day:.0f}", delta_color="off",
                help=f"{b.episodes} episodes over {b.patient_days} patient-days of replay")
    c[1].metric("Actionable / patient-day", f"{b.actionable_per_patient_day:.1f}",
                delta=f"target < {cfg.actionable_target:.0f}", delta_color="off",
                help="WARNING or CRITICAL episodes (including WATCH escalated after 2 evaluations)")
    c[2].metric("Critical share", f"{b.critical_share:.0%}", help=f"{b.critical} critical episodes")
    rab = (eval_summary or {}).get("recall_at_budget") if isinstance(eval_summary, dict) else None
    c[3].metric(f"Recall at budget ({ev_label})", f"{rab:.0%}" if rab is not None else "n/a",
                help="From reports/eval_summary.json. MOCK values are placeholders, not results.")
    if not b.within_budget:
        st.warning("Alert burden is above budget: thresholds need tuning before this is usable.", icon="⚠️")

    queue = list(snap.queue)
    f1, f2 = st.columns(2)
    cohorts = f1.multiselect("Cohort", ["healthy", "prediabetes", "t2d"], default=["healthy", "prediabetes", "t2d"])
    tiers = f2.multiselect("Tier", ["CRITICAL", "WARNING", "WATCH"], default=["CRITICAL", "WARNING", "WATCH"])
    queue = [a for a in queue if a.cohort in cohorts and al.TIER_LABEL[a.tier] in tiers]

    if not queue:
        st.success("No active alerts for the selected filters at this time.", icon="✅")
        return

    table = pd.DataFrame([{
        "Tier": ("? UNKNOWN" if a.event == "unknown" else
                 f"{common.TIER_STYLE[a.tier][0]} {al.TIER_LABEL[a.tier]}") + (" ↑" if a.escalated else ""),
        "Patient": a.patient_id, "Cohort": a.cohort,
        "Event": {"hypo": "Hypo < 70", "spike": "Spike > 180", "unknown": "Data stale"}[a.event],
        "Probability": a.probability, "Confidence": a.confidence, "Lead time": a.lead_label,
        "Detectors": ", ".join(a.detectors) or "n/a", "Data quality": a.data_quality,
        "Age (min)": a.age_min(snap.now),
    } for a in queue])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Probability": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
        "Confidence": st.column_config.NumberColumn(format="%.2f"),
        "Data quality": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption("Sorted by tier, then lead time, then probability. ↑ = WATCH escalated after 2 consecutive "
               "evaluations. Low confidence only demotes to WATCH; hypo alerts are never suppressed. "
               "Lead time is the earliest predicted crossing, not a precise prediction.")

    st.markdown("#### Alert details")
    for a in queue:
        unknown = a.event == "unknown"
        title = f"{a.patient_id} · {a.event} · {al.TIER_LABEL[a.tier]} · {a.lead_label}"
        with st.expander(title, expanded=a.tier == al.Tier.CRITICAL):
            common.md(f"{common.tier_badge(a.tier, unknown)} {common.prov_badge('MOCK' if 'mock' in (a.model_version or '') else 'MODEL_ESTIMATE')}"
                      f" &nbsp; <b>{a.reason}</b>")
            st.caption(f"Alert {a.id} · opened {ts.hhmm(a.opened_at)} · updated {ts.hhmm(a.updated_at)} · "
                       f"horizon {a.horizon_min} min · detectors: {', '.join(a.detectors) or 'none'} · "
                       f"model {a.model_version or 'n/a'} · evaluations {a.n_evals}")
            c1, c2 = st.columns([1, 3])
            if c1.button("Open patient", key=f"open-{a.id}", icon=":material/arrow_forward:"):
                common.open_patient(a.patient_id)
                st.switch_page("pages/forecast_view.py")
            with c2:
                health.feedback_buttons(a, key=f"fb-{a.id}")


common.guard(render)
