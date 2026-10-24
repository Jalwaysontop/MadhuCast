"""F7 Alert queue: the calm triage landing page. The alert engine (src/serving/alerts.py) does the logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import timedelta

import streamlit as st

from dashboard.components import common, health, kit
from src.serving import alerts as al
from src.serving import backend
from src.serving import timeseries as ts

common.init_session()
TIERS = {"Critical": al.Tier.CRITICAL, "Warning": al.Tier.WARNING, "Watch": al.Tier.WATCH}
EVENT = {"hypo": "Hypo below 70", "spike": "Spike above 180", "unknown": "Data stale"}


def _timeline(snap: al.Snapshot) -> None:
    start, end, _ = backend.time_window()
    span = (end - start).total_seconds()
    pos = lambda t: max(0.0, min(100.0, (t - start).total_seconds() / span * 100))  # noqa: E731
    seen: dict[str, al.Alert] = {}
    for s in common.snapshots_until(snap.now):
        for a in s.queue:
            seen.setdefault(a.id, a)
    ticks = "".join(
        f'<i title="{kit.esc(a.patient_id)} {kit.esc(a.event)} {ts.hhmm(a.opened_at)}" style="position:absolute;left:{pos(a.opened_at):.2f}%;'
        f"top:2px;width:3px;height:12px;border-radius:1px;background:var(--mt-"
        f'{"unknown" if a.event == "unknown" else common.TIER_NAME[a.max_tier].lower()})"></i>'
        for a in seen.values()
    )
    labels = []
    t = start.astimezone(ts.IST).replace(minute=0, second=0, microsecond=0)
    while t <= end:
        if t >= start and t.hour % 3 == 0:
            labels.append(
                f'<span style="position:absolute;left:{pos(t):.2f}%;transform:translateX(-50%)">{t:%H:%M}</span>'
            )
        t += timedelta(hours=1)
    now_p = pos(snap.now)
    kit.md(
        f'<div class="st-key-panel-timeline" style="padding:12px 20px" role="group" aria-label="Replay timeline">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        f'<span class="mt-scrubber-title">Replay the day</span><span class="mt-caption">Episodes opened, coloured by tier. '
        f"Drag “Replay time” in the sidebar.</span></div>"
        f'<div style="position:relative;height:30px">'
        f'<div style="position:absolute;left:0;right:0;top:15px;height:3px;border-radius:2px;background:var(--mt-hairline)"></div>'
        f'<div style="position:absolute;left:0;width:{now_p:.2f}%;top:15px;height:3px;border-radius:2px;background:var(--mt-jamun)"></div>'
        f"{ticks}"
        f'<div style="position:absolute;left:calc({now_p:.2f}% - 7px);top:9.5px;width:14px;height:14px;border-radius:50%;'
        f'background:var(--mt-surface);border:2px solid var(--mt-jamun)"></div></div>'
        f'<div class="mt-caption" style="position:relative;height:16px;margin:0 8px">{"".join(labels)}</div></div>'
    )


def render() -> None:
    st.title("Alert queue")
    thr, calibrated = common.thresholds()
    snap = common.snapshot()
    if snap is None:
        kit.empty_state("No evaluations yet.", "Move “Replay time” in the sidebar later than the start of the day.")
        return
    cal = (
        kit.tier_badge("CLEAR").replace("No alert", "Calibrated thresholds")
        if calibrated
        else '<span class="mt-badge">Default thresholds</span>'
    )
    n = len(common.patients())
    kit.md(
        f'<div style="display:flex;gap:12px;align-items:center;margin:-6px 0 12px">{cal}'
        f'<span class="mt-muted" style="margin-left:auto">{n} patients, re-evaluated every 15 min, replay at {ts.hhmm(snap.now)} IST</span></div>'
    )
    _timeline(snap)
    st.write("")

    b, cfg = snap.burden, al.AlertConfig()
    eval_summary, ev_label = common.report("eval_summary")
    rab = eval_summary.get("recall_at_budget") if isinstance(eval_summary, dict) else None
    kit.readouts(
        [
            kit.Readout(
                "Alert burden",
                f"{b.per_patient_day:.1f}",
                "per patient-day",
                f"Budget {cfg.budget_per_patient_day:.0f}, {'within budget' if b.within_budget else 'over budget'}",
                foot_tone=None if b.within_budget else "warn",
            ),
            kit.Readout(
                "Actionable alerts",
                f"{b.actionable_per_patient_day:.1f}",
                "per patient-day",
                f"Target under {cfg.actionable_target:.0f}"
                + ("" if b.actionable_per_patient_day < cfg.actionable_target else ", needs tuning"),
                foot_tone=None if b.actionable_per_patient_day < cfg.actionable_target else "warn",
            ),
            kit.Readout(
                "Critical share", f"{b.critical_share * 100:.0f}", "%", f"{b.critical} of {b.episodes} episodes"
            ),
            kit.Readout(
                "Recall at budget",
                f"{rab * 100:.0f}" if rab is not None else "n/a",
                "%" if rab is not None else "",
                "From the evaluation summary",
                badge=kit.provenance_badge(ev_label if ev_label != "MISSING" else "MOCK"),
            ),
        ]
    )
    if not b.within_budget:
        kit.disclaimer("Alert burden is above budget: thresholds need tuning before this is usable.")
    st.write("")

    f1, f2 = st.columns(2)
    cohorts = f1.multiselect("Cohort", ["healthy", "prediabetes", "t2d"], default=["healthy", "prediabetes", "t2d"])
    tiers = f2.multiselect("Tier", list(TIERS), default=list(TIERS))
    allowed = {TIERS[t] for t in tiers}
    queue = [a for a in snap.queue if a.cohort in cohorts and a.tier in allowed]
    if not queue:
        kit.empty_state(
            "No active alerts for these filters.", "Widen the cohort or tier filters, or move the replay time."
        )
        return

    rows = [
        {
            "id": a.id,
            "tier": "UNKNOWN" if a.event == "unknown" else common.TIER_NAME[a.tier],
            "pid": a.patient_id,
            "cohort": "T2D" if a.cohort == "t2d" else a.cohort.capitalize(),
            "event": (
                EVENT[a.event] if a.event != "unknown" else f"CGM silent, {a.reason.split('for ')[-1].split(';')[0]}"
            ),
            "prob": a.probability,
            "prob_txt": "n/a" if a.probability is None else _pct(a.probability),
            "conf": a.confidence,
            "lead_frac": None if a.lead_min is None else 1 - min(a.lead_min, 120) / 120,
            "lead_txt": a.lead_label,
            "dq": a.data_quality,
            "age": f"{a.age_min(snap.now)} min",
            "escalated": a.escalated,
        }
        for a in queue
    ]
    kit.md(kit.triage_table(rows))
    kit.caption(
        "Sorted by tier, then lead time, then probability. Low confidence only demotes to Watch; a hypo alert is "
        "never suppressed. Missing data is Unknown, never low risk. ↑ marks a Watch escalated after 2 evaluations. "
        "Lead time is the earliest predicted crossing, not a precise prediction."
    )

    st.subheader("Alert details")
    for a in queue:
        tier = "Unknown" if a.event == "unknown" else common.TIER_NAME[a.tier].capitalize()
        with st.expander(
            f"{a.patient_id}: {EVENT[a.event].lower()}, {tier}, lead {a.lead_label}",
            expanded=a.tier == al.Tier.CRITICAL,
        ):
            kit.md(
                f"{common.tier_badge(a.tier, a.event == 'unknown')} "
                f"&nbsp;<b>{kit.esc(a.reason)}</b>"
            )
            kit.caption(
                f"Alert {a.id}, opened {ts.hhmm(a.opened_at)}, updated {ts.hhmm(a.updated_at)}. Horizon "
                f"{a.horizon_min} min. Detectors: {', '.join(a.detectors) or 'none'}. Model {a.model_version or 'n/a'}. "
                f"Evaluations: {a.n_evals}."
            )
            c1, c2 = st.columns([1, 3])
            if c1.button("Open patient", key=f"open-{a.id}", icon=":material/arrow_forward:", type="primary"):
                common.open_patient(a.patient_id)
                st.switch_page("pages/forecast_view.py")
            with c2:
                health.feedback_buttons(a, key=f"fb-{a.id}")


def _pct(p: float) -> str:
    return ">99%" if p > 0.99 else ("<1%" if p < 0.01 else f"{p:.0%}")


common.guard(render)
