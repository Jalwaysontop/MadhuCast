"""F10 Forecast: the hero. Readouts on top; patient record (30%) beside the Twin Curve (70%);
why-line; quiet data & system health; FHIR export (F9)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import timedelta

import streamlit as st

from dashboard.components import charts, common, health, kit
from src.explain import templates as T
from src.serving import alerts as al
from src.serving import fhir_export
from src.serving import timeseries as ts

common.init_session()


def render() -> None:
    state, fc = common.run()
    p = state.profile
    mode = common.theme()
    st.title("Forecast")

    g_now = ts.glucose_now(state.history, state.now)
    slope = ts.glucose_slope(state.history, state.now)
    curve = ts.forecast_curve(fc, g_now)
    i = int(curve["q50"].idxmax())
    peak_t = state.now + timedelta(minutes=int(curve.at[i, "t_min"]))
    trend = None if slope is None or abs(slope) < 0.3 else ("up" if slope > 0 else "down")
    snap = common.snapshot()
    mine = [a for a in (snap.queue if snap else ()) if a.patient_id == p.patient_id]
    top = al.sort_queue(mine)[0] if mine else None
    alert = (
        kit.alert_card(
            "UNKNOWN" if top.event == "unknown" else common.TIER_NAME[top.tier],
            _headline(top),
            f"Detectors: {', '.join(top.detectors) or 'none'}",
        )
        if top
        else kit.alert_card("CLEAR", "No active alert", "Engine re-evaluates every 15 min")
    )
    kit.readouts(
        [
            kit.Readout(
                "Current glucose",
                f"{g_now:.0f}" if g_now is not None else "UNKNOWN",
                "mg/dL" if g_now else "",
                (
                    "No CGM sample in the last 15 min"
                    if g_now is None
                    else (
                        f"{'Rising' if slope > 0 else 'Falling'} {abs(slope):.1f} mg/dL/min"
                        if trend
                        else "Steady, last 30 min"
                    )
                ),
                trend=trend,
                foot_tone="warn" if g_now is None else None,
            ),
            kit.Readout(
                "Predicted peak",
                f"{curve.at[i, 'q50']:.0f}",
                "mg/dL",
                f"Around {ts.hhmm(peak_t)}, 80% range {curve.at[i, 'q10']:.0f}–{curve.at[i, 'q90']:.0f}",
            ),
            kit.Readout(
                "Spike risk, next 2 h",
                T.fmt_pct(fc.spike_prob).rstrip("%"),
                "%",
                common.spike_source(fc) or f"Hypo {T.fmt_pct(fc.hypo_prob)}, confidence {fc.confidence:.2f}",
                foot_tone="warn" if common.spike_source(fc) else None,
            ),
        ],
        alert_html=alert,
    )
    st.write("")

    left, right = st.columns([3, 7], gap="medium")
    with left, st.container(key="panel-ehr"):
        st.subheader("Patient record")
        prov = p.provenance

        def pb(field: str) -> str:
            return common.prov_badge(prov[field]) if prov.get(field) else ""

        rows = [
            ("Age, sex", f"{p.age}, {'female' if p.sex == 'F' else 'male'}", ""),
            ("HbA1c", f"{p.hba1c:.1f}%", pb("hba1c")),
            ("Fasting glucose", f"{p.fasting_glucose:g} mg/dL", pb("fasting_glucose")),
            ("BMI", f"{p.bmi:.1f} kg/m²", pb("bmi")),
            ("Comorbidities", ", ".join(p.comorbidities) or "none", ""),
            ("Medications", ", ".join(p.medications) or "none", ""),
        ]
        if p.genetic_risk_score is not None:
            rows.append(("Genetic risk", f"{p.genetic_risk_score:.2f}", common.prov_badge("SYNTHETIC")))
        if p.pairing:
            rows.append(
                (
                    "Pairing",
                    f"CGMacros #{p.pairing.get('cgmacros_id', '?')}, d {p.pairing.get('match_distance', '?')}",
                    common.prov_badge("SYNTHETIC"),
                )
            )
        kit.md(kit.definition_list(rows))
        kit.caption("Medications are display only and never a what-if lever.")
    with right, st.container(key="panel-twin"):
        st.subheader("Twin curve")
        fig = charts.forecast_figure(state, fc, y_range=charts.y_range_for((state, fc)), height=430, mode=mode)
        st.plotly_chart(fig, width="stretch", key="forecast-main", config={"displayModeBar": False}, theme=None)
        if curve.attrs.get("interpolated"):
            kit.caption("Curve interpolated between forecast horizons (+30, +60, +120 min).")
        kit.md(f'<div style="font-size:14px">{kit.esc(T.explain(fc, k=1)[0])}</div>')
        st.page_link("pages/explain_panel.py", label="Why this forecast", icon=":material/help:")

    st.write("")
    health.health_panel(state, fc, key="fv-health")
    if mine:
        with st.expander(f"Active alerts for {p.patient_id} ({len(mine)}) and feedback"):
            for a in mine:
                common.md(
                    f"{common.tier_badge(a.tier, a.event == 'unknown')} <b>{kit.esc(_headline(a))}</b> "
                    f'<span class="mt-muted">{kit.esc(a.reason)}</span>'
                )
                health.feedback_buttons(a, key=f"fv-{a.id}")

    with st.expander("FHIR export"):
        bundle = fhir_export.build_bundle(state, fc)
        st.download_button(
            "Download FHIR bundle (JSON)",
            fhir_export.to_json(bundle),
            icon=":material/download:",
            file_name=f"madhutwin_{p.patient_id}_{state.now.astimezone(ts.IST):%Y%m%d_%H%M}.json",
            mime="application/fhir+json",
        )
        kit.caption(
            f"FHIR {fhir_export.FHIR_VERSION} collection: Patient (pseudonymous, tagged synthetic), CGM "
            "Observations (LOINC 99504-3, UCUM mg/dL), a RiskAssessment for spike and hypo, and a preliminary "
            f"forecast Observation. {len(bundle['entry'])} resources, validated before download."
        )
        st.json(bundle, expanded=False)


def _headline(a: al.Alert) -> str:
    if a.event == "unknown":
        return "Data stale, risk cannot be assessed"
    what = "Spike above 180" if a.event == "spike" else "Hypo below 70"
    return f"{what} {'now' if a.lead_label == 'now' else 'in ' + a.lead_label.replace('~', 'about ')}"


common.guard(render)
