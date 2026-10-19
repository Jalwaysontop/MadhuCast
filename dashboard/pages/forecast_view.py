"""F10 Forecast view (architecture-doc wireframe): EHR summary left, timeline right, active
alerts and why-panel below, data & system health at the bottom. FHIR export (F9) here too."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.components import charts, common, health
from src.explain import templates as T
from src.serving import fhir_export
from src.serving import timeseries as ts

common.init_session()


def render() -> None:
    state, fc = common.run()
    p = state.profile
    st.title("Forecast")
    g_now = ts.glucose_now(state.history, state.now)
    curve = ts.forecast_curve(fc, g_now)

    badges = [common.prov_badge(fc.provenance)]
    for name, prob in (("Spike > 180", fc.spike_prob), ("Hypo < 70", fc.hypo_prob)):
        cls = "tier-critical" if prob >= 0.6 else "tier-warning" if prob >= 0.3 else "tier-none"
        icon = "▲" if prob >= 0.3 else "✓"
        badges.append(common.badge(f"{icon} {name}: {T.fmt_pct(prob)}", cls))
    badges.append(common.badge(f"confidence {fc.confidence:.2f}", "tier-watch" if fc.confidence < 0.5 else "tier-none"))
    common.md(" ".join(badges))

    left, right = st.columns([1, 2.6])
    with left:
        st.markdown("#### EHR summary")
        st.markdown(
            f"**{p.patient_id}** · {p.cohort.upper()} · {p.age}{p.sex}  \n"
            f"HbA1c **{p.hba1c:.1f}%** · FPG **{p.fasting_glucose:g}** mg/dL · BMI **{p.bmi:.1f}**  \n"
            f"Comorbidities: {', '.join(p.comorbidities) or 'none'}  \n"
            f"Medications *(display only)*: {', '.join(p.medications) or 'none'}"
        )
        st.metric(
            "Current glucose",
            f"{g_now:.0f} mg/dL" if g_now is not None else "UNKNOWN",
            help="UNKNOWN when the last CGM sample is older than 15 min. Never treated as low risk.",
        )
        for h in (30, 60, 120):
            q = fc.horizon(h)
            st.caption(f"+{h} min: **{q[1]:.0f}** mg/dL (80% range {q[0]:.0f}–{q[2]:.0f})")
    with right:
        fig = charts.forecast_figure(state, fc, label="Forecast", y_range=charts.y_range_for((state, fc)))
        st.plotly_chart(fig, width="stretch", key="forecast-main")
        if curve.attrs.get("interpolated"):
            st.caption("Curve interpolated between forecast horizons (+30/+60/+120 min).")
        st.caption(
            "Solid blue: observed CGM (past 4 h). Orange: forecast median, 80% band (darker) and 90% band "
            "(lighter). Triangles: meals, sized by glycaemic load. Hatched: missing CGM, never interpolated."
        )

    a_col, w_col = st.columns(2)
    with a_col:
        st.markdown("#### Active alerts")
        snap = common.snapshot()
        mine = [a for a in (snap.queue if snap else ()) if a.patient_id == p.patient_id]
        if not mine:
            st.write("No active alerts for this patient.")
        for a in mine:
            common.md(
                f"{common.tier_badge(a.tier, a.event == 'unknown')} <b>{a.event}</b> · lead {a.lead_label} · "
                f"{a.reason}"
            )
            health.feedback_buttons(a, key=f"fv-{a.id}")
    with w_col:
        st.markdown("#### Why (top contributors)")
        st.write(T.risk_sentence(fc))
        for s in T.explain(fc, k=3):
            st.markdown(f"- {s}")
        st.caption(T.CAPTION + " See the Why page for details.")

    st.divider()
    health.health_panel(state, fc, key="fv-health")

    st.divider()
    st.markdown("#### FHIR export")
    bundle = fhir_export.build_bundle(state, fc)
    payload = fhir_export.to_json(bundle)
    st.download_button(
        "Download FHIR bundle (JSON)",
        payload,
        icon=":material/download:",
        file_name=f"madhutwin_{p.patient_id}_{state.now.astimezone(ts.IST):%Y%m%d_%H%M}.json",
        mime="application/fhir+json",
    )
    st.caption(
        f"FHIR {fhir_export.FHIR_VERSION} collection: Patient (pseudonymous, tagged synthetic), CGM "
        "Observations (LOINC 99504-3, UCUM mg/dL), a RiskAssessment for spike/hypo and a preliminary "
        "forecast Observation. Validated with fhir.resources before download."
    )
    with st.expander(f"Show bundle ({len(bundle['entry'])} resources)"):
        st.json(bundle, expanded=False)


common.guard(render)
