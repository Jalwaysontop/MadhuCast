"""Model-trust page: calibration, subgroups, Clarke grid, ablation, sanity, provenance, limits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import charts, common, tables
from src.explain import templates as T
from src.serving import backend

common.init_session()


def _label(lbl: str) -> str:
    return common.prov_badge("MOCK") if lbl == "MOCK" else common.prov_badge("MODEL_ESTIMATE") if lbl != "MISSING" \
        else common.badge("NOT YET AVAILABLE", "tier-unknown")


def _section(title: str, name: str):
    payload, lbl = common.report(name)
    st.markdown(f"#### {title}")
    common.md(_label(lbl))
    if payload is None:
        st.info(f"`reports/{name}` is not yet available. P2's `scripts/evaluate.py` writes it.")
    return payload


@st.cache_data(show_spinner="Running the attribution / what-if agreement check across the cohort…")
def _cohort_agreement(mode: str, now_iso: str) -> tuple[int, int]:
    from datetime import datetime

    agree = total = 0
    for p in backend.list_patients():
        state = backend.get_state(p, datetime.fromisoformat(now_iso))
        res = T.agreement_check(state, backend.predict(state), lambda s, sc: backend.predict(s, sc))
        agree += sum(r.agree for r in res)
        total += len(res)
    return agree, total


def render() -> None:
    st.title("Model trust")
    st.caption("How far to trust the forecasts. Every number here carries its provenance. MOCK values are "
               "placeholders until P2's evaluation lands; they are not results.")

    summary, slbl = common.report("eval_summary")
    if isinstance(summary, dict) and "horizons" in summary:
        st.markdown("#### Headline accuracy")
        common.md(_label(slbl))
        cols = st.columns(len(summary["horizons"]) + 2)
        for col, (h, m) in zip(cols, summary["horizons"].items()):
            v, lo, hi = m["rmse"]
            col.metric(f"RMSE +{h} min", f"{v:.1f} mg/dL", help=f"95% CI {lo:.1f}–{hi:.1f}")
        cov = summary.get("coverage", {})
        cols[-2].metric("Coverage 80% / 90%", f"{cov.get('80', float('nan')):.0%} / {cov.get('90', float('nan')):.0%}")
        lt = summary.get("lead_time_min", {})
        cols[-1].metric("Lead time median (p25)", f"{lt.get('median', '?')} ({lt.get('p25', '?')}) min")
        st.caption("Context: published T1D forecasters report about 19–21 mg/dL RMSE at 30 min and 31–33 at 60 min on "
                   "different data. That is a reference range, not a target.")

    c1, c2 = st.columns(2)
    with c1:
        cal = _section("Interval calibration", "calibration")
        if cal is not None:
            st.plotly_chart(charts.calibration_figure(cal, "coverage", "Nominal vs empirical coverage"),
                            width="stretch", key="cov")
    with c2:
        if cal is not None:
            st.markdown("#### Event probability reliability")
            which = st.radio("Event", ["spike", "hypo"], horizontal=True, key="rel")
            st.plotly_chart(charts.calibration_figure(cal, f"reliability_{which}", f"Reliability: {which}"),
                            width="stretch", key="relfig")

    c1, c2 = st.columns(2)
    with c1:
        sub = _section("Subgroup error (bootstrap 95% CI)", "subgroups")
        if sub is not None:
            st.dataframe(sub.assign(ci=sub.apply(lambda r: f"[{r['ci_low']:.3g}, {r['ci_high']:.3g}]", axis=1))
                         [["cohort", "metric", "value", "ci", "n_participants"]], hide_index=True, width="stretch")
    with c2:
        clarke = _section("Clarke Error Grid", "clarke_grid")
        if clarke is not None:
            st.plotly_chart(charts.clarke_figure(clarke), width="stretch", key="clarke")

    abl = _section("Fusion ablation", "ablation")
    if abl is not None:
        piv = tables.ablation_pivot(abl)
        best = {m: tables.best_without_overlap(abl, m) for m in abl["metric"].unique()}

        def _style(row):
            return ["font-weight: 700; background-color: rgba(0,158,115,0.25)"
                    if col in best and best[col] == row["variant"] else "" for col in row.index]

        st.dataframe(piv.style.apply(_style, axis=1), hide_index=True, width="stretch")
        notes = [f"{m}: **{v}** is best with non-overlapping CIs" if v else f"{m}: no variant is clearly best "
                 "(CIs overlap), so none is highlighted" for m, v in best.items()]
        st.caption(" · ".join(notes) + ". Variants: EHR-only, wearable-only, concat, FiLM, FiLM + physics.")

    c1, c2 = st.columns(2)
    with c1:
        san = _section("What-if directional sanity", "whatif_sanity")
        if san is not None:
            st.dataframe(san, hide_index=True, width="stretch",
                         column_config={"pass_rate": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                     format="%.2f")})
    with c2:
        st.markdown("#### Attribution ↔ what-if agreement")
        agree, total = _cohort_agreement(common.mode_key(), common.now().isoformat())
        if total:
            st.metric("Agreement rate (cohort, current replay time)", f"{agree / total:.0%}", help=f"{agree}/{total}")
        else:
            st.write("No testable levers at this time.")
        st.caption("On the mock, attributions are faithful by construction, so 100% is expected. The real model is "
                   "where this check matters.")

    st.markdown("#### Data provenance register")
    st.dataframe(tables.PROVENANCE_REGISTER, hide_index=True, width="stretch")

    st.markdown("#### Honest limits")
    for lim in tables.LIMITATIONS:
        st.markdown(f"- {lim}")
    thr, calibrated = common.thresholds()
    if not calibrated:
        st.caption("Alert thresholds in use are UNCALIBRATED placeholders: " + thr.get("stratum_rule", ""))
    st.caption("Positioning: an open, ablation-tested, EHR-conditioned T2D digital twin with a clinician-facing "
               "what-if lab. We do not claim to outperform commercial products.")
    pd.set_option("display.precision", 3)


common.guard(render)
