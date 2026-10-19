"""Model trust: an editorial page, because it is an argument about honesty. Data first, then words."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime

import streamlit as st

from dashboard.components import charts, common, kit, tables
from src.explain import templates as T
from src.serving import backend

common.init_session()


def _badge(lbl: str) -> str:
    if lbl == "MISSING":
        return kit.tier_badge("UNKNOWN").replace("Unknown", "Not yet available")
    return common.prov_badge("MOCK" if lbl == "MOCK" else "MODEL_ESTIMATE")


def _section(title: str, name: str):
    payload, lbl = common.report(name)
    kit.md(
        f'<div style="display:flex;align-items:center;gap:10px;margin:18px 0 6px">'
        f'<h3 style="margin:0;padding:0;font-size:20px;font-weight:600">{kit.esc(title)}</h3>{_badge(lbl)}</div>'
    )
    if payload is None:
        kit.empty_state(
            f"reports/{name} is not available yet.",
            "Person 2's scripts/evaluate.py writes it. " "Set MADHUTWIN_MODEL=mock to preview with placeholders.",
        )
    return payload


@st.cache_data(show_spinner="Running the attribution / what-if agreement check across the cohort…")
def _cohort_agreement(mode: str, now_iso: str) -> tuple[int, int]:
    agree = total = 0
    for p in backend.list_patients():
        state = backend.get_state(p, datetime.fromisoformat(now_iso))
        res = T.agreement_check(state, backend.predict(state), lambda s, sc: backend.predict(s, sc))
        agree += sum(r.agree for r in res)
        total += len(res)
    return agree, total


def render() -> None:
    mode = common.theme()
    st.title("Model trust")
    kit.lead(
        "How far to trust the twin. Every number on this page carries its provenance. Mock values are "
        "placeholders until Person 2's evaluation lands, and are not results."
    )

    summary, slbl = common.report("eval_summary")
    if isinstance(summary, dict) and "horizons" in summary:
        cov, lt = summary.get("coverage", {}), summary.get("lead_time_min", {})
        items = [
            kit.Readout(
                f"RMSE +{h} min",
                f"{m['rmse'][0]:.1f}",
                "mg/dL",
                f"95% CI {m['rmse'][1]:.1f}–{m['rmse'][2]:.1f}",
                badge=_badge(slbl),
            )
            for h, m in summary["horizons"].items()
        ]
        items.append(
            kit.Readout(
                "Coverage, 80% and 90%",
                f"{cov.get('80', 0):.0%} / {cov.get('90', 0):.0%}",
                "",
                "Nominal 80% and 90%",
                badge=_badge(slbl),
            )
        )
        items.append(
            kit.Readout(
                "Lead time, median",
                f"{lt.get('median', '?')}",
                "min",
                f"25th percentile {lt.get('p25', '?')} min",
                badge=_badge(slbl),
            )
        )
        kit.readouts(items)
        kit.caption(
            "Context: published T1D forecasters report about 19–21 mg/dL RMSE at 30 min and 31–33 at 60 min on "
            "different data. That is a reference range, not a target."
        )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        cal = _section("Interval calibration", "calibration")
        if cal is not None:
            st.plotly_chart(
                charts.calibration_figure(cal, "coverage", "Nominal vs empirical coverage", mode=mode),
                width="stretch",
                key="cov",
                config={"displayModeBar": False},
                theme=None,
            )
    with c2:
        if cal is not None:
            kit.md('<h3 style="margin:18px 0 6px;font-size:20px;font-weight:600">Event probability reliability</h3>')
            which = st.segmented_control("Event", ["Spike", "Hypo"], default="Spike", key="rel") or "Spike"
            st.plotly_chart(
                charts.calibration_figure(
                    cal, f"reliability_{which.lower()}", f"Reliability: {which.lower()}", mode=mode
                ),
                width="stretch",
                key="relfig",
                config={"displayModeBar": False},
                theme=None,
            )

    abl = _section("Fusion ablation", "ablation")
    if abl is not None:
        best = {m: tables.best_without_overlap(abl, m) for m in abl["metric"].unique()}
        cols = st.columns(len(best), gap="medium")
        for col, (m, v) in zip(cols, best.items()):
            col.plotly_chart(
                charts.ablation_dots(abl, m, mode=mode, highlight=v),
                width="stretch",
                key=f"abl-{m}",
                config={"displayModeBar": False},
                theme=None,
            )
        notes = [
            (
                f"{m.replace('_', ' ')}: {v} is best with non-overlapping intervals"
                if v
                else f"{m.replace('_', ' ')}: no variant is clearly best (intervals overlap), so none is highlighted"
            )
            for m, v in best.items()
        ]
        kit.caption(
            ". ".join(notes) + ". Variants: glucose-only, EHR-only, wearable-only, concat, FiLM, FiLM + physics. "
            "Dots are point estimates, whiskers are 95% bootstrap intervals."
        )
        with st.expander("Ablation table"):
            st.dataframe(tables.ablation_pivot(abl), hide_index=True, width="stretch")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        sub = _section("Subgroup error, bootstrap 95% CI", "subgroups")
        if sub is not None:
            st.dataframe(
                sub.assign(ci=sub.apply(lambda r: f"{r['ci_low']:.3g} to {r['ci_high']:.3g}", axis=1))[
                    ["cohort", "metric", "value", "ci", "n_participants"]
                ],
                hide_index=True,
                width="stretch",
            )
    with c2:
        clarke = _section("Clarke Error Grid", "clarke_grid")
        if clarke is not None:
            st.plotly_chart(
                charts.clarke_figure(clarke, mode=mode),
                width="stretch",
                key="clarke",
                config={"displayModeBar": False},
                theme=None,
            )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        san = _section("What-if directional sanity", "whatif_sanity")
        if san is not None:
            st.dataframe(
                san,
                hide_index=True,
                width="stretch",
                column_config={
                    "pass_rate": st.column_config.ProgressColumn("Pass rate", min_value=0, max_value=1, format="%.2f")
                },
            )
    with c2:
        kit.md('<h3 style="margin:18px 0 6px;font-size:20px;font-weight:600">Attribution and what-if agreement</h3>')
        agree, total = _cohort_agreement(common.mode_key(), common.now().isoformat())
        if total:
            kit.readouts(
                [
                    kit.Readout(
                        "Agreement rate, cohort at the replay time",
                        f"{agree / total * 100:.0f}",
                        "%",
                        f"{agree} of {total} checks agree",
                    )
                ]
            )
        else:
            kit.empty_state("No testable levers at this time.", "Move the replay time to after a meal.")
        kit.caption(
            "On the mock, attributions go through the same pipeline as the what-if engine, so about 100% is "
            "expected; anything lower flags an inconsistency (this check caught one during development). "
            "The real model is where it matters most."
        )

    kit.md('<h3 style="margin:24px 0 6px;font-size:20px;font-weight:600">Data provenance register</h3>')
    st.dataframe(tables.PROVENANCE_REGISTER, hide_index=True, width="stretch")

    kit.md('<h3 style="margin:24px 0 6px;font-size:20px;font-weight:600">Honest limits</h3>')
    kit.md(
        '<ul class="mt-prose" style="font-size:16px;line-height:26px">'
        + "".join(f"<li>{kit.esc(lim)}</li>" for lim in tables.LIMITATIONS)
        + "</ul>"
    )
    thr, calibrated = common.thresholds()
    if not calibrated:
        kit.caption("Alert thresholds in use are uncalibrated placeholders: " + thr.get("stratum_rule", ""))
    kit.caption(
        "Positioning: an open, ablation-tested, EHR-conditioned T2D digital twin with a clinician-facing "
        "what-if lab. We do not claim to outperform commercial products."
    )


common.guard(render)
