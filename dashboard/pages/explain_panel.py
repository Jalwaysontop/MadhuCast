"""F6 Why panel: lead sentence, grouped diverging drivers, agreement checks, attention strip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.components import charts, common, kit
from src.explain import templates as T
from src.serving import backend

common.init_session()
GROUP_TITLE = {"EHR": "Patient record"}


def _sentence_case(s: str) -> str:
    return s[:1].upper() + s[1:]


def render() -> None:
    state, fc = common.run()
    mode = common.theme()
    st.title("Why this forecast?")
    kit.md(
        f'<div style="display:flex;gap:8px;margin:-6px 0 12px">{common.prov_badge("MODEL_ESTIMATE")} '
        f"{common.prov_badge(fc.provenance)}</div>"
    )
    sentences = T.explain(fc, k=2)
    kit.lead(" ".join(sentences))
    kit.caption(
        T.CAPTION
        + " Sentences come from fixed templates, not a language model, so they cannot invent claims. "
        + T.risk_sentence(fc)
    )
    st.write("")

    left, right = st.columns([62, 38], gap="medium")
    with left, st.container(key="panel-drivers"):
        st.subheader("Drivers of the predicted 2-hour peak")
        if not fc.shap_values:
            kit.empty_state(
                "No attributions for this forecast.",
                (
                    "The real model is an XGBoost spike classifier shipped without SHAP values, so nothing is "
                    "attributed. " + T.FALLBACK.split(". ", 1)[1]
                    if backend.model_mode() == "real"
                    else T.FALLBACK
                ),
            )
        else:
            vals = fc.feature_values or {}
            base = fc.feature_baselines or {}
            groups: dict[str, list[tuple[str, str, float]]] = {}
            for key, eff in sorted(fc.shap_values.items(), key=lambda kv: -abs(kv[1])):
                if abs(eff) < 1.0:
                    continue
                spec = T.spec(key)
                value = T.fmt_value(key, vals[key]) if key in vals else ""
                if key in base and key in vals and round(base[key], spec.decimals) != round(vals[key], spec.decimals):
                    value += f", usual {T.fmt_value(key, base[key])}"
                groups.setdefault(GROUP_TITLE.get(spec.group, spec.group), []).append(
                    (_sentence_case(spec.label.replace("Recent meal ", "")), value, eff)
                )
            order = ["Meal", "Activity", "Sleep (proxy)", "Patient record", "Recent glucose", "Other"]
            ordered = [(g, groups[g]) for g in order if g in groups]
            kit.md(kit.driver_bars(ordered, max(abs(v) for v in fc.shap_values.values())))
            quiet = [T.spec(k).label for k, v in fc.shap_values.items() if abs(v) < 1.0]
            kit.caption(
                "mg/dL change in the predicted peak, relative to a neutral reference."
                + (f" Little or no effect today: {', '.join(quiet).lower()}." if quiet else "")
            )

    with right:
        with st.container(key="panel-agree"):
            st.subheader("Do attributions agree with the what-if engine?")
            res = T.agreement_check(state, fc, lambda s, sc: backend.predict(s, sc))
            if not fc.shap_values:
                kit.empty_state("Nothing to test.", "This forecast has no attributions to check against what-if.")
            elif not res:
                kit.empty_state(
                    "Nothing to test for this forecast.",
                    "No attributed what-if lever moves the peak here (for example, no walking to vary).",
                )
            else:
                kit.md(
                    "".join(
                        kit.check_row(
                            r.agree,
                            r.label,
                            f"Tested {r.lever}: {r.observed_delta:+.1f} mg/dL, "
                            f"expected {'a rise' if r.expected_sign > 0 else 'a fall'}",
                        )
                        for r in res
                    )
                )
                kit.caption(
                    f"Agreement {sum(r.agree for r in res)} of {len(res)} for this patient. "
                    "The cohort-wide rate is on the Model trust page."
                )
        st.write("")
        with st.container(key="panel-attn"):
            st.subheader("Attention, past 4 hours")
            if fc.attention_weights:
                st.plotly_chart(
                    charts.attention_figure(fc.attention_weights, state.now, mode=mode),
                    width="stretch",
                    key="attn",
                    config={"displayModeBar": False},
                    theme=None,
                )
            else:
                kit.empty_state("No attention weights.", "The model did not return them for this forecast.")
            kit.md(
                f'<div class="mt-caption" style="display:flex;gap:6px;align-items:center">{kit.icon("approximately-equal", 14)}'
                f"{kit.esc(T.ATTENTION_CAPTION)}</div>"
            )


common.guard(render)
