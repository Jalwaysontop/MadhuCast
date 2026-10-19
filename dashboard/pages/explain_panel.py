"""F6 Why panel: signed attributions, template sentences, attention heatmap, reliability check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import charts, common
from src.explain import templates as T
from src.serving import backend

common.init_session()


def render() -> None:
    state, fc = common.run()
    st.title("Why this forecast?")
    st.caption(T.CAPTION + " Attributions describe how each input moves the model's predicted 2-hour peak.")
    common.md(common.prov_badge(fc.provenance))
    st.write(T.risk_sentence(fc))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Top contributors")
        if fc.shap_values:
            st.plotly_chart(charts.attribution_figure(fc.shap_values, k=6), width="stretch", key="attr")
            groups: dict[str, float] = {}
            for k, v in fc.shap_values.items():
                groups[T.spec(k).group] = groups.get(T.spec(k).group, 0) + v
            st.caption(
                "By group: "
                + " · ".join(f"{g} {v:+.0f}" for g, v in sorted(groups.items(), key=lambda kv: -abs(kv[1])))
            )
        else:
            st.info(T.FALLBACK)
    with right:
        st.markdown("#### In plain language")
        for s in T.explain(fc, k=4):
            st.markdown(f"- {s}")
        st.caption("Sentences come from deterministic templates, not a language model, so they cannot invent claims.")

    st.markdown("#### Attention over the past 4 hours")
    if fc.attention_weights:
        st.plotly_chart(charts.attention_figure(fc.attention_weights, state.now), width="stretch", key="attn")
    else:
        st.write("The model did not return attention weights.")
    st.caption(T.ATTENTION_CAPTION)

    st.markdown("#### Reliability check: do attributions agree with the what-if engine?")
    st.caption(
        "For the top attributed levers, we run the matching what-if (which increases that input) and check "
        "that the predicted peak moves in the direction the attribution implies."
    )
    res = T.agreement_check(state, fc, lambda s, sc: backend.predict(s, sc))
    if not res:
        st.write("No attributed what-if levers to test for this forecast (e.g. no walking to vary).")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": r.label,
                        "What-if tested": r.lever,
                        "Expected direction": "↑ peak" if r.expected_sign > 0 else "↓ peak",
                        "Observed Δ peak": f"{r.observed_delta:+.1f} mg/dL",
                        "Agree": "✓" if r.agree else "✗",
                    }
                    for r in res
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Agreement {sum(r.agree for r in res)}/{len(res)} for this patient. The cohort-wide rate is on the "
            "Model Trust page."
        )


common.guard(render)
