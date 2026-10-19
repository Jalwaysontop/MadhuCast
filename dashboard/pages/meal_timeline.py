"""F10 Meal timeline: the day's meals with macros, GL and provenance, aligned with CGM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.components import charts, common
from src.serving import backend
from src.serving import timeseries as ts

common.init_session()
GL_SCALE = 80  # a GL bar is full at 80


def render() -> None:
    state = common.get_state()
    meals = backend.day_meals(state.profile.patient_id, state.now)
    st.title("Meals")
    st.caption(
        "Meal glycaemic load (GL) = carbohydrate × GI / 100, summed over items. Low GL < 10, high GL > 20 per "
        "item. Our food table is a small, openly sourced seed. It is not comprehensive."
    )
    if not meals:
        st.info("No meals logged for this day.")
        return
    st.plotly_chart(charts.day_figure(state.history, meals, state.now), width="stretch", key="day-cgm")

    for i, m in enumerate(meals):
        later = m.t > state.now
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.2, 3, 1.2])
            with c1:
                st.markdown(f"**{ts.hhmm(m.t)}** {'*(after now)*' if later else ''}")
                common.md(common.prov_badge(m.provenance))
            with c2:
                st.markdown(f"**{m.label}**")
                if m.items:
                    st.caption(" + ".join(f"{it['portion']:g} {it['unit']} {it['food']}" for it in m.items))
                common.md(
                    "".join(
                        f"<span class='mt-chip'>{lab} {v:.0f} g</span>"
                        for lab, v in (
                            ("carbs", m.carbs_g),
                            ("protein", m.protein_g),
                            ("fat", m.fat_g),
                            ("fibre", m.fibre_g),
                        )
                    )
                )
                for flag in m.assumption_flags or []:
                    st.caption(f"⚠ assumption: {flag}")
            with c3:
                st.progress(min(m.gl / GL_SCALE, 1.0), text=f"GL {m.gl:.0f}")
                if st.button(
                    "Open in What-if",
                    key=f"whatif-{i}",
                    disabled=later,
                    icon=":material/science:",
                    help="Meals after the replay time can't be opened yet" if later else None,
                ):
                    st.session_state.whatif_meal = i
                    st.switch_page("pages/whatif_lab.py")


common.guard(render)
