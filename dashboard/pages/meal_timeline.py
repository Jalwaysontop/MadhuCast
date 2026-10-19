"""F10 Meal timeline: the day's meals with macros, GL and provenance, aligned with CGM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.components import charts, common, kit
from src.serving import backend
from src.serving import timeseries as ts

common.init_session()
GL_SCALE = 80  # a GL bar is full at 80


def render() -> None:
    state = common.get_state()
    meals = backend.day_meals(state.profile.patient_id, state.now)
    st.title("Meals")
    kit.caption(
        "Glycaemic load (GL) = carbohydrate × GI ÷ 100, summed over items; under 10 is low, over 20 is high. "
        "Our food table is a small, openly sourced seed, not a comprehensive database."
    )
    if not meals:
        kit.empty_state("No meals logged for this day.", "Person 3's meals file for this patient is empty or missing.")
        return
    with st.container(key="panel-day"):
        st.plotly_chart(
            charts.day_figure(state.history, meals, state.now, mode=common.theme()),
            width="stretch",
            key="day-cgm",
            config={"displayModeBar": False},
            theme=None,
        )
    st.write("")
    for i, m in enumerate(meals):
        later = m.t > state.now
        with st.container(key=f"panel-meal-{i}"):
            c1, c2, c3 = st.columns([1, 3.2, 1.3], vertical_alignment="center")
            with c1:
                kit.md(
                    f'<div style="font-size:20px;font-weight:700" class="mt-num">{ts.hhmm(m.t)}</div>'
                    f'<div class="mt-caption">{"after now" if later else "eaten"}</div>{common.prov_badge(m.provenance)}'
                )
            with c2:
                items = " + ".join(f"{it['portion']:g} {it['unit']} {kit.deva(it['food'])}" for it in (m.items or []))
                flags = "".join(
                    f'<div class="mt-caption" style="color:var(--mt-warning)">⚠ Assumption: {kit.esc(f)}</div>'
                    for f in m.assumption_flags or []
                )
                kit.md(
                    f'<div style="font-weight:600">{kit.esc(m.label)}</div><div class="mt-muted">{items}</div>'
                    f'<div style="margin-top:4px">{kit.meal_chips(m)}</div>{flags}'
                )
            with c3:
                kit.md(
                    f'<div class="mt-caption">Glycaemic load <b style="color:var(--mt-ink)">{m.gl:.0f}</b></div>'
                    f'{kit.bar(m.gl / GL_SCALE, "warning" if m.gl > 40 else "jamun")}'
                )
                if st.button(
                    "Open in What-if",
                    key=f"whatif-{i}",
                    disabled=later,
                    icon=":material/tune:",
                    help="Meals after the replay time can't be opened yet" if later else None,
                ):
                    st.session_state.whatif_meal = i
                    st.switch_page("pages/whatif_lab.py")


common.guard(render)
