"""F4 What-if lab: levers → scenario.apply (pure) → predict; baseline vs scenario side by side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import charts, common
from src.contracts import Scenario
from src.explain import templates as T
from src.serving import backend, scenario
from src.serving import timeseries as ts

common.init_session()
MAX_SAVED = 4
DEFAULTS = {"lv_portion": 1.0, "lv_walk_on": False, "lv_walk_min": 20, "lv_walk_offset": 15, "lv_shift": 0,
            "lv_sleep_on": False, "lv_sleep": 7.0}


def _reset_levers(items: list[dict]) -> None:
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss[k] = v
    ss.lv_items = [{"food": i["food"], "portion": float(i["portion"])} for i in items]
    ss.lv_items_ver = ss.get("lv_items_ver", 0) + 1


def _apply_preset(sc: Scenario, items: list[dict]) -> None:
    _reset_levers(items)
    ss = st.session_state
    ss.lv_portion = sc.portion_scale
    if sc.walk_minutes:
        ss.lv_walk_on, ss.lv_walk_min, ss.lv_walk_offset = True, sc.walk_minutes, sc.walk_start_offset_min
    ss.lv_shift = sc.meal_time_shift_min
    if sc.prior_sleep_hours is not None:
        ss.lv_sleep_on, ss.lv_sleep = True, sc.prior_sleep_hours
    if sc.meal_swap and sc.meal_swap.get("items"):
        ss.lv_items = [{"food": i["food"], "portion": float(i["portion"])} for i in sc.meal_swap["items"]]


def _describe(sc: Scenario, swapped: bool) -> str:
    parts = []
    if swapped:
        parts.append("meal swap")
    if sc.portion_scale != 1:
        parts.append(f"portion ×{sc.portion_scale:g}")
    if sc.walk_minutes:
        parts.append(f"+{sc.walk_minutes} min walk")
    if sc.meal_time_shift_min:
        parts.append(f"meal {sc.meal_time_shift_min:+d} min")
    if sc.prior_sleep_hours is not None:
        parts.append(f"sleep {sc.prior_sleep_hours:g} h")
    return ", ".join(parts) or "baseline"


def render() -> None:
    ss = st.session_state
    st.title("What-if lab")
    common.disclaimer()

    state_now = common.get_state()
    meals = [m for m in backend.day_meals(state_now.profile.patient_id, state_now.now) if m.t <= state_now.now]
    if not meals:
        st.info("No meals before the replay time. Move the time slider later in the day.")
        return
    default = ss.whatif_meal if isinstance(ss.whatif_meal, int) and ss.whatif_meal < len(meals) else len(meals) - 1
    mlabels = [f"{ts.hhmm(m.t)} · {m.label} · GL {m.gl:.0f}" for m in meals]
    sel = mlabels.index(st.selectbox("Meal to explore", mlabels, index=default))
    ss.whatif_meal = sel
    meal = meals[sel]
    base_state, base_fc = common.run(at=meal.t)
    idx = next(i for i, m in enumerate(base_state.meals) if m.t == meal.t)
    items = meal.items or []

    meal_key = f"{base_state.profile.patient_id}|{meal.t.isoformat()}"
    if ss.get("lv_meal_key") != meal_key:
        _reset_levers(items)
        ss.lv_meal_key = meal_key
    st.caption(f"Evaluated at the meal time ({ts.hhmm(meal.t)} IST) so the 2-hour window covers this meal's response.")

    st.markdown("**Presets**")
    pcols = st.columns(len(scenario.PRESETS) + 1)
    for col, name in zip(pcols, scenario.PRESETS):
        if col.button(name, key=f"preset-{name}", width="stretch"):
            try:
                _apply_preset(scenario.preset(name, base_state, idx), items)
                st.rerun()
            except ValueError as e:
                st.toast(f"{name}: not applicable ({e}).", icon="ℹ️")
    if pcols[-1].button("Reset", key="preset-reset", icon=":material/restart_alt:", width="stretch"):
        _reset_levers(items)
        st.rerun()

    db = common.food_db()
    units = dict(zip(db["food"], db["portion_unit"]))
    with st.form("levers", border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Meal swap** (edit foods or portions)")
            edited = st.data_editor(
                pd.DataFrame(ss.lv_items), key=f"items-{ss.lv_items_ver}", num_rows="dynamic", hide_index=True,
                width="stretch",
                column_config={"food": st.column_config.SelectboxColumn("Food", options=sorted(units), required=True),
                               "portion": st.column_config.NumberColumn("Portion", min_value=0.25, max_value=10,
                                                                        step=0.25, required=True)})
            st.slider("Portion scale", 0.25, 2.0, step=0.05, key="lv_portion")
        with c2:
            st.toggle("Post-meal walk", key="lv_walk_on")
            w1, w2 = st.columns(2)
            w1.slider("Walk minutes", 5, 60, step=5, key="lv_walk_min")
            w2.slider("Starts after meal (min)", 0, 60, step=5, key="lv_walk_offset")
            st.slider("Meal-time shift (min)", -120, 120, step=15, key="lv_shift")
            st.toggle("Override previous-night sleep (proxy)", key="lv_sleep_on")
            st.slider("Sleep hours", 3.0, 10.0, step=0.5, key="lv_sleep")
        st.form_submit_button("Run scenario", type="primary", icon=":material/play_arrow:")

    rows = [r for r in edited.dropna().to_dict("records") if r.get("food")]
    new_items = [{"food": r["food"], "portion": float(r["portion"]), "unit": units[r["food"]]} for r in rows]
    orig = [(i["food"], float(i["portion"])) for i in items]
    swapped = bool(new_items) and [(i["food"], i["portion"]) for i in new_items] != orig
    sc = Scenario(
        meal_swap={"meal_index": idx, "items": new_items} if swapped else None,
        portion_scale=ss.lv_portion, walk_minutes=ss.lv_walk_min if ss.lv_walk_on else 0,
        walk_start_offset_min=ss.lv_walk_offset, meal_time_shift_min=ss.lv_shift,
        prior_sleep_hours=ss.lv_sleep if ss.lv_sleep_on else None, target_meal_index=idx,
    )
    if swapped:
        try:
            resolved = backend.resolve_meal(new_items)
            st.caption(f"Swap resolves to: {resolved.label} · GL {resolved.gl:.0f} (was {meal.gl:.0f})"
                       + ("".join(f" · ⚠ {f}" for f in resolved.assumption_flags or [])))
        except ValueError as e:
            st.error(f"Meal swap could not be resolved: {e}")
            return

    scen_state, scen_fc = common.run(at=meal.t, scenario=sc)
    label = _describe(sc, swapped)
    g_b = ts.glucose_now(base_state.history, base_state.now)
    g_s = ts.glucose_now(scen_state.history, scen_state.now)
    cmp = scenario.compare(base_fc, scen_fc, g_b, g_s)

    st.markdown(f"#### Baseline vs scenario: *{label}*")
    d = st.columns(4)
    d[0].metric("Δ peak", f"{cmp.d_peak:+.0f} mg/dL", delta=f"{cmp.peak_base:.0f} → {cmp.peak_scen:.0f}",
                delta_color="off")
    d[1].metric("Δ time above 180", f"{cmp.d_time_above_180:+d} min")
    d[2].metric("Δ spike probability", f"{cmp.d_spike_prob:+.0%}")
    d[3].metric("Δ hypo probability", f"{cmp.d_hypo_prob:+.0%}")
    if cmp.overlap_note:
        st.info(cmp.overlap_note, icon="↔️")
    if not sc.is_baseline():
        st.write(T.counterfactual_sentence(f"the scenario ({label})", cmp.base_band_at_peak,
                                           cmp.scen_band_at_peak, cmp.d_peak))
    if scen_state.now != base_state.now:
        st.caption(f"The earlier meal moves the scenario's evaluation time to {ts.hhmm(scen_state.now)}.")

    yr = charts.y_range_for((base_state, base_fc), (scen_state, scen_fc))
    if st.toggle("Overlay on one chart", key="overlay"):
        fig = charts.forecast_figure(base_state, base_fc, label="Baseline", y_range=yr,
                                     compare=(scen_state, scen_fc, "Scenario", charts.GREEN))
        st.plotly_chart(fig, width="stretch", key="whatif-overlay")
    else:
        b, s = st.columns(2)
        with b:
            st.plotly_chart(charts.forecast_figure(base_state, base_fc, label="Baseline", y_range=yr, height=380,
                                                   title="Baseline", show_90=False), width="stretch", key="wb")
        with s:
            st.plotly_chart(charts.forecast_figure(scen_state, scen_fc, label="Scenario", color=charts.GREEN,
                                                   y_range=yr, height=380, title=f"Scenario: {label}",
                                                   show_90=False), width="stretch", key="ws")
    common.md(f"{common.prov_badge(scen_fc.provenance)} <span class='mt-delta-note'>Bands are the model's 80% "
              "range. Medication is intentionally not a lever.</span>")

    st.divider()
    saved = ss.saved_scenarios
    c1, c2 = st.columns([1, 1])
    if c1.button("Save scenario for comparison", icon=":material/bookmark_add:",
                 disabled=sc.is_baseline() or len(saved) >= MAX_SAVED):
        saved.append({"label": label, "scenario": sc.model_dump_json(), "meal_t": meal.t.isoformat(),
                      "pid": base_state.profile.patient_id, "d_peak": cmp.d_peak})
        st.rerun()
    if c2.button("Clear saved", icon=":material/delete:", disabled=not saved):
        ss.saved_scenarios = []
        st.rerun()
    if saved:
        st.markdown(f"#### Saved scenarios ({len(saved)}/{MAX_SAVED})")
        cols = st.columns(2)
        for i, item in enumerate(saved):
            from datetime import datetime

            at = datetime.fromisoformat(item["meal_t"])
            bs, bf = common.run(p=item["pid"], at=at)
            ss_, sf = common.run(p=item["pid"], at=at, scenario=Scenario.model_validate_json(item["scenario"]))
            fig = charts.forecast_figure(bs, bf, label="Baseline", height=260, show_meals=False, show_90=False,
                                         compare=(ss_, sf, "Scenario", charts.GREEN),
                                         y_range=charts.y_range_for((bs, bf), (ss_, sf)),
                                         title=f"{item['pid']} · {item['label']} · Δpeak {item['d_peak']:+.0f}")
            fig.update_layout(showlegend=False)
            cols[i % 2].plotly_chart(fig, width="stretch", key=f"saved-{i}")


common.guard(render)
