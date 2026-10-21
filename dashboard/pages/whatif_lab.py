"""F4 What-if lab: levers → scenario.apply (pure) → predict. Baseline and scenario overlaid on one
Twin Curve that morphs when a lever changes (the one orchestrated motion)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components import charts, common, kit, twin_morph
from dashboard.theme.tokens import FOOD_HINDI
from src.contracts import Scenario
from src.explain import templates as T
from src.serving import backend, scenario
from src.serving import timeseries as ts

common.init_session()
MAX_SAVED = 4
DEFAULTS = {
    "lv_portion": 1.0,
    "lv_walk_on": False,
    "lv_walk_min": 20,
    "lv_walk_offset": 15,
    "lv_shift": 0,
    "lv_sleep_on": False,
    "lv_sleep": 7.0,
}
PRESET_LABEL = {
    "Rice → millet roti": "Rice → millet roti",
    "+20 min post-meal walk": "+20 min walk",
    "Short sleep (5 h)": "Short sleep",
    "Combined": "Combined",
}


def food_label(food: str) -> str:
    """'बाजरा रोटी  millet roti (bajra)': Devanagari name beside the English key (display only)."""
    hi = FOOD_HINDI.get(food)
    return f"{hi}  {food}" if hi else food


def _reset_levers(items: list[dict]) -> None:
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss[k] = v
    ss.lv_items = [{"food": food_label(i["food"]), "portion": float(i["portion"])} for i in items]
    ss.lv_items_ver = ss.get("lv_items_ver", 0) + 1
    ss.lv_preset = None


def _apply_preset(name: str, sc: Scenario, items: list[dict]) -> None:
    _reset_levers(items)
    ss = st.session_state
    ss.lv_preset = name
    ss.lv_portion = sc.portion_scale
    if sc.walk_minutes:
        ss.lv_walk_on, ss.lv_walk_min, ss.lv_walk_offset = True, sc.walk_minutes, sc.walk_start_offset_min
    ss.lv_shift = sc.meal_time_shift_min
    if sc.prior_sleep_hours is not None:
        ss.lv_sleep_on, ss.lv_sleep = True, sc.prior_sleep_hours
    if sc.meal_swap and sc.meal_swap.get("items"):
        ss.lv_items = [{"food": food_label(i["food"]), "portion": float(i["portion"])} for i in sc.meal_swap["items"]]


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


def _signed(v: float, digits: int = 0) -> str:
    r = round(v, digits)
    if r == 0:
        return "0"
    return f"{'+' if r > 0 else '−'}{abs(r):.{digits}f}"


def render() -> None:
    ss = st.session_state
    mode = common.theme()
    st.title("What-if lab")
    state_now = common.get_state()
    meals = [m for m in backend.day_meals(state_now.profile.patient_id, state_now.now) if m.t <= state_now.now]
    if not meals:
        kit.empty_state("No meals before the replay time.", "Move “Replay time” in the sidebar later in the day.")
        return
    default = ss.whatif_meal if isinstance(ss.whatif_meal, int) and ss.whatif_meal < len(meals) else len(meals) - 1
    mlabels = [f"{ts.hhmm(m.t)}  {m.label}, GL {m.gl:.0f}" for m in meals]

    model_badge = "MOCK" if "MOCK" in backend.model_parts().values() else "MODEL_ESTIMATE"
    kit.md(
        f'<div style="display:flex;gap:8px;margin:-6px 0 12px">{common.prov_badge("MODEL_ESTIMATE")} '
        f"{common.prov_badge(model_badge)}</div>"
    )
    left, right = st.columns([6.5, 3.5], gap="medium")
    with right:
        tray = st.container(key="lever-tray")
    with tray:
        st.subheader("Levers")
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

    db = common.food_db()
    units = dict(zip(db["food"], db["portion_unit"]))
    label_to_food = {food_label(f): f for f in units}
    millet = next((f for f in scenario.MILLET_ROTI if f in units), scenario.MILLET_ROTI[0])

    # ------------------------------------------------------------ lever tray (the one shadow)
    with tray:
        kit.caption("Presets")
        pr = st.columns(2)
        for i, name in enumerate(scenario.PRESETS):
            if pr[i % 2].button(
                PRESET_LABEL[name],
                key=f"preset-{name}",
                width="stretch",
                type="primary" if ss.get("lv_preset") == name else "secondary",
            ):
                try:
                    _apply_preset(name, scenario.preset(name, base_state, idx, millet), items)
                    st.rerun()
                except ValueError as e:
                    st.toast(f"{PRESET_LABEL[name]} does not apply here: {e}.")
        with st.form("levers", border=False):
            kit.md('<div style="font-size:14px;font-weight:600;margin-top:6px">Meal swap</div>')
            edited = st.data_editor(
                pd.DataFrame(ss.lv_items),
                key=f"items-{ss.lv_items_ver}",
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "food": st.column_config.SelectboxColumn(
                        "Food", options=sorted(label_to_food), required=True, width=180
                    ),
                    "portion": st.column_config.NumberColumn(
                        "Portion",
                        min_value=0.25,
                        max_value=10,
                        step=0.25,
                        required=True,
                        help="In katori, roti (piece) or plate",
                        width=72,
                    ),
                },
            )
            st.slider("Portion", 0.25, 2.0, step=0.05, key="lv_portion", help="Scales every item (katori, roti)")
            st.toggle("Post-meal walk", key="lv_walk_on")
            w1, w2 = st.columns(2)
            w1.slider("Walk minutes", 5, 60, step=5, key="lv_walk_min")
            w2.slider("Starts after meal", 0, 60, step=5, key="lv_walk_offset", help="Minutes after the meal")
            st.slider("Meal-time shift (min)", -120, 120, step=15, key="lv_shift")
            st.toggle("Override sleep last night (proxy)", key="lv_sleep_on")
            st.slider(
                "Sleep hours (proxy)",
                3.0,
                10.0,
                step=0.5,
                key="lv_sleep",
                help="The sleep signal is a proxy derived from night METs and heart rate.",
            )
            st.form_submit_button("Run scenario", type="primary", icon=":material/play_arrow:", width="stretch")
        if st.button(
            "Reset levers", key="preset-reset", icon=":material/restart_alt:", width="stretch", type="tertiary"
        ):
            _reset_levers(items)
            st.rerun()

    rows = [r for r in edited.dropna().to_dict("records") if r.get("food")]
    new_items = [
        {
            "food": label_to_food.get(r["food"], r["food"]),
            "portion": float(r["portion"]),
            "unit": units[label_to_food.get(r["food"], r["food"])],
        }
        for r in rows
    ]
    orig = [(i["food"], float(i["portion"])) for i in items]
    swapped = bool(new_items) and [(i["food"], i["portion"]) for i in new_items] != orig
    sc = Scenario(
        meal_swap={"meal_index": idx, "items": new_items} if swapped else None,
        portion_scale=ss.lv_portion,
        walk_minutes=ss.lv_walk_min if ss.lv_walk_on else 0,
        walk_start_offset_min=ss.lv_walk_offset,
        meal_time_shift_min=ss.lv_shift,
        prior_sleep_hours=ss.lv_sleep if ss.lv_sleep_on else None,
        target_meal_index=idx,
    )
    resolved = None
    if swapped:
        try:
            resolved = backend.resolve_meal(new_items)
        except ValueError as e:
            with left:
                kit.empty_state(
                    "This swap cannot be resolved.", f"{e}. Pick a food from the list and a positive portion."
                )
            return

    scen_state, scen_fc = common.run(at=meal.t, scenario=sc)
    label = _describe(sc, swapped)
    cmp = scenario.compare(
        base_fc,
        scen_fc,
        ts.glucose_now(base_state.history, base_state.now),
        ts.glucose_now(scen_state.history, scen_state.now),
    )

    blind = backend.spike_blind_levers(sc, scen_fc)

    # ------------------------------------------------------------ comparison
    with left:
        with st.container(key="panel-whatif"):
            hl, hr = st.columns([3, 1], vertical_alignment="center")
            hl.subheader(f"Baseline vs scenario: {label}")
            split = hr.toggle("Split view", key="split")
            if split:
                yr = charts.y_range_for((base_state, base_fc), (scen_state, scen_fc))
                b, s = st.columns(2)
                b.plotly_chart(
                    charts.forecast_figure(
                        base_state,
                        base_fc,
                        label="Baseline, as eaten",
                        y_range=yr,
                        height=380,
                        show_90=False,
                        mode=mode,
                        title="Baseline, as eaten",
                    ),
                    width="stretch",
                    key="wb",
                    config={"displayModeBar": False},
                    theme=None,
                )
                s.plotly_chart(
                    charts.forecast_figure(
                        scen_state,
                        scen_fc,
                        label="Scenario",
                        color="scenario",
                        y_range=yr,
                        height=380,
                        show_90=False,
                        mode=mode,
                        title=f"Scenario: {label}",
                    ),
                    width="stretch",
                    key="ws",
                    config={"displayModeBar": False},
                    theme=None,
                )
            else:
                twin_morph.render(
                    twin_morph.payload(
                        base_state, base_fc, scen_state, scen_fc, cmp.d_peak, baseline_only=sc.is_baseline()
                    ),
                    mode=mode,
                    height=420,
                )
            kit.caption(
                f"Evaluated at the meal time ({ts.hhmm(meal.t)} IST) so the 2-hour window covers this meal. "
                "Bands are the model's 80% range."
            )
        st.write("")
        good = lambda v: "good" if v < -0.5 else ("bad" if v > 0.5 else None)  # noqa: E731
        kit.delta_strip(
            [
                kit.Readout(
                    "Δ peak",
                    _signed(cmp.d_peak),
                    "mg/dL",
                    f"{cmp.peak_base:.0f} → {cmp.peak_scen:.0f}",
                    tone=good(cmp.d_peak),
                ),
                kit.Readout(
                    "Δ time above 180",
                    _signed(cmp.d_time_above_180),
                    "min",
                    "of the next 2 hours",
                    tone=good(cmp.d_time_above_180),
                ),
                (
                    kit.Readout(
                        "Δ spike probability",
                        "n/a",
                        "",
                        "Classifier cannot see this lever",
                        foot_tone="warn",
                    )
                    if blind
                    else kit.Readout(
                        "Δ spike probability",
                        _signed(cmp.d_spike_prob * 100),
                        "pts",
                        f"{T.fmt_pct(base_fc.spike_prob)} → {T.fmt_pct(scen_fc.spike_prob)}",
                        tone=good(cmp.d_spike_prob * 100),
                    )
                ),
                kit.Readout(
                    "Δ hypo probability",
                    _signed(cmp.d_hypo_prob * 100),
                    "pts",
                    f"{T.fmt_pct(base_fc.hypo_prob)} → {T.fmt_pct(scen_fc.hypo_prob)}",
                    tone=good(cmp.d_hypo_prob * 100),
                ),
            ]
        )
        st.write("")
        notes = []
        if not sc.is_baseline():
            notes.append(
                cmp.overlap_note
                or "Bands separate at the peak, so for this model the difference is larger than its own uncertainty."
            )
            notes.append(
                T.counterfactual_sentence(
                    f"the scenario ({label})", cmp.base_band_at_peak, cmp.scen_band_at_peak, cmp.d_peak
                )
            )
        if resolved is not None:
            names = " + ".join(f"{i['portion']:g} {i['unit']} {food_label(i['food'])}" for i in new_items)
            flags = "".join(f" ⚠ {f}." for f in resolved.assumption_flags or [])
            notes.append(f"Swap resolves to {names}: GL {resolved.gl:.0f}, was {meal.gl:.0f}.{flags}")
        if blind:
            notes.append(
                f"Spike probability is not compared: it comes from the XGBoost classifier, which reads only the "
                f"past 6 hours, so it cannot see {' or '.join(blind)}. The curve, peak and time above 180 come "
                "from the MOCK predictor."
            )
        if scen_state.now != base_state.now:
            notes.append(f"The earlier meal moves the scenario's evaluation time to {ts.hhmm(scen_state.now)}.")
        for n in notes:
            kit.md(f'<p class="mt-prose" style="font-size:14px;margin:0 0 6px">{kit.esc(n)}</p>')
        common.disclaimer()

        st.write("")
        saved = ss.saved_scenarios
        c1, c2, _ = st.columns([1.4, 1, 2])
        if c1.button(
            "Save scenario for comparison",
            icon=":material/bookmark_add:",
            disabled=sc.is_baseline() or len(saved) >= MAX_SAVED,
        ):
            saved.append(
                {
                    "label": label,
                    "scenario": sc.model_dump_json(),
                    "meal_t": meal.t.isoformat(),
                    "pid": base_state.profile.patient_id,
                    "d_peak": cmp.d_peak,
                }
            )
            st.rerun()
        if c2.button("Clear saved", icon=":material/delete:", disabled=not saved, type="tertiary"):
            ss.saved_scenarios = []
            st.rerun()
        if saved:
            st.subheader(f"Saved scenarios ({len(saved)} of {MAX_SAVED})")
            cols = st.columns(2)
            for i, item in enumerate(saved):
                at = datetime.fromisoformat(item["meal_t"])
                bs, bf = common.run(p=item["pid"], at=at)
                ss_, sf = common.run(p=item["pid"], at=at, scenario=Scenario.model_validate_json(item["scenario"]))
                fig = charts.forecast_figure(
                    bs,
                    bf,
                    label="Baseline",
                    height=250,
                    show_meals=False,
                    show_90=False,
                    compare=(ss_, sf, "Scenario", "scenario"),
                    mode=mode,
                    y_range=charts.y_range_for((bs, bf), (ss_, sf)),
                    title=f"{item['pid']}: {item['label']}, Δ peak {_signed(item['d_peak'])} mg/dL",
                )
                fig.update_layout(showlegend=False)
                cols[i % 2].plotly_chart(
                    fig, width="stretch", key=f"saved-{i}", config={"displayModeBar": False}, theme=None
                )


common.guard(render)
