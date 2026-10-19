"""What-if levers applied to a model input: pure functions, no Streamlit (ADR-005).

``apply(state, scenario, resolve)`` returns a new PatientState in which:
  * meal levers (swap, portion, time shift) rewrite only the target meal;
  * the walk lever adds an ActivityBlock (and raises METs/HR for any past part of it);
  * the sleep lever rewrites only the previous-night window of the sleep proxy (and the
    METs/HR it is derived from).
Medication is deliberately not a lever.

If a meal is shifted earlier than ``now``, the evaluation time moves back to the new
meal time so the 2-hour forecast still covers that meal's response.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.contracts import MIN_HISTORY_MIN, ActivityBlock, Forecast, Meal, PatientState, Scenario
from src.serving import timeseries as ts

WALK_METS = 3.5
RICE_TO_MILLET_ROTI = 2.0  # 1 katori cooked rice ≈ 2 millet rotis by carbohydrate (~40 g)
Resolver = Callable[[list[dict[str, Any]]], Meal]


def target_index(state: PatientState, scenario: Scenario) -> int | None:
    if not state.meals:
        return None
    if scenario.target_meal_index is not None:
        idx = scenario.target_meal_index
    elif scenario.meal_swap and "meal_index" in scenario.meal_swap:
        idx = int(scenario.meal_swap["meal_index"])
    else:
        past = [i for i, m in enumerate(state.meals) if m.t <= state.now]
        idx = past[-1] if past else len(state.meals) - 1
    if not 0 <= idx < len(state.meals):
        raise ValueError(f"meal index {idx} out of range (0..{len(state.meals) - 1})")
    return idx


def _scale(meal: Meal, k: float) -> Meal:
    items = [{**i, "portion": round(float(i.get("portion", 1)) * k, 2)} for i in meal.items] if meal.items else None
    return meal.model_copy(update={
        "carbs_g": meal.carbs_g * k, "protein_g": meal.protein_g * k, "fat_g": meal.fat_g * k,
        "fibre_g": meal.fibre_g * k, "gl": meal.gl * k, "items": items,
    })


def _rewrite_sleep(hist: pd.DataFrame, now: datetime, hours: float) -> pd.DataFrame:
    start, end = ts.night_window(now)
    t = pd.to_datetime(hist["t"], utc=True)
    win = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
    if not win.any():
        return hist
    asleep_rows = hist.loc[win & (hist["sleep_proxy"] >= 0.5), "t"]
    wake = (pd.to_datetime(asleep_rows, utc=True).max() + pd.Timedelta(minutes=ts.STEP_MIN)
            if len(asleep_rows) else pd.Timestamp(start) + pd.Timedelta(hours=10, minutes=40))
    onset = wake - pd.Timedelta(hours=hours)
    new_sleep = ((t >= onset) & (t < wake)).astype(float)
    changed = win & (new_sleep != hist["sleep_proxy"])
    hist = hist.copy()
    hist.loc[win, "sleep_proxy"] = new_sleep[win]
    hist.loc[changed, "mets"] = new_sleep[changed].map({1.0: 0.95, 0.0: 1.3})
    hist.loc[changed, "hr"] = 52 + 10 * hist.loc[changed, "mets"]
    return hist


def apply(state: PatientState, scenario: Scenario, resolve: Resolver) -> PatientState:
    if scenario.is_baseline():
        return state
    meals = list(state.meals)
    hist = state.history.copy()
    now = state.now
    planned = list(state.planned_activity)
    idx = target_index(state, scenario)
    anchor_t = now

    if idx is not None:
        m = meals[idx]
        if scenario.meal_swap and scenario.meal_swap.get("items"):
            new = resolve(scenario.meal_swap["items"])
            prefix = m.label.split(":")[0]
            m = new.model_copy(update={"t": m.t, "label": f"{prefix} (swap): {new.label}"})
        if scenario.portion_scale != 1.0:
            m = _scale(m, scenario.portion_scale)
        if scenario.meal_time_shift_min:
            m = m.model_copy(update={"t": m.t + timedelta(minutes=scenario.meal_time_shift_min)})
            if scenario.meal_time_shift_min < 0 and m.t < now:
                earliest = pd.Timestamp(hist["t"].iloc[0]).to_pydatetime() + timedelta(minutes=MIN_HISTORY_MIN)
                now = max(ts.floor5(m.t), earliest)
        meals[idx] = m
        anchor_t = m.t

    if scenario.walk_minutes:
        block = ActivityBlock(start=anchor_t + timedelta(minutes=scenario.walk_start_offset_min),
                              minutes=scenario.walk_minutes, mets=WALK_METS)
        t = pd.to_datetime(hist["t"], utc=True)
        on = (t >= pd.Timestamp(block.start)) & (t < pd.Timestamp(block.start) + pd.Timedelta(minutes=block.minutes))
        hist.loc[on, "mets"] = hist.loc[on, "mets"].clip(lower=WALK_METS)
        hist.loc[on, "hr"] = hist.loc[on, "hr"].clip(lower=52 + 10 * WALK_METS)
        planned.append(block)

    if scenario.prior_sleep_hours is not None:
        hist = _rewrite_sleep(hist, now, scenario.prior_sleep_hours)

    if now != state.now:
        hist = hist[pd.to_datetime(hist["t"], utc=True) <= pd.Timestamp(now)].reset_index(drop=True)
        meals = [mm for i, mm in enumerate(meals) if mm.t <= now or i == idx]

    return PatientState(
        profile=state.profile, history=hist.reset_index(drop=True), meals=meals, now=now,
        data_quality=ts.data_quality(hist, now) if now != state.now else state.data_quality,
        stream_last_seen=ts.stream_last_seen(hist, now) if now != state.now else state.stream_last_seen,
        uplink_mode=state.uplink_mode, planned_activity=planned,
    )


# --------------------------------------------------------------------------- comparison


@dataclass(frozen=True)
class Comparison:
    peak_base: float
    peak_scen: float
    d_peak: float
    d_time_above_180: int
    d_spike_prob: float
    d_hypo_prob: float
    bands_overlap: bool
    scen_band_at_peak: tuple[float, float]
    base_band_at_peak: tuple[float, float]

    @property
    def overlap_note(self) -> str | None:
        if self.bands_overlap:
            return "Bands overlap at the peak, so the difference may not be meaningful."
        return None


def compare(base_fc: Forecast, scen_fc: Forecast, base_now_g: float | None = None,
            scen_now_g: float | None = None) -> Comparison:
    cb, cs = ts.forecast_curve(base_fc, base_now_g), ts.forecast_curve(scen_fc, scen_now_g)
    ib, is_ = int(cb["q50"].idxmax()), int(cs["q50"].idxmax())
    bb = (float(cb.at[ib, "q10"]), float(cb.at[ib, "q90"]))
    bs = (float(cs.at[is_, "q10"]), float(cs.at[is_, "q90"]))
    return Comparison(
        peak_base=float(cb.at[ib, "q50"]), peak_scen=float(cs.at[is_, "q50"]),
        d_peak=float(cs.at[is_, "q50"] - cb.at[ib, "q50"]),
        d_time_above_180=ts.time_above(cs) - ts.time_above(cb),
        d_spike_prob=scen_fc.spike_prob - base_fc.spike_prob,
        d_hypo_prob=scen_fc.hypo_prob - base_fc.hypo_prob,
        bands_overlap=bs[0] <= bb[1] and bb[0] <= bs[1],
        scen_band_at_peak=bs, base_band_at_peak=bb,
    )


# --------------------------------------------------------------------------- presets


def rice_to_millet(items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Swap rice items for millet roti at equal carbohydrate; None if the meal has no rice."""
    if not any("rice" in i["food"] and "biryani" not in i["food"] for i in items):
        return None
    out = []
    for i in items:
        if "rice" in i["food"] and "biryani" not in i["food"]:
            out.append({"food": "millet roti (bajra)", "portion": round(float(i["portion"]) * RICE_TO_MILLET_ROTI),
                        "unit": "piece"})
        else:
            out.append(dict(i))
    return out


PRESETS = ("Rice → millet roti", "+20 min post-meal walk", "Short sleep (5 h)", "Combined")


def preset(name: str, state: PatientState, meal_index: int | None) -> Scenario:
    """Scenario for a named preset. Raises ValueError when the preset does not apply."""
    swap = None
    if name in ("Rice → millet roti", "Combined"):
        if meal_index is None or not state.meals[meal_index].items:
            raise ValueError("no itemised meal selected")
        items = rice_to_millet(state.meals[meal_index].items)
        if items is None and name == "Rice → millet roti":
            raise ValueError("selected meal contains no rice")
        swap = {"meal_index": meal_index, "items": items} if items else None
    if name == "Rice → millet roti":
        return Scenario(meal_swap=swap, target_meal_index=meal_index)
    if name == "+20 min post-meal walk":
        return Scenario(walk_minutes=20, target_meal_index=meal_index)
    if name == "Short sleep (5 h)":
        return Scenario(prior_sleep_hours=5.0, target_meal_index=meal_index)
    if name == "Combined":
        return Scenario(meal_swap=swap, walk_minutes=20, target_meal_index=meal_index)
    raise ValueError(f"unknown preset {name!r}")
