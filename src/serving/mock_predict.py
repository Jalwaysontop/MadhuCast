"""Physiologically plausible MOCK predictor (stand-in for P2's predict_api until Day 14).

    G(t) = G_basal + Σ_meals k_p · GL_i · γ(t − t_i) − walk_effect(t) + sleep_penalty + anchor(t)

* k_p rises with HbA1c and BMI (insulin-resistance proxy) and with sleep debt.
* γ is a peak-normalised gamma kernel, peaking ~45–60 min (later and flatter with
  fibre and fat; slower in T2D).
* walk_effect integrates METs above 1.5 with a 60-min decay, scaled by k_p, capped.
* anchor: the residual between the last observed CGM values and the model at ``now``
  decays slowly (~6 h), and the residual's 30-min slope is extrapolated (damped). This
  lets the mock follow drifts it does not model (P-011's evening decline).

Attributions are finite-difference perturbations through this same pipeline (including the
re-anchoring a what-if goes through), so the mock's explanations are faithful by construction:
a ground truth for testing templates and the attribution/what-if agreement check.
Output provenance is MOCK. This is NOT a clinical model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.contracts import HYPO_THRESHOLD, SPIKE_THRESHOLD, Forecast, PatientState, Provenance, Scenario
from src.serving import timeseries as ts

MODEL_VERSION = "mock-0.1.0"
FEATURE_LIB_VERSION = "mockfeat-0.1.0"
K0 = 1.0  # mg/dL per unit GL at the kernel peak, for a reference (healthy) person
REF = {"hba1c": 5.4, "bmi": 23.0, "sleep_hours_prev": 7.5}
WALK_METS = 3.5
HORIZON = 120
PAST = 240
ANCHOR_TAU, SLOPE_TAU = 360.0, 60.0
SLOPE_DEADBAND = 0.15  # mg/dL/min
G_MIN, G_MAX = 40.0, 400.0


@dataclass(frozen=True)
class Physio:
    fpg: float
    hba1c: float
    bmi: float
    sleep_h: float
    t2d: bool


def kp(p: Physio) -> float:
    debt = max(0.0, 7.0 - p.sleep_h)
    return K0 * (1 + 0.18 * (p.hba1c - 5.4)) * (1 + 0.015 * (p.bmi - 23)) * (1 + 0.04 * debt)


def basal(p: Physio) -> float:
    return p.fpg + 4.0 * max(0.0, 7.0 - p.sleep_h)


def meal_kernel(tau: np.ndarray, fibre: float, fat: float, t2d: bool) -> np.ndarray:
    mode = min(90.0, 45 + 0.8 * fibre + 0.25 * fat) + (10 if t2d else 0)
    a = 2.6 if t2d else 3.0
    x = np.clip(tau, 0, None) / mode
    k = np.where(tau > 0, x ** (a - 1) * np.exp(-(a - 1) * (x - 1)), 0.0)
    return k / (1 + 0.015 * fibre + 0.004 * fat)


def model_curve(p: Physio, t: np.ndarray, meals: np.ndarray, mets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Glucose on a uniform 5-min grid ``t`` (minutes). meals: rows [t_min, gl, fibre, fat]."""
    k = kp(p)
    meal = np.zeros_like(t, dtype=float)
    for tm, gl, fib, fat in np.atleast_2d(meals) if len(meals) else []:
        meal += k * gl * meal_kernel(t - tm, fib, fat, p.t2d)
    excess = np.clip(mets - 1.5, 0, None)
    a = np.zeros_like(t, dtype=float)
    decay = np.exp(-ts.STEP_MIN / 60)
    for i in range(len(t)):
        a[i] = (a[i - 1] * decay if i else 0.0) + excess[i] * ts.STEP_MIN
    walk = np.minimum(45.0, 0.6 * (k / K0) * a * (0.4 + 0.6 * np.minimum(1.0, meal / 40)))
    return basal(p) + meal - walk, meal


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Inputs:
    phys: Physio
    t: np.ndarray  # minutes relative to now, uniform 5-min grid from -PAST to +HORIZON
    meals: np.ndarray  # [t_rel, gl, fibre, fat]
    mets: np.ndarray
    obs: np.ndarray  # observed glucose on the past part of the grid (NaN = missing / future)
    dq: float
    stale: float  # minutes since last CGM sample


def build_inputs(state: PatientState) -> Inputs:
    now = state.now
    t = np.arange(-PAST, HORIZON + 1, ts.STEP_MIN, dtype=float)
    h = state.history
    rel = (pd.to_datetime(h["t"], utc=True).to_numpy("datetime64[ns]") - ts._now64(now)) / np.timedelta64(1, "m")
    pos = np.rint((rel + PAST) / ts.STEP_MIN).astype(int)
    ok = (pos >= 0) & (pos < len(t)) & (np.abs(rel - (pos * ts.STEP_MIN - PAST)) <= 2) & (rel <= 0)
    mets = np.full(len(t), 1.3)
    obs = np.full(len(t), np.nan)
    hm = h["mets"].to_numpy(dtype=float)
    hg = h["glucose"].to_numpy(dtype=float)
    miss = h["missing_mask"].to_numpy(dtype=bool)
    good = ok & ~np.isnan(hm)
    mets[pos[good]] = hm[good]
    seen = ok & ~miss
    obs[pos[seen]] = hg[seen]
    for b in state.planned_activity:
        s = (b.start - now).total_seconds() / 60
        mets = np.where((t >= s) & (t < s + b.minutes), np.maximum(mets, b.mets), mets)
    meals = np.array([[(m.t - now).total_seconds() / 60, m.gl, m.fibre_g, m.fat_g] for m in state.meals])
    sleep = ts.prior_sleep_hours(state.history, now)
    pr = state.profile
    phys = Physio(
        fpg=pr.fasting_glucose,
        hba1c=pr.hba1c,
        bmi=pr.bmi,
        sleep_h=sleep if sleep is not None else REF["sleep_hours_prev"],
        t2d=pr.cohort == "t2d",
    )
    dq = state.data_quality if state.data_quality is not None else ts.data_quality(state.history, now)
    return Inputs(phys, t, meals.reshape(-1, 4), mets, obs, dq, ts.stale_minutes(state.history, now))


def _anchor(inp: Inputs, g: np.ndarray) -> tuple[float, float]:
    """(residual at the last fresh sample, residual slope mg/dL/min)."""
    past = inp.t <= 0
    res = inp.obs[past] - g[past]
    tt = inp.t[past]
    ok = ~np.isnan(res)
    if not ok.any():
        return 0.0, 0.0
    last = np.where(ok)[0][-1]
    if inp.t[last] < -60:
        return 0.0, 0.0
    recent = ok & (tt > tt[last] - 10)
    r = float(np.mean(res[recent]))
    win = ok & (tt > -30)
    s = float(np.polyfit(tt[win], res[win], 1)[0]) if win.sum() >= 3 and inp.stale <= ts.FRESH_MIN else 0.0
    s = np.sign(s) * max(0.0, abs(s) - SLOPE_DEADBAND)  # don't extrapolate sensor noise
    return r, float(np.clip(s, -2, 2))


def _q50(inp: Inputs, r: float, s: float) -> np.ndarray:
    g, _ = model_curve(inp.phys, inp.t, inp.meals, inp.mets)
    h = inp.t[inp.t > 0]
    fut = g[inp.t > 0] + r * np.exp(-(h + inp.stale) / ANCHOR_TAU) + s * SLOPE_TAU * (1 - np.exp(-h / SLOPE_TAU))
    return np.clip(fut, G_MIN, G_MAX)


def _peak(inp: Inputs, r: float, s: float) -> float:
    return float(_q50(inp, r, s).max())


def _without_meals(inp: Inputs, col: int) -> Inputs:
    m = inp.meals.copy()
    near = (m[:, 0] >= -180) & (m[:, 0] <= HORIZON)
    m[near, col] = 0.0
    return replace(inp, meals=m)


# --------------------------------------------------------------------------- predict


def _peak_reanchored(inp: Inputs) -> float:
    """Peak exactly as predict() would compute it for these inputs (anchor recomputed)."""
    g, _ = model_curve(inp.phys, inp.t, inp.meals, inp.mets)
    r, s = _anchor(inp, g)
    return _peak(inp, r, s)


def _attributions(inp: Inputs, r: float, s: float) -> dict[str, float]:
    """Model-input features are perturbed through the full pipeline (re-anchored to observed
    glucose, as a what-if is), so attributions match what-if behaviour. The two anchor
    features are perturbed directly."""
    base = _peak(inp, r, s)
    near = (inp.meals[:, 0] >= -180) & (inp.meals[:, 0] <= HORIZON) if len(inp.meals) else np.array([], bool)
    window = inp.t >= -180
    perturbed: dict[str, float] = {
        "meal_gl": _peak_reanchored(_without_meals(inp, 1)) if near.any() else base,
        "meal_fibre": _peak_reanchored(_without_meals(inp, 2)) if near.any() else base,
        "meal_fat": _peak_reanchored(_without_meals(inp, 3)) if near.any() else base,
        "walk_minutes": _peak_reanchored(replace(inp, mets=np.where(window, np.minimum(inp.mets, 1.3), inp.mets))),
        "sleep_hours_prev": _peak_reanchored(replace(inp, phys=replace(inp.phys, sleep_h=REF["sleep_hours_prev"]))),
        "hba1c": _peak_reanchored(replace(inp, phys=replace(inp.phys, hba1c=REF["hba1c"]))),
        "bmi": _peak_reanchored(replace(inp, phys=replace(inp.phys, bmi=REF["bmi"]))),
        "glucose_now": _peak(inp, 0.0, s),
        "glucose_slope_30m": _peak(inp, r, 0.0),
    }
    return {k: round(base - v, 2) for k, v in perturbed.items()}


def _feature_values(state: PatientState, inp: Inputs) -> dict[str, float]:
    near = (inp.meals[:, 0] >= -180) & (inp.meals[:, 0] <= HORIZON) if len(inp.meals) else np.array([], bool)
    m = inp.meals[near] if near.any() else np.zeros((0, 4))
    past_meals = inp.meals[inp.meals[:, 0] <= 0, 0] if len(inp.meals) else np.array([])
    walk = float(((inp.mets >= 2.5) & (inp.t >= -180)).sum() * ts.STEP_MIN)
    out = {
        "meal_gl": float(m[:, 1].sum()),
        "meal_fibre": float(m[:, 2].sum()),
        "meal_fat": float(m[:, 3].sum()),
        "walk_minutes": walk,
        "sleep_hours_prev": inp.phys.sleep_h,
        "hba1c": inp.phys.hba1c,
        "bmi": inp.phys.bmi,
        "mets_1h": float(np.nanmean(inp.mets[(inp.t > -60) & (inp.t <= 0)])),
    }
    if len(past_meals):
        out["minutes_since_meal"] = abs(float(past_meals.max()))
    g_now = ts.glucose_now(state.history, state.now)
    if g_now is not None:
        out["glucose_now"] = g_now
    slope = ts.glucose_slope(state.history, state.now)
    if slope is not None:
        out["glucose_slope_30m"] = round(slope, 2)
    return out


def _feature_baselines(state: PatientState) -> dict[str, float]:
    from src.serving.mock_data import USUAL  # mock-only lookup of the patient's typical values

    out = dict(USUAL.get(state.profile.patient_id, {}))
    g = state.history["glucose"].dropna()
    if len(g):
        out["glucose_now"] = round(float(g.mean()), 1)
    return out


def _attention(inp: Inputs, steps: int = 48) -> list[float]:
    past_t = inp.t[inp.t <= 0][-steps:]
    obs = inp.obs[inp.t <= 0][-steps:]
    d = np.abs(np.diff(np.nan_to_num(obs, nan=np.nanmean(obs) if np.isfinite(obs).any() else 0), prepend=np.nan))
    d = np.nan_to_num(d)
    w = 0.2 + d / (d.max() + 1e-9)
    for tm in inp.meals[:, 0] if len(inp.meals) else []:
        w += np.exp(-np.abs(past_t - tm) / 30)
    w = np.where(np.isnan(obs), 0.0, w)
    return [round(float(x), 4) for x in w / w.sum()] if w.sum() else []


def predict(state: PatientState, scenario: Scenario | None = None, *, explain: bool = True) -> Forecast:
    """``scenario`` is accepted for signature parity; levers are already applied (ADR-005)."""
    inp = build_inputs(state)
    g, _ = model_curve(inp.phys, inp.t, inp.meals, inp.mets)
    r, s = _anchor(inp, g)
    q50 = _q50(inp, r, s)
    h = inp.t[inp.t > 0]
    widen = (1 + 1.5 * (1 - inp.dq)) * (0.45 + 0.3 * min(kp(inp.phys) / K0, 1.8))
    hw = 30 * ((h + min(inp.stale, 120)) / 120) ** 0.66 * widen
    lo, hi = hw, hw * 1.1
    q10, q90 = np.clip(q50 - lo, G_MIN, G_MAX), np.clip(q50 + hi, G_MIN, G_MAX)
    q05, q95 = np.clip(q50 - lo * 1.28, G_MIN, G_MAX), np.clip(q50 + hi * 1.28, G_MIN, G_MAX)
    sd = (lo + hi) / 2 / 1.2816
    spike = float(np.max(norm.cdf((q50 - SPIKE_THRESHOLD) / sd)))
    hypo = float(np.max(norm.cdf((HYPO_THRESHOLD - q50) / sd)))
    conf = float(np.clip((1 - 2 * hw[-1] / 200) * inp.dq, 0, 1))

    traj = [
        {
            "t_min": int(t),
            "q05": round(a, 1),
            "q10": round(b, 1),
            "q50": round(c, 1),
            "q90": round(d, 1),
            "q95": round(e, 1),
        }
        for t, a, b, c, d, e in zip(h, q05, q10, q50, q90, q95)
    ]
    at = {int(t): i for i, t in enumerate(h)}

    def hz(m: int) -> tuple[float, float, float]:
        i = at[m]
        return round(float(q10[i]), 1), round(float(q50[i]), 1), round(float(q90[i]), 1)

    g_now = ts.glucose_now(state.history, state.now)
    slope = ts.glucose_slope(state.history, state.now)
    rule_net = g_now is not None and (
        g_now < HYPO_THRESHOLD or g_now > 300 or (g_now < 90 and slope is not None and slope < -1.5)
    )
    fc: dict[str, Any] = dict(
        glucose_30=hz(30),
        glucose_60=hz(60),
        glucose_120=hz(120),
        spike_prob=round(spike, 3),
        hypo_prob=round(hypo, 3),
        confidence=round(conf, 3),
        trajectory=traj,
        quantiles_90={str(m): (round(float(q05[at[m]]), 1), round(float(q95[at[m]]), 1)) for m in (30, 60, 120)},
        detector_flags={
            "quantile": bool((q90 > SPIKE_THRESHOLD).any() or (q10 < HYPO_THRESHOLD).any()),
            "classifier": bool(spike >= 0.5 or hypo >= 0.3),
            "rule_net": bool(rule_net),
        },
        model_version=MODEL_VERSION,
        feature_lib_version=FEATURE_LIB_VERSION,
        provenance=Provenance.MOCK,
    )
    if explain:
        fc |= dict(
            shap_values=_attributions(inp, r, s),
            attention_weights=_attention(inp),
            feature_values=_feature_values(state, inp),
            feature_baselines=_feature_baselines(state),
        )
    return Forecast(**fc)


# --------------------------------------------------------------------------- mock reports


def _mock_eval_summary() -> dict:
    return {
        "note": "MOCK placeholder values, not results. Replaced by P2's scripts/evaluate.py output.",
        "horizons": {
            "30": {"rmse": [21.4, 18.9, 24.2], "mae": [15.2, 13.4, 17.1]},
            "60": {"rmse": [31.8, 28.1, 35.9], "mae": [23.0, 20.3, 25.8]},
            "120": {"rmse": [41.5, 36.2, 47.3], "mae": [30.9, 27.0, 35.1]},
        },
        "event": {
            "spike": {"recall": 0.81, "pr_auc": 0.62, "f2": 0.71},
            "hypo": {"recall": 0.74, "pr_auc": 0.41, "f2": 0.58},
        },
        "lead_time_min": {"median": 35, "p25": 20},
        "false_alarms_per_patient_day": 1.6,
        "coverage": {"80": 0.77, "90": 0.87},
        "recall_at_budget": 0.76,
        "alert_budget_per_patient_day": 4,
    }


def _mock_ablation() -> pd.DataFrame:
    rows = []
    spec = {  # variant: (rmse_60, ci_halfwidth, recall_spike)
        "glucose_only": (34.6, 3.9, 0.74),
        "ehr_only": (47.2, 5.1, 0.52),
        "wearable_only": (33.9, 3.8, 0.75),
        "concat": (32.7, 3.7, 0.78),
        "film": (31.8, 3.6, 0.80),
        "film_physics": (31.2, 3.5, 0.81),
    }
    for v, (rmse, hwid, rec) in spec.items():
        rows.append((v, "rmse_60", rmse, rmse - hwid, rmse + hwid, 45))
        rows.append((v, "recall_spike", rec, rec - 0.07, min(1.0, rec + 0.07), 45))
    return pd.DataFrame(rows, columns=["variant", "metric", "value", "ci_low", "ci_high", "n_participants"])


def _mock_calibration() -> pd.DataFrame:
    rows = [("coverage", n, e, 4200) for n, e in ((0.5, 0.47), (0.8, 0.77), (0.9, 0.87), (0.95, 0.92))]
    for kind, bias in (("reliability_spike", 0.9), ("reliability_hypo", 0.8)):
        for b in np.arange(0.05, 1.0, 0.1):
            rows.append((kind, round(float(b), 2), round(float(b * bias + 0.02), 3), int(400 * (1 - b) + 20)))
    return pd.DataFrame(rows, columns=["kind", "nominal", "empirical", "n"])


def _mock_subgroups() -> pd.DataFrame:
    rows = []
    for cohort, rmse, rec in (("healthy", 24.1, 0.70), ("prediabetes", 30.5, 0.79), ("t2d", 39.8, 0.84)):
        rows.append((cohort, "rmse_60", rmse, rmse - 5.5, rmse + 5.5, 15))
        rows.append((cohort, "recall_spike", rec, rec - 0.12, min(1.0, rec + 0.12), 15))
    return pd.DataFrame(rows, columns=["cohort", "metric", "value", "ci_low", "ci_high", "n_participants"])


def _mock_clarke() -> pd.DataFrame:
    return pd.DataFrame({"zone": list("ABCDE"), "share": [0.78, 0.19, 0.01, 0.02, 0.0]})


def mock_thresholds() -> dict:
    return {
        "calibrated": False,
        "strata": {
            "low": {"spike": 0.50, "hypo": 0.35},
            "mid": {"spike": 0.45, "hypo": 0.30},
            "high": {"spike": 0.40, "hypo": 0.25},
        },
        "confidence_gate": 0.5,
        "stratum_rule": "low = healthy, mid = prediabetes, high = T2D (placeholder, not chosen on validation)",
    }


@lru_cache(maxsize=1)
def _mock_whatif_sanity() -> pd.DataFrame:
    """Directional tests computed live on the mock cohort at the demo time."""
    from src.serving import mock_data
    from src.serving import scenario as scenario_mod

    tests: dict[str, list[bool]] = {
        "more_carbs_raises_peak": [],
        "walk_lowers_peak": [],
        "short_sleep_raises_peak": [],
        "millet_swap_lowers_peak": [],
    }
    for pid in mock_data.list_patients():
        st = mock_data.get_state(pid, mock_data.DEMO_NOW - timedelta(minutes=0))
        if not st.meals:
            continue
        idx = len(st.meals) - 1
        st = mock_data.get_state(pid, st.meals[idx].t)

        def peak(sc: Scenario, st=st) -> float:
            # The mock cohort's own resolver, whatever MADHUTWIN_DATA says: these are mock_data patients.
            evaluated = scenario_mod.apply(st, sc, mock_data.resolve_meal)
            return max(p.q50 for p in predict(evaluated, sc, explain=False).trajectory)

        base = peak(Scenario())

        tests["more_carbs_raises_peak"].append(peak(Scenario(portion_scale=1.5)) > base)
        tests["walk_lowers_peak"].append(peak(Scenario(walk_minutes=20)) <= base)
        tests["short_sleep_raises_peak"].append(peak(Scenario(prior_sleep_hours=5)) >= base)
        items = st.meals[idx].items or []
        swapped = scenario_mod.rice_to_millet(items) if items else None
        if swapped:
            tests["millet_swap_lowers_peak"].append(
                peak(Scenario(meal_swap={"meal_index": idx, "items": swapped})) < base
            )
    rows = [(k, int(sum(v)), len(v), round(sum(v) / len(v), 3) if v else float("nan")) for k, v in tests.items()]
    return pd.DataFrame(rows, columns=["test", "passed", "n_cases", "pass_rate"])


def mock_report(name: str) -> Any:
    return {
        "eval_summary": _mock_eval_summary,
        "ablation": _mock_ablation,
        "calibration": _mock_calibration,
        "subgroups": _mock_subgroups,
        "clarke_grid": _mock_clarke,
        "thresholds": mock_thresholds,
        "whatif_sanity": _mock_whatif_sanity,
    }[name]()
