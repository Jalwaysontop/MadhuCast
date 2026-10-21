"""Attributions → plain-language sentences. Deterministic templates only, never an LLM:
generated free text can hallucinate clinical claims (see README, "Why templates").

Attributions explain the model, not the biology. Every sentence describes how a feature
moves the model's *predicted 2-hour peak*, in mg/dL, rounded to clinically sensible
precision (nearest 5 mg/dL above 10, else nearest 1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.contracts import Forecast, PatientState, Scenario
from src.serving import timeseries as ts


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    unit: str
    group: str
    decimals: int = 0
    reference: float | None = None  # neutral value used for direction reasoning
    lever: str | None = None  # what-if lever that increases this feature


GROUPS = ("Meal", "Activity", "Sleep (proxy)", "EHR", "Recent glucose", "Other")

FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "meal_gl": FeatureSpec("Recent meal glycaemic load", "", "Meal", 0, 0.0, "portion"),
    "meal_carbs": FeatureSpec("Recent meal carbohydrate", "g", "Meal", 0, 0.0, "portion"),
    "meal_fibre": FeatureSpec("Recent meal fibre", "g", "Meal", 0, 0.0),
    "meal_fat": FeatureSpec("Recent meal fat", "g", "Meal", 0, 0.0),
    "minutes_since_meal": FeatureSpec("Time since last meal", "min", "Meal", 0),
    "walk_minutes": FeatureSpec("Walking (last 3 h and planned)", "min", "Activity", 0, 0.0, "walk"),
    "mets_1h": FeatureSpec("Activity level, last hour", "METs", "Activity", 1),
    "hr_mean_1h": FeatureSpec("Heart rate, last hour", "bpm", "Activity", 0),
    "sleep_hours_prev": FeatureSpec("Sleep last night (proxy)", "h", "Sleep (proxy)", 1, 7.5, "sleep"),
    "hba1c": FeatureSpec("HbA1c", "%", "EHR", 1, 5.4),
    "bmi": FeatureSpec("BMI", "kg/m²", "EHR", 1, 23.0),
    "age": FeatureSpec("Age", "years", "EHR", 0),
    "glucose_now": FeatureSpec("Current glucose", "mg/dL", "Recent glucose", 0),
    "glucose_slope_30m": FeatureSpec("Glucose trend, last 30 min", "mg/dL/min", "Recent glucose", 1, 0.0),
}

FALLBACK = (
    "Feature attributions are not available for this forecast, so no explanation is shown. "
    "The forecast and its uncertainty band are still valid model estimates."
)
CAPTION = "Explains the model, not the biology."
ATTENTION_CAPTION = "Attention weights are a directional indicator, not a rigorous attribution."
MIN_EFFECT = 1.0  # mg/dL; smaller attributions are reported as "little effect"


def spec(key: str) -> FeatureSpec:
    return FEATURE_REGISTRY.get(key, FeatureSpec(key, "", "Other"))


# --------------------------------------------------------------------------- formatting


def round_mgdl(x: float) -> int:
    x = abs(x)
    return int(5 * round(x / 5)) if x >= 10 else int(round(x))


def fmt_mgdl(x: float) -> str:
    return f"about {round_mgdl(x)} mg/dL"


def plural(n: float, one: str, many: str | None = None) -> str:
    many = many or one + "s"
    return f"{n:g} {one if n == 1 else many}"


def fmt_value(key: str, v: float) -> str:
    s = spec(key)
    if s.unit == "min":
        return plural(int(round(v)), "minute")
    if s.unit == "h":
        return f"{v:.{s.decimals}f} h"
    if s.unit == "%":
        return f"{v:.{s.decimals}f}%"
    num = f"{v:.{s.decimals}f}"
    return f"{num} {s.unit}".strip()


def fmt_pct(p: float) -> str:
    if p < 0.01:
        return "<1%"
    if p > 0.99:
        return ">99%"
    return f"{p:.0%}"


# --------------------------------------------------------------------------- sentences


def risk_sentence(fc: Forecast) -> str:
    return (
        f"Estimated chance of glucose above 180 mg/dL within 2 hours: {fmt_pct(fc.spike_prob)}; "
        f"below 70 mg/dL: {fmt_pct(fc.hypo_prob)} (model estimate, confidence {fmt_pct(fc.confidence)})."
    )


def contributor_sentence(key: str, shap: float, value: float | None = None, baseline: float | None = None) -> str:
    s = spec(key)
    if abs(shap) < MIN_EFFECT:
        subject = f"{s.label} ({fmt_value(key, value)})" if value is not None else s.label
        return f"{subject} has little effect on the predicted peak."
    verb = "raising" if shap > 0 else "lowering"
    effect = f"{verb} the predicted peak by {fmt_mgdl(shap)}"
    if value is not None and baseline is not None and round(value, s.decimals) != round(baseline, s.decimals):
        return f"{s.label} was {fmt_value(key, value)} vs the usual {fmt_value(key, baseline)}, {effect}."
    if value is not None:
        return f"{s.label} ({fmt_value(key, value)}) is {effect}."
    return f"{s.label} is {effect}."


def top_contributors(fc: Forecast, k: int = 3) -> list[tuple[str, float]]:
    items = [(key, v) for key, v in fc.shap_values.items() if abs(v) >= MIN_EFFECT]
    return sorted(items, key=lambda kv: (-abs(kv[1]), kv[0]))[:k]


def explain(fc: Forecast, k: int = 3) -> list[str]:
    if not fc.shap_values:
        return [FALLBACK]
    top = top_contributors(fc, k)
    if not top:
        return ["No single feature moves the predicted peak by more than 1 mg/dL."]
    vals, base = fc.feature_values or {}, fc.feature_baselines or {}
    return [contributor_sentence(key, v, vals.get(key), base.get(key)) for key, v in top]


def counterfactual_sentence(
    alternative: str, base_peak_band: tuple[float, float], alt_peak_band: tuple[float, float], d_peak: float
) -> str:
    """d_peak = scenario − baseline. The range is a conservative band difference (80% bands
    at each peak, treated as independent), so it is wide on purpose."""
    diff = -d_peak  # how much higher the current meal is than the alternative
    lo, hi = base_peak_band[0] - alt_peak_band[1], base_peak_band[1] - alt_peak_band[0]
    if round_mgdl(diff) == 0:
        return f"Compared with {alternative}, the predicted peak is about the same."
    verb = "raise" if diff > 0 else "lower"
    return (
        f"Compared with {alternative}, the current plan is predicted to {verb} peak glucose by "
        f"~{round_mgdl(diff)} mg/dL (80% range {lo:+.0f} to {hi:+.0f} mg/dL)."
    )


# --------------------------------------------------------------------------- reliability check


@dataclass(frozen=True)
class Agreement:
    key: str
    label: str
    lever: str
    expected_sign: int
    observed_delta: float
    agree: bool


def _lever_scenario(key: str, value: float | None, meal_index: int | None) -> tuple[Scenario, str] | None:
    lever = spec(key).lever
    if lever == "portion":
        return Scenario(portion_scale=1.25, target_meal_index=meal_index), "portion +25%"
    if lever == "walk":
        return Scenario(walk_minutes=20, target_meal_index=meal_index), "+20 min walk"
    if lever == "sleep":
        hours = min(10.0, (value or 7.0) + 1.5)
        return Scenario(prior_sleep_hours=hours, target_meal_index=meal_index), f"sleep {hours:.1f} h"
    return None


def agreement_check(
    state: PatientState, fc: Forecast, run: Callable[[PatientState, Scenario], Forecast], k: int = 3
) -> list[Agreement]:
    """For the top-k attributed *levers*, run the matching what-if (which increases the
    feature) and check the sign of the peak change against the attribution's implied
    direction: sign(shap) × sign(value − reference)."""
    vals = fc.feature_values or {}
    base_peak = float(ts.forecast_curve(fc)["q50"].max())
    out: list[Agreement] = []
    for key, v in sorted(fc.shap_values.items(), key=lambda kv: -abs(kv[1])):
        s = spec(key)
        if s.lever is None or abs(v) < MIN_EFFECT or key not in vals or s.reference is None:
            continue
        direction = (vals[key] > s.reference) - (vals[key] < s.reference)
        if direction == 0:
            continue
        built = _lever_scenario(key, vals.get(key), None)
        if built is None:
            continue
        sc, desc = built
        delta = float(ts.forecast_curve(run(state, sc))["q50"].max()) - base_peak
        expected = (1 if v > 0 else -1) * direction
        observed = (delta > 0.5) - (delta < -0.5)
        out.append(Agreement(key, s.label, desc, expected, round(delta, 1), observed == expected))
        if len(out) == k:
            break
    return out
