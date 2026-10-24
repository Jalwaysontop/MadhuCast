"""Provider adapter: the only module pages import to get data, predictions or reports.

Selection (env vars, read on every call so tests can monkeypatch):
    MADHUTWIN_BACKEND = mock | real     default for both subsystems (default: real)
    MADHUTWIN_DATA    = mock | real     overrides data (profiles, history, food DB, meal engine)
    MADHUTWIN_MODEL   = mock | real     overrides model (predict) and reports/

The Day-10 (P3 data) and Day-14 (P2 model) handoffs land separately, hence the split.
Every payload crossing from a real provider is validated against src.contracts;
failures raise ContractError, which pages render as a visible error card.

Caching here is functools-based so this module stays Streamlit-free and testable;
the dashboard wraps calls with st.cache_data / st.cache_resource.
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from src.contracts import (
    FOOD_DB_COLUMNS,
    REPORT_CSV_COLUMNS,
    REPORT_FILES,
    ContractError,
    Forecast,
    Meal,
    PatientProfile,
    PatientState,
    Provenance,
    Scenario,
)
from src.serving import timeseries as ts

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROFILES_DIR = DATA_DIR / "processed" / "patient_profiles"
TIMESERIES_DIR = DATA_DIR / "processed" / "timeseries"
MEALS_DIR = DATA_DIR / "processed" / "meals"
FOOD_DB_PATH = DATA_DIR / "indian_food_db.csv"
REPORTS_DIR = ROOT / "reports"

Mode = Literal["mock", "real"]


# --------------------------------------------------------------------------- mode selection


def _mode(var: str) -> Mode:
    raw = os.environ.get(var) or os.environ.get("MADHUTWIN_BACKEND") or "real"
    raw = raw.strip().lower()
    if raw not in ("mock", "real"):
        raise ValueError(f"{var} must be 'mock' or 'real', got {raw!r}")
    return raw  # type: ignore[return-value]


def data_mode() -> Mode:
    return _mode("MADHUTWIN_DATA")


def model_mode() -> Mode:
    return _mode("MADHUTWIN_MODEL")


def describe() -> dict[str, str]:
    """For the sidebar footer: which provider backs each subsystem."""
    return {"data": data_mode().upper(), "model": model_mode().upper()}


def _validate(model: type, payload: Any, source: str):
    try:
        return model.model_validate(payload)
    except ValidationError as e:
        raise ContractError(source, str(e)) from e


# --------------------------------------------------------------------------- data (P3)


def list_patients() -> list[str]:
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.list_patients()
    if not PROFILES_DIR.exists():
        raise ContractError("P3:patient_profiles", f"{PROFILES_DIR} not found")
    return sorted(p.stem for p in PROFILES_DIR.glob("P-*.json"))


@lru_cache(maxsize=64)
def _real_profile(pid: str) -> PatientProfile:
    path = PROFILES_DIR / f"{pid}.json"
    if not path.exists():
        raise ContractError("P3:patient_profiles", f"{path.name} not found")
    return _validate(PatientProfile, json.loads(path.read_text()), f"P3:{path.name}")


def _real_state(pid: str, now: datetime | None) -> PatientState:
    profile = _real_profile(pid)
    ts_path = TIMESERIES_DIR / f"{pid}.parquet"
    meals_path = MEALS_DIR / f"{pid}.json"
    if not ts_path.exists():
        raise ContractError("P3:timeseries", f"{ts_path.name} not found")
    hist = pd.read_parquet(ts_path)
    hist["t"] = pd.to_datetime(hist["t"], utc=True)
    meals_raw = json.loads(meals_path.read_text()) if meals_path.exists() else []
    meals = [_validate(Meal, m, f"P3:{meals_path.name}") for m in meals_raw]
    if now is None:
        now = hist["t"].iloc[-1].to_pydatetime()
    hist = hist[hist["t"] <= pd.Timestamp(now)].reset_index(drop=True)
    return _validate(
        PatientState,
        {
            "profile": profile,
            "history": hist,
            "meals": [m for m in meals if m.t <= now],
            "now": now,
            "data_quality": ts.data_quality(hist, now),
            "stream_last_seen": ts.stream_last_seen(hist, now),
        },
        f"P3:state:{pid}",
    )


def data_provenance() -> str:
    """Badge for the active data: MOCK, or REAL only if every patient's wearable stream is REAL
    (P3's demo cohort is SYNTHETIC even when MADHUTWIN_DATA=real)."""
    if data_mode() == "mock":
        return Provenance.MOCK.value
    worn = {_real_profile(p).provenance.get("wearable", Provenance.REAL) for p in list_patients()}
    return Provenance.REAL.value if worn <= {Provenance.REAL} else Provenance.SYNTHETIC.value


def get_state(pid: str, now: datetime | None = None) -> PatientState:
    """Patient state at ``now`` (None = latest available sample)."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.get_state(pid, now)
    return _real_state(pid, now)


def resolve_meal(items: list[dict[str, Any]]) -> Meal:
    """Callers overwrite ``t`` and ``label`` on the returned Meal."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.resolve_meal(items)
    from src.serving import p3_bridge  # P3's engine has no resolve_meal(); the bridge maps it (ADR-016)

    out = p3_bridge.resolve_meal(items, FOOD_DB_PATH)
    return out if isinstance(out, Meal) else _validate(Meal, out, "P3:resolve_meal")


@lru_cache(maxsize=1)
def _real_food_db() -> pd.DataFrame:
    from src.serving import p3_bridge

    df = p3_bridge.food_db(FOOD_DB_PATH)
    missing = [c for c in FOOD_DB_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError("P3:indian_food_db", f"missing columns {missing}")
    return df


def food_db() -> pd.DataFrame:
    """Read-only food table with FOOD_DB_COLUMNS. Callers must not mutate it."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.food_db().copy()
    return _real_food_db().copy()


# --------------------------------------------------------------------------- model (P2)


@lru_cache(maxsize=1)
def _real_predictor():
    try:
        return importlib.import_module("src.serving.predict_api")
    except ImportError as e:
        raise ContractError("P2:predict_api", f"src/serving/predict_api.py not importable: {e}") from e


def _provider_predict(state: PatientState, scenario: Scenario, explain: bool) -> Forecast:
    if model_mode() == "mock":
        from src.serving import mock_predict

        return mock_predict.predict(state, scenario, explain=explain)
    try:
        raw = _real_predictor().predict(state, scenario)
    except ContractError as e:
        if e.source != "P2:input":
            raise
        return _history_fallback(state, explain, e.detail)
    return raw if isinstance(raw, Forecast) else _validate(Forecast, raw, "P2:predict")


def _history_fallback(state: PatientState, explain: bool, why: str) -> Forecast:
    """The real model refused its input (e.g. under six hours of fresh history). That is a data
    condition, not a contract break, so one patient must not take down the alert queue: fall back
    to the mock for this forecast, labelled MOCK, with the classifier marked as not run."""
    from src.serving import mock_predict

    fc = mock_predict.predict(state, explain=False)
    flags = dict(fc.detector_flags or {}) | {"classifier": False}
    return fc.model_copy(
        update={
            "detector_flags": flags,
            "model_version": f"{fc.model_version} (real model skipped: {why.split(';')[0]})",
            "provenance": Provenance.MOCK,
        }
    )


def model_parts() -> dict[str, str]:
    """Provenance per forecast part for the UI badges, e.g. {"spike probability": "MODEL_ESTIMATE",
    "curve, bands, hypo": "MOCK"} for P2's hybrid adapter. A predictor that does not declare
    OUTPUT_PROVENANCE is taken as fully MODEL_ESTIMATE."""
    if model_mode() == "mock":
        return {"forecast": "MOCK"}
    return dict(getattr(_real_predictor(), "OUTPUT_PROVENANCE", None) or {"forecast": "MODEL_ESTIMATE"})


def is_real_spike(fc: Forecast) -> bool:
    """True when this forecast's spike_prob came from the real model (not the mock or a fallback)."""
    return (
        model_mode() == "real"
        and "real model skipped" not in (fc.model_version or "")
        and (model_parts().get("spike probability", model_parts().get("forecast")) == "MODEL_ESTIMATE")
    )


def spike_blind_levers(scenario: Scenario, fc: Forecast) -> list[str]:
    """What-if levers in ``scenario`` that ``fc.spike_prob`` cannot respond to, as UI phrases.

    P2's classifier reads only past history (SPIKE_BLIND_LEVERS in predict_api), so a walk, a later
    meal or a sleep change leaves its probability unchanged, and a later meal even drops out of its
    input. The what-if page must not present that as an effect."""
    if not is_real_spike(fc):
        return []
    blind = set(getattr(_real_predictor(), "SPIKE_BLIND_LEVERS", ()))
    out = []
    if "walk" in blind and scenario.walk_minutes:
        out.append("a post-meal walk")
    if "later_meal" in blind and scenario.meal_time_shift_min > 0:
        out.append("a later meal time")
    if "sleep" in blind and scenario.prior_sleep_hours is not None:
        out.append("last night's sleep")
    return out


def run_scenario(
    state: PatientState, scenario: Scenario | None = None, *, explain: bool = True
) -> tuple[PatientState, Forecast]:
    """Apply what-if levers (P1, ADR-005), then predict. Returns the evaluated state too,
    because an earlier meal-time shift moves the evaluation time back."""
    from src.serving import scenario as scenario_mod

    scenario = scenario or Scenario()
    evaluated = scenario_mod.apply(state, scenario, resolve_meal)
    return evaluated, _provider_predict(evaluated, scenario, explain)


def predict(state: PatientState, scenario: Scenario | None = None, *, explain: bool = True) -> Forecast:
    """``explain=False`` lets the mock skip attributions (alert replay); real models ignore it."""
    return run_scenario(state, scenario, explain=explain)[1]


def load_report(name: str) -> tuple[Any | None, str]:
    """Return (payload, provenance_label).

    payload is a dict (json) or DataFrame (csv), or None when not yet available.
    provenance_label is "MOCK", "MODEL_ESTIMATE" (real eval output) or "MISSING".
    """
    if name not in REPORT_FILES:
        raise KeyError(f"unknown report {name!r}; expected one of {sorted(REPORT_FILES)}")
    if model_mode() == "mock":
        from src.serving import mock_predict

        return mock_predict.mock_report(name), "MOCK"
    path = REPORTS_DIR / REPORT_FILES[name]
    if not path.exists():
        return None, "MISSING"
    if path.suffix == ".json":
        return json.loads(path.read_text()), "MODEL_ESTIMATE"
    df = pd.read_csv(path)
    expected = REPORT_CSV_COLUMNS.get(name, ())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ContractError(f"P2:reports/{path.name}", f"missing columns {missing}")
    return df, "MODEL_ESTIMATE"


def thresholds() -> tuple[dict, bool]:
    """(thresholds, calibrated). Falls back to placeholders flagged UNCALIBRATED."""
    from src.serving.mock_predict import mock_thresholds

    payload, label = load_report("thresholds")
    if payload is None:
        return mock_thresholds(), False
    return payload, bool(payload.get("calibrated", label == "MODEL_ESTIMATE"))


# --------------------------------------------------------------------------- replay window


def day_meals(pid: str, now: datetime | None = None) -> list[Meal]:
    """All meals on the day of ``now`` (including ones after it), for the meal timeline."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.day_meals(pid)
    meals_path = MEALS_DIR / f"{pid}.json"
    if not meals_path.exists():
        return []
    meals = [_validate(Meal, m, f"P3:{meals_path.name}") for m in json.loads(meals_path.read_text())]
    if now is None:
        return meals
    day = now.astimezone(ts.IST).date()
    return [m for m in meals if m.t.astimezone(ts.IST).date() == day]


def time_window() -> tuple[datetime, datetime, datetime]:
    """(replay_start, replay_end, default_now) for the global time slider."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.REPLAY_START, mock_data.DAY_END, mock_data.DEMO_NOW
    ends = []
    for pid in list_patients():
        path = TIMESERIES_DIR / f"{pid}.parquet"
        if path.exists():
            t = pd.to_datetime(pd.read_parquet(path, columns=["t"])["t"], utc=True)
            ends.append(t.max())
    if not ends:
        raise ContractError("P3:timeseries", "no time series found")
    end = min(ends).to_pydatetime()
    start = end - pd.Timedelta(hours=18)
    return start.to_pydatetime() if hasattr(start, "to_pydatetime") else start, end, end


def clear_caches() -> None:
    from src.serving import p3_bridge

    for fn in (_real_profile, _real_food_db, _real_predictor):
        fn.cache_clear()
    p3_bridge.clear_cache()
