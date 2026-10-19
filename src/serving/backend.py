"""Provider adapter: the only module pages import to get data, predictions or reports.

Selection (env vars, read on every call so tests can monkeypatch):
    MADHUTWIN_BACKEND = mock | real     default for both subsystems (default: mock)
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
    Scenario,
)

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
    raw = os.environ.get(var) or os.environ.get("MADHUTWIN_BACKEND") or "mock"
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
    hist = hist[hist["t"] <= now]
    window = hist[hist["t"] > now - pd.Timedelta(hours=24)]
    expected = max(len(window), 1)
    dq = float((~window["missing_mask"].astype(bool)).sum()) / expected
    last_seen = {
        "cgm": window.loc[~window["missing_mask"].astype(bool), "t"].max(),
        "wearable": window.loc[window["hr"].notna(), "t"].max(),
    }
    return _validate(
        PatientState,
        {
            "profile": profile,
            "history": hist.reset_index(drop=True),
            "meals": [m for m in meals if m.t <= now],
            "now": now,
            "data_quality": dq,
            "stream_last_seen": {k: v.to_pydatetime() for k, v in last_seen.items() if pd.notna(v)},
        },
        f"P3:state:{pid}",
    )


def get_state(pid: str, now: datetime | None = None) -> PatientState:
    """Patient state at ``now`` (None = latest available sample)."""
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.get_state(pid, now)
    return _real_state(pid, now)


def resolve_meal(items: list[dict[str, Any]]) -> Meal:
    if data_mode() == "mock":
        from src.serving import mock_data

        return mock_data.resolve_meal(items)
    try:
        engine = importlib.import_module("src.data.meal_engine")
    except ImportError as e:
        raise ContractError("P3:meal_engine", f"src/data/meal_engine.py not importable: {e}") from e
    out = engine.resolve_meal(items)
    return out if isinstance(out, Meal) else _validate(Meal, out, "P3:resolve_meal")


@lru_cache(maxsize=1)
def _real_food_db() -> pd.DataFrame:
    if not FOOD_DB_PATH.exists():
        raise ContractError("P3:indian_food_db", f"{FOOD_DB_PATH} not found")
    df = pd.read_csv(FOOD_DB_PATH)
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


def predict(state: PatientState, scenario: Scenario | None = None) -> Forecast:
    scenario = scenario or Scenario()
    if model_mode() == "mock":
        from src.serving import mock_predict

        return mock_predict.predict(state, scenario)
    raw = _real_predictor().predict(state, scenario)
    return raw if isinstance(raw, Forecast) else _validate(Forecast, raw, "P2:predict")


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


def clear_caches() -> None:
    for fn in (_real_profile, _real_food_db, _real_predictor):
        fn.cache_clear()
