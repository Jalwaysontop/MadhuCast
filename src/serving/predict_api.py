"""MadhuTwin P2 adapter for the existing CGMacros XGBoost spike classifier.

IMPORTANT: the provided XGBoost artifact predicts future spike risk only. The
current MadhuTwin Forecast contract also requires glucose trajectories,
uncertainty estimates and hypo risk. Until a model that predicts those values
exists, this adapter keeps those supporting fields from the existing MOCK
predictor and conservatively labels the whole Forecast as MOCK. The classifier
probability itself is inserted into ``spike_prob``. Do not present the returned
Forecast as a fully real/clinically validated digital twin.

Expected artifact: <project-root>/models/xgb_all_cgmacros.json
Expected metadata: <project-root>/models/xgb_all_cgmacros_metadata.json
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import ContractError, Forecast, PatientState, Provenance, Scenario

MODEL_VERSION = "xgb-all-cgmacros-json-spike-0.2.0"
FEATURE_LIB_VERSION = "cgmacros-flat-514-v1"
DEFAULT_PROBABILITY_THRESHOLD = 0.5  # technical default only; not clinically calibrated
SEQUENCE_COLUMNS = ["glucose", "hr", "mets", "carbs", "protein", "fat", "fiber"]
STATIC_COLUMNS = [
    "sex_male", "age", "bmi", "hba1c", "fasting_glu", "fasting_insulin",
    "tg", "chol", "hdl", "ldl",
]
SEQUENCE_STEPS = 72
STEP_MINUTES = 5
EXPECTED_FEATURES = SEQUENCE_STEPS * len(SEQUENCE_COLUMNS) + len(STATIC_COLUMNS)  # 514
# Which Forecast parts this adapter really models (read by backend.model_parts() for the UI badges).
OUTPUT_PROVENANCE = {"spike probability": "MODEL_ESTIMATE", "curve, bands, hypo": "MOCK"}
# What-if levers the classifier cannot see: it reads only the past six hours of glucose/HR/METs and
# logged macros, so a future walk, a later meal or the sleep proxy leave spike_prob unchanged.
SPIKE_BLIND_LEVERS = ("walk", "later_meal", "sleep")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_all_cgmacros.json"
METADATA_PATH = PROJECT_ROOT / "models" / "xgb_all_cgmacros_metadata.json"


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.tz_convert("UTC")


def _lipid_value(lipids: dict[str, Any], *names: str) -> float:
    lowered = {str(k).strip().lower().replace(" ", "_"): v for k, v in (lipids or {}).items()}
    for name in names:
        key = name.lower().replace(" ", "_")
        if key in lowered and lowered[key] is not None:
            try:
                value = float(lowered[key])
                return value if np.isfinite(value) else np.nan
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _static_vector(state: PatientState, metadata: dict[str, Any]) -> np.ndarray:
    profile = state.profile
    values: dict[str, float] = {
        "sex_male": 1.0 if str(profile.sex).upper() == "M" else 0.0,
        "age": float(profile.age),
        "bmi": float(profile.bmi),
        "hba1c": float(profile.hba1c),
        "fasting_glu": float(profile.fasting_glucose),
        # PatientState/PatientProfile does not provide fasting insulin. Keep NaN;
        # XGBoost supports missing numeric features, subject to the trained model.
        "fasting_insulin": np.nan,
        "tg": _lipid_value(profile.lipids or {}, "tg", "triglycerides", "triglyceride"),
        "chol": _lipid_value(profile.lipids or {}, "chol", "cholesterol", "total_cholesterol", "total"),
        "hdl": _lipid_value(profile.lipids or {}, "hdl"),
        "ldl": _lipid_value(profile.lipids or {}, "ldl"),
    }
    columns = list(metadata.get("static_columns", []))
    if columns != STATIC_COLUMNS:
        raise ContractError(
            "P2:model_metadata",
            f"static_columns do not match the adapter's verified order: {columns!r}",
        )
    return np.asarray([values[c] for c in columns], dtype=np.float32)


def _sequence_matrix(state: PatientState, metadata: dict[str, Any]) -> np.ndarray:
    """Build the model's 72 x 7 (six-hour, 5-minute) sequence from PatientState.

    The training dataset builder used the exact channel order in SEQUENCE_COLUMNS;
    meal macros are summed per right-closed 5-minute bin. Only past meals are used.
    Glucose/HR/MET gaps are forward-filled for at most 30 minutes, matching the
    project window-builder rule. Longer gaps fail visibly instead of being silently
    invented.
    """
    seq_columns = list(metadata.get("sequence_columns", []))
    if seq_columns != SEQUENCE_COLUMNS:
        raise ContractError(
            "P2:model_metadata",
            f"sequence_columns do not match the adapter's verified order: {seq_columns!r}",
        )

    now = _as_utc(state.now)
    end = now.floor(f"{STEP_MINUTES}min")
    grid = pd.date_range(end=end, periods=SEQUENCE_STEPS, freq=f"{STEP_MINUTES}min", tz="UTC")

    history = state.history.copy()
    history["t"] = pd.to_datetime(history["t"], utc=True, errors="coerce")
    history = history.loc[history["t"].notna() & (history["t"] <= now)].copy()
    if history.empty:
        raise ContractError("P2:input", "no historical samples exist at or before state.now")
    history = history.set_index("t").sort_index()
    observed = history[["glucose", "hr", "mets"]].apply(pd.to_numeric, errors="coerce")
    observed = observed.resample(f"{STEP_MINUTES}min", label="right", closed="right").mean()
    observed = observed.reindex(grid)

    # The training builder forward-filled these sensor channels only, with a
    # maximum 30-minute carry. It did not backfill or interpolate from future data.
    observed = observed.ffill(limit=30 // STEP_MINUTES)

    meal_rows: list[dict[str, Any]] = []
    for meal in state.meals:
        meal_time = _as_utc(meal.t)
        if meal_time <= now:
            meal_rows.append({
                "t": meal_time,
                "carbs": float(meal.carbs_g),
                "protein": float(meal.protein_g),
                "fat": float(meal.fat_g),
                "fiber": float(meal.fibre_g),
            })
    macro_columns = ["carbs", "protein", "fat", "fiber"]
    if meal_rows:
        macros = pd.DataFrame(meal_rows).set_index("t").sort_index()
        macros = macros[macro_columns].resample(
            f"{STEP_MINUTES}min", label="right", closed="right"
        ).sum(min_count=1)
        macros = macros.reindex(grid).fillna(0.0)
    else:
        macros = pd.DataFrame(0.0, index=grid, columns=macro_columns)

    sequence = pd.concat([observed, macros], axis=1)[SEQUENCE_COLUMNS]
    matrix = sequence.to_numpy(dtype=np.float32)
    if matrix.shape != (SEQUENCE_STEPS, len(SEQUENCE_COLUMNS)):
        raise ContractError("P2:input", f"unexpected sequence shape {matrix.shape}; expected (72, 7)")
    if not np.isfinite(matrix).all():
        missing = [
            f"{SEQUENCE_COLUMNS[j]} ({int(np.isnan(matrix[:, j]).sum())} missing)"
            for j in range(matrix.shape[1]) if not np.isfinite(matrix[:, j]).all()
        ]
        raise ContractError(
            "P2:input",
            "not enough fresh history to construct the model's six-hour input; unresolved channels: "
            + ", ".join(missing),
        )
    return matrix


def build_feature_matrix(state: PatientState, metadata: dict[str, Any]) -> np.ndarray:
    """Return one raw feature row in the model's documented 514-feature order."""
    if list(metadata.get("sequence_columns", [])) != SEQUENCE_COLUMNS:
        raise ContractError("P2:model_metadata", "unexpected sequence_columns order")
    if list(metadata.get("static_columns", [])) != STATIC_COLUMNS:
        raise ContractError("P2:model_metadata", "unexpected static_columns order")
    sequence = _sequence_matrix(state, metadata)
    static = _static_vector(state, metadata)
    row = np.concatenate([sequence.reshape(-1), static]).astype(np.float32, copy=False)
    if row.size != EXPECTED_FEATURES:
        raise ContractError("P2:input", f"model needs {EXPECTED_FEATURES} features, got {row.size}")
    return row.reshape(1, -1)


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise ContractError(
            "P2:model",
            f"model artifact not found at {MODEL_PATH}. Copy xgb_all_cgmacros.json into the project's models/ folder.",
        )
    if not METADATA_PATH.exists():
        raise ContractError("P2:model_metadata", f"metadata file not found at {METADATA_PATH}")
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError("P2:model_metadata", f"could not read metadata JSON: {exc}") from exc

    if int(metadata.get("n_features", -1)) != EXPECTED_FEATURES:
        raise ContractError(
            "P2:model_metadata",
            f"metadata says n_features={metadata.get('n_features')}; expected {EXPECTED_FEATURES}",
        )
    if float(metadata.get("threshold", -1)) != 180.0:
        raise ContractError(
            "P2:model_metadata",
            f"metadata threshold is {metadata.get('threshold')!r}; this adapter expects the 180 mg/dL spike target",
        )
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    except Exception as exc:
        raise ContractError(
            "P2:model",
            f"could not load native XGBoost JSON {MODEL_PATH.name}: {exc}. Verify the file and use the XGBoost version pinned in requirements-model.txt.",
        ) from exc

    model_features = getattr(model, "n_features_in_", None)
    if model_features is None and hasattr(model, "get_booster"):
        try:
            model_features = model.get_booster().num_features()
        except Exception:
            model_features = None
    if model_features is not None and int(model_features) != EXPECTED_FEATURES:
        raise ContractError(
            "P2:model",
            f"loaded model expects {model_features} features but metadata/adapter provide {EXPECTED_FEATURES}",
        )
    if not hasattr(model, "predict_proba"):
        raise ContractError("P2:model", "loaded artifact has no predict_proba method; expected a binary classifier")
    return model, metadata


def _spike_probability(model: Any, features: np.ndarray) -> float:
    try:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive = classes.index(1)
        probability = float(probabilities[0, positive])
    except Exception as exc:
        raise ContractError("P2:model_prediction", f"XGBoost probability prediction failed: {exc}") from exc
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ContractError("P2:model_prediction", f"invalid spike probability {probability!r}")
    return probability


def predict(
    state: PatientState,
    scenario: Scenario | None = None,
    *,
    explain: bool = True,
) -> Forecast:
    """Predict spike risk with XGBoost; keep unsupported outputs explicitly MOCK.

    ``backend.run_scenario`` applies what-if edits before calling this function.
    Do not apply ``scenario`` again here. The scenario parameter remains for API
    signature compatibility with the existing backend.
    """
    del scenario, explain  # already applied upstream; XGBoost explanations are not bundled
    model, metadata = _load_artifacts()
    features = build_feature_matrix(state, metadata)
    probability = _spike_probability(model, features)

    # Preserve the current dashboard contract while being transparent that the
    # classifier does not produce glucose trajectories, quantiles, hypo risk or SHAP.
    # Those secondary values remain from the existing MOCK implementation.
    # Import lazily so feature-preparation tests do not need the full mock stack.
    from src.serving import mock_predict

    support = mock_predict.predict(state, explain=False)
    payload = support.model_dump()
    payload["spike_prob"] = round(probability, 3)
    detector_flags = dict(payload.get("detector_flags") or {})
    # Technical default only. It is not a validation-derived/clinical threshold.
    threshold = float(os.environ.get("MADHUTWIN_XGB_SPIKE_THRESHOLD", DEFAULT_PROBABILITY_THRESHOLD))
    if not 0.0 <= threshold <= 1.0:
        raise ContractError("P2:config", "MADHUTWIN_XGB_SPIKE_THRESHOLD must be between 0 and 1")
    detector_flags["classifier"] = probability >= threshold
    payload["detector_flags"] = detector_flags
    payload["shap_values"] = {}
    payload["attention_weights"] = []
    payload["feature_values"] = None
    payload["feature_baselines"] = None
    payload["model_version"] = f"{MODEL_VERSION}+mock-support-{support.model_version}"
    payload["feature_lib_version"] = FEATURE_LIB_VERSION
    # The Forecast has only one provenance field. MOCK is the conservative label
    # because most fields are still mock-generated; model_version distinguishes the mix.
    payload["provenance"] = Provenance.MOCK
    try:
        return Forecast.model_validate(payload)
    except Exception as exc:
        raise ContractError("P2:forecast_contract", f"adapter output did not satisfy Forecast: {exc}") from exc
