"""
Feature Builder
===============
Time-series windowing, sleep/activity feature extraction, and
participant-level cross-validation splits.

Design (from spec Section 7):
  - History window:  ~4 hours at 5-min resolution (48 steps)
  - Horizon:         2 hours (24 steps at 5-min)
  - Features per window:
      • Glucose history (48 values + lag stats)
      • Heart rate (recent mean, variability)
      • Activity / steps (count, intensity proxy)
      • Sleep (duration, recency, quality)
      • Meal features (recent GL, time since meal, cumulative carbs)
      • EHR static vector (age, sex, BMI, HbA1c, lipids, comorbidities,
        genetic risk score)

Interface contract (output → Person 2):
    data/processed/splits.json       {train: [...], val: [...], test: [...]}
    data/processed/windows.pkl       DataFrame/dict with windowed features

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Window parameters ────────────────────────────────────────────────────────
HISTORY_MINUTES = 240          # 4 hours
HORIZON_MINUTES = 120          # 2 hours
SAMPLE_INTERVAL_MINUTES = 5    # 5-min CGM resolution
HISTORY_STEPS = HISTORY_MINUTES // SAMPLE_INTERVAL_MINUTES   # 48
HORIZON_STEPS = HORIZON_MINUTES // SAMPLE_INTERVAL_MINUTES   # 24

# Minimum data requirements for a valid window
MIN_GLUCOSE_COVERAGE = 0.75    # At least 75% of history steps must be non-NaN
MIN_WINDOW_GLUCOSE_POINTS = int(HISTORY_STEPS * MIN_GLUCOSE_COVERAGE)

# ── Spike / hypo thresholds (from spec Section 2) ───────────────────────────
SPIKE_THRESHOLD_MG_DL = 180
HYPO_THRESHOLD_MG_DL  = 70

# ── Split ratios (participant-level) ─────────────────────────────────────────
DEFAULT_SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


# ═════════════════════════════════════════════════════════════════════════════
#  Participant-level splits
# ═════════════════════════════════════════════════════════════════════════════

def create_participant_splits(
    participant_ids: List[str],
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create participant-level train/val/test splits.

    Parameters
    ----------
    participant_ids : List of unique participant IDs.
    ratios          : Dict with keys 'train', 'val', 'test' summing to 1.0.
    seed            : Random seed for reproducibility.

    Returns
    -------
    dict : {'train': [...], 'val': [...], 'test': [...]}
    """
    ratios = ratios or DEFAULT_SPLIT_RATIOS.copy()
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    ids = sorted(participant_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(n * ratios["train"]))
    n_val   = max(1, int(n * ratios["val"]))
    # Test gets the remainder
    n_test  = n - n_train - n_val

    splits = {
        "train": sorted(ids[:n_train]),
        "val":   sorted(ids[n_train:n_train + n_val]),
        "test":  sorted(ids[n_train + n_val:]),
    }

    logger.info(
        "Participant splits — train: %d, val: %d, test: %d (seed=%d)",
        len(splits["train"]), len(splits["val"]), len(splits["test"]), seed
    )
    return splits


def save_splits(
    splits: Dict[str, List[str]],
    output_path: Path | str = "data/processed/splits.json",
) -> Path:
    """Save participant splits to JSON (interface contract with Person 2)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Splits saved to %s", output_path)
    return output_path


def load_splits(
    path: Path | str = "data/processed/splits.json",
) -> Dict[str, List[str]]:
    """Load participant splits from JSON."""
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
#  Feature extraction  –  per-timestamp auxiliary features
# ═════════════════════════════════════════════════════════════════════════════

def extract_glucose_features(
    glucose_history: np.ndarray,
) -> Dict[str, float]:
    """
    Compute summary statistics from a glucose history window.

    Parameters
    ----------
    glucose_history : 1-D array of glucose values (may contain NaN).

    Returns
    -------
    dict with keys:
        glucose_mean, glucose_std, glucose_min, glucose_max,
        glucose_range, glucose_last, glucose_roc_5, glucose_roc_15,
        glucose_cv, glucose_trend (slope of linear fit),
        glucose_nan_fraction
    """
    valid = glucose_history[~np.isnan(glucose_history)]
    n_total = len(glucose_history)
    n_valid = len(valid)

    if n_valid < 3:
        return {k: np.nan for k in [
            "glucose_mean", "glucose_std", "glucose_min", "glucose_max",
            "glucose_range", "glucose_last", "glucose_roc_5", "glucose_roc_15",
            "glucose_cv", "glucose_trend", "glucose_nan_fraction"
        ]}

    mean_val = float(np.nanmean(glucose_history))
    std_val  = float(np.nanstd(glucose_history))

    # Rate of change (mg/dL per 5 min)
    roc_5 = float(glucose_history[-1] - glucose_history[-2]) if n_total >= 2 and not (np.isnan(glucose_history[-1]) or np.isnan(glucose_history[-2])) else np.nan
    roc_15 = float(glucose_history[-1] - glucose_history[-4]) / 3 if n_total >= 4 and not (np.isnan(glucose_history[-1]) or np.isnan(glucose_history[-4])) else np.nan

    # Linear trend (slope via polyfit)
    x = np.arange(n_total)[~np.isnan(glucose_history)]
    y = valid
    trend = float(np.polyfit(x, y, 1)[0]) if n_valid >= 3 else np.nan

    return {
        "glucose_mean":         round(mean_val, 2),
        "glucose_std":          round(std_val, 2),
        "glucose_min":          float(np.nanmin(glucose_history)),
        "glucose_max":          float(np.nanmax(glucose_history)),
        "glucose_range":        float(np.nanmax(glucose_history) - np.nanmin(glucose_history)),
        "glucose_last":         float(glucose_history[-1]) if not np.isnan(glucose_history[-1]) else np.nan,
        "glucose_roc_5":        round(roc_5, 2) if not np.isnan(roc_5) else np.nan,
        "glucose_roc_15":       round(roc_15, 2) if not np.isnan(roc_15) else np.nan,
        "glucose_cv":           round(std_val / mean_val, 4) if mean_val > 0 else np.nan,
        "glucose_trend":        round(trend, 4),
        "glucose_nan_fraction": round(1 - n_valid / n_total, 3),
    }


def extract_hr_features(
    hr_values: np.ndarray,
) -> Dict[str, float]:
    """
    Heart rate features from the history window.

    Returns dict with: hr_mean, hr_std, hr_min, hr_max, hr_resting_proxy
    """
    valid = hr_values[~np.isnan(hr_values)]
    if len(valid) < 2:
        return {k: np.nan for k in ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_resting_proxy"]}

    return {
        "hr_mean":          round(float(np.mean(valid)), 1),
        "hr_std":           round(float(np.std(valid)), 1),
        "hr_min":           float(np.min(valid)),
        "hr_max":           float(np.max(valid)),
        "hr_resting_proxy": round(float(np.percentile(valid, 10)), 1),
    }


def extract_activity_features(
    step_values: np.ndarray,
    interval_minutes: int = SAMPLE_INTERVAL_MINUTES,
) -> Dict[str, float]:
    """
    Activity features from the history window.

    Returns dict with: steps_total, steps_mean_per_interval,
                       activity_intensity_proxy, active_fraction
    """
    valid = step_values[~np.isnan(step_values)]
    if len(valid) == 0:
        return {k: np.nan for k in [
            "steps_total", "steps_mean_per_interval",
            "activity_intensity_proxy", "active_fraction"
        ]}

    total = float(np.sum(valid))
    mean_per_interval = total / len(valid) if len(valid) > 0 else 0

    # Active fraction: intervals with > 10 steps
    active = np.sum(valid > 10)
    active_frac = active / len(valid) if len(valid) > 0 else 0

    # Intensity proxy: steps per minute
    intensity = total / (len(valid) * interval_minutes) if len(valid) > 0 else 0

    return {
        "steps_total":               round(total, 0),
        "steps_mean_per_interval":   round(mean_per_interval, 1),
        "activity_intensity_proxy":  round(intensity, 2),
        "active_fraction":           round(active_frac, 3),
    }


def extract_sleep_features(
    sleep_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> Dict[str, float]:
    """
    Sleep features relative to the current window time.

    Parameters
    ----------
    sleep_df     : Sleep logs for this participant (sleep_start, sleep_end,
                   duration_minutes, efficiency).
    current_time : The current timestamp of the window.

    Returns dict with: sleep_last_duration_min, sleep_hours_ago,
                       sleep_efficiency, slept_last_night (bool→int)
    """
    if sleep_df.empty:
        return {
            "sleep_last_duration_min": np.nan,
            "sleep_hours_ago":         np.nan,
            "sleep_efficiency":        np.nan,
            "slept_last_night":        0,
        }

    # Most recent sleep before current_time
    prior = sleep_df[sleep_df["sleep_end"] <= current_time].copy()
    if prior.empty:
        return {
            "sleep_last_duration_min": np.nan,
            "sleep_hours_ago":         np.nan,
            "sleep_efficiency":        np.nan,
            "slept_last_night":        0,
        }

    last = prior.iloc[-1]
    hours_ago = (current_time - last["sleep_end"]).total_seconds() / 3600

    return {
        "sleep_last_duration_min": float(last.get("duration_minutes", np.nan)),
        "sleep_hours_ago":         round(hours_ago, 1),
        "sleep_efficiency":        float(last.get("efficiency", np.nan)),
        "slept_last_night":        int(hours_ago < 24),
    }


def extract_meal_features(
    meals_df: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback_hours: float = 4.0,
) -> Dict[str, float]:
    """
    Meal features within the lookback window.

    Returns dict with: recent_meal_gl, recent_meal_carbs_g,
                       recent_meal_calories, time_since_last_meal_min,
                       cumulative_carbs_4h, meal_count_4h
    """
    if meals_df.empty:
        return {
            "recent_meal_gl":             np.nan,
            "recent_meal_carbs_g":        np.nan,
            "recent_meal_calories":       np.nan,
            "time_since_last_meal_min":   np.nan,
            "cumulative_carbs_4h":        0.0,
            "meal_count_4h":              0,
        }

    lookback_start = current_time - pd.Timedelta(hours=lookback_hours)
    window_meals = meals_df[
        (meals_df["timestamp"] >= lookback_start) &
        (meals_df["timestamp"] <= current_time)
    ]

    if window_meals.empty:
        # Check for any prior meal
        prior = meals_df[meals_df["timestamp"] <= current_time]
        if prior.empty:
            time_since = np.nan
        else:
            time_since = (current_time - prior.iloc[-1]["timestamp"]).total_seconds() / 60
        return {
            "recent_meal_gl":             np.nan,
            "recent_meal_carbs_g":        np.nan,
            "recent_meal_calories":       np.nan,
            "time_since_last_meal_min":   round(time_since, 1) if not np.isnan(time_since) else np.nan,
            "cumulative_carbs_4h":        0.0,
            "meal_count_4h":              0,
        }

    last_meal = window_meals.iloc[-1]
    time_since = (current_time - last_meal["timestamp"]).total_seconds() / 60

    # Glycemic load — use if available, otherwise estimate from carbs
    if "glycemic_load" in window_meals.columns:
        recent_gl = float(last_meal.get("glycemic_load", np.nan))
    else:
        # Rough GL estimate: assume average GI of 60
        recent_gl = round(0.6 * float(last_meal.get("carbs_g", 0)), 1)

    return {
        "recent_meal_gl":             recent_gl,
        "recent_meal_carbs_g":        float(last_meal.get("carbs_g", np.nan)),
        "recent_meal_calories":       float(last_meal.get("calories_kcal", np.nan)),
        "time_since_last_meal_min":   round(time_since, 1),
        "cumulative_carbs_4h":        round(float(window_meals["carbs_g"].sum()), 1),
        "meal_count_4h":              len(window_meals),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  EHR static feature vector
# ═════════════════════════════════════════════════════════════════════════════

def build_ehr_static_vector(
    ehr_row: pd.Series,
) -> Dict[str, float]:
    """
    Build the static EHR feature vector for one patient.

    Returns dict with: age, sex_male, bmi, hba1c, fasting_glucose,
                       total_cholesterol, ldl, hdl, triglycerides,
                       genetic_risk_score, diabetes_duration_years,
                       comorbidity_hypertension, comorbidity_ckd,
                       comorbidity_cvd, comorbidity_retinopathy,
                       comorbidity_neuropathy
    """
    return {
        "age":                       float(ehr_row.get("age", np.nan)),
        "sex_male":                  1.0 if str(ehr_row.get("sex", "")).upper() == "M" else 0.0,
        "bmi":                       float(ehr_row.get("bmi", np.nan)),
        "hba1c":                     float(ehr_row.get("hba1c", np.nan)),
        "fasting_glucose":           float(ehr_row.get("fasting_glucose", np.nan)),
        "total_cholesterol":         float(ehr_row.get("total_cholesterol", np.nan)),
        "ldl":                       float(ehr_row.get("ldl", np.nan)),
        "hdl":                       float(ehr_row.get("hdl", np.nan)),
        "triglycerides":             float(ehr_row.get("triglycerides", np.nan)),
        "genetic_risk_score":        float(ehr_row.get("genetic_risk_score", np.nan)),
        "diabetes_duration_years":   float(ehr_row.get("diabetes_duration_years", np.nan)),
        "comorbidity_hypertension":  float(ehr_row.get("comorbidity_hypertension", 0)),
        "comorbidity_ckd":           float(ehr_row.get("comorbidity_ckd", 0)),
        "comorbidity_cvd":           float(ehr_row.get("comorbidity_cvd", 0)),
        "comorbidity_retinopathy":   float(ehr_row.get("comorbidity_retinopathy", 0)),
        "comorbidity_neuropathy":    float(ehr_row.get("comorbidity_neuropathy", 0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Windowing engine  –  build full feature windows
# ═════════════════════════════════════════════════════════════════════════════

def build_windows(
    cgm_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
    participant_ids: Optional[List[str]] = None,
    stride_minutes: int = 15,
) -> List[Dict[str, Any]]:
    """
    Build feature windows for all participants.

    Parameters
    ----------
    cgm_df   : Resampled CGM data (participant_id, timestamp, glucose_mg_dl).
    hr_df    : Heart rate data.
    steps_df : Step count data.
    sleep_df : Sleep logs.
    meals_df : Meal logs.
    ehr_df   : EHR records (one row per patient).
    participant_ids : Subset of participants to process (None = all).
    stride_minutes  : Step size between windows (default 15 min).

    Returns
    -------
    List of dicts, each representing one window with:
        - participant_id
        - window_start, window_end (history bounds)
        - target_end (horizon end)
        - glucose_history (array of HISTORY_STEPS values)
        - glucose_targets (array of HORIZON_STEPS values)
        - spike_target (bool: any target > 180 mg/dL)
        - hypo_target  (bool: any target < 70 mg/dL)
        - glucose_target_30/60/120 (scalar targets at specific horizons)
        - All extracted features (glucose, HR, activity, sleep, meal, EHR)
        - is_augmented (bool, always False here — set by augmentation.py)
    """
    if participant_ids is None:
        participant_ids = sorted(cgm_df["participant_id"].unique())

    stride = pd.Timedelta(minutes=stride_minutes)
    history_td = pd.Timedelta(minutes=HISTORY_MINUTES)
    horizon_td = pd.Timedelta(minutes=HORIZON_MINUTES)

    windows: List[Dict[str, Any]] = []
    total_skipped = 0

    for pid in participant_ids:
        # Subset all streams for this participant
        pid_cgm   = cgm_df[cgm_df["participant_id"] == pid].set_index("timestamp").sort_index()
        pid_hr    = hr_df[hr_df["participant_id"] == pid].set_index("timestamp").sort_index() if not hr_df.empty else pd.DataFrame()
        pid_steps = steps_df[steps_df["participant_id"] == pid].set_index("timestamp").sort_index() if not steps_df.empty else pd.DataFrame()
        pid_sleep = sleep_df[sleep_df["participant_id"] == pid] if not sleep_df.empty else pd.DataFrame()
        pid_meals = meals_df[meals_df["participant_id"] == pid] if not meals_df.empty else pd.DataFrame()

        # EHR row
        ehr_match = ehr_df[
            (ehr_df.get("patient_id", ehr_df.get("participant_id", pd.Series())) == pid) |
            (ehr_df.get("participant_id", pd.Series()) == pid)
        ]
        ehr_row = ehr_match.iloc[0] if not ehr_match.empty else pd.Series()

        if pid_cgm.empty or len(pid_cgm) < HISTORY_STEPS + HORIZON_STEPS:
            logger.debug("Skipping participant %s — insufficient CGM data", pid)
            total_skipped += 1
            continue

        # Slide windows
        t_min = pid_cgm.index.min() + history_td
        t_max = pid_cgm.index.max() - horizon_td
        current = t_min

        while current <= t_max:
            w_start = current - history_td
            w_end   = current
            t_end   = current + horizon_td

            # Extract glucose arrays
            hist_mask = (pid_cgm.index >= w_start) & (pid_cgm.index < w_end)
            tgt_mask  = (pid_cgm.index >= w_end) & (pid_cgm.index < t_end)

            glucose_hist = pid_cgm.loc[hist_mask, "glucose_mg_dl"].values
            glucose_tgt  = pid_cgm.loc[tgt_mask, "glucose_mg_dl"].values

            # Pad/truncate to exact step counts
            glucose_hist = _pad_or_truncate(glucose_hist, HISTORY_STEPS)
            glucose_tgt  = _pad_or_truncate(glucose_tgt, HORIZON_STEPS)

            # Skip if too sparse
            if np.sum(~np.isnan(glucose_hist)) < MIN_WINDOW_GLUCOSE_POINTS:
                current += stride
                continue

            # ── Build feature dict ───────────────────────────────────────────
            feat: Dict[str, Any] = {
                "participant_id": pid,
                "window_start":   w_start,
                "window_end":     w_end,
                "target_end":     t_end,
            }

            # Raw arrays (for the temporal encoder)
            feat["glucose_history"] = glucose_hist.tolist()
            feat["glucose_targets"] = glucose_tgt.tolist()

            # Scalar targets at specific horizons (indices 5, 11, 23 → 30, 60, 120 min)
            feat["glucose_target_30"]  = _safe_index(glucose_tgt, 5)
            feat["glucose_target_60"]  = _safe_index(glucose_tgt, 11)
            feat["glucose_target_120"] = _safe_index(glucose_tgt, 23)

            # Event labels
            feat["spike_target"] = bool(np.nanmax(glucose_tgt) > SPIKE_THRESHOLD_MG_DL) if not np.all(np.isnan(glucose_tgt)) else False
            feat["hypo_target"]  = bool(np.nanmin(glucose_tgt) < HYPO_THRESHOLD_MG_DL) if not np.all(np.isnan(glucose_tgt)) else False

            # Extracted features
            feat.update(extract_glucose_features(glucose_hist))

            # HR features
            if not pid_hr.empty:
                hr_window = pid_hr.loc[
                    (pid_hr.index >= w_start) & (pid_hr.index < w_end),
                    "heart_rate_bpm"
                ].values
                feat.update(extract_hr_features(hr_window))
            else:
                feat.update(extract_hr_features(np.array([np.nan])))

            # Activity features
            if not pid_steps.empty:
                steps_window = pid_steps.loc[
                    (pid_steps.index >= w_start) & (pid_steps.index < w_end),
                    "steps"
                ].values
                feat.update(extract_activity_features(steps_window))
            else:
                feat.update(extract_activity_features(np.array([np.nan])))

            # Sleep features
            feat.update(extract_sleep_features(pid_sleep, w_end))

            # Meal features
            feat.update(extract_meal_features(pid_meals, w_end))

            # EHR static vector
            if not ehr_row.empty:
                feat.update(build_ehr_static_vector(ehr_row))

            feat["is_augmented"] = False
            windows.append(feat)

            current += stride

    logger.info(
        "Built %d windows from %d participants (%d skipped for insufficient data)",
        len(windows), len(participant_ids) - total_skipped, total_skipped
    )
    return windows


# ═════════════════════════════════════════════════════════════════════════════
#  Save / load windows
# ═════════════════════════════════════════════════════════════════════════════

def save_windows(
    windows: List[Dict[str, Any]],
    output_path: Path | str = "data/processed/windows.pkl",
) -> Path:
    """Save feature windows to pickle (interface contract with Person 2)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d windows to %s", len(windows), output_path)
    return output_path


def load_windows(
    path: Path | str = "data/processed/windows.pkl",
) -> List[Dict[str, Any]]:
    """Load feature windows from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def windows_to_dataframe(windows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert windows list to a flat DataFrame (excludes array columns).

    The raw glucose_history and glucose_targets arrays are dropped
    since they don't fit neatly in a 2-D frame.  Use the pickle for
    the full data.
    """
    rows = []
    for w in windows:
        row = {k: v for k, v in w.items()
               if not isinstance(v, (list, np.ndarray))}
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with NaN or truncate to target length."""
    if len(arr) >= target_len:
        return arr[:target_len].astype(float)
    padded = np.full(target_len, np.nan)
    padded[:len(arr)] = arr
    return padded


def _safe_index(arr: np.ndarray, idx: int) -> float:
    """Safe array indexing that returns NaN on out-of-bounds."""
    if idx < len(arr):
        return float(arr[idx])
    return np.nan
