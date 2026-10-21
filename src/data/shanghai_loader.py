"""
Shanghai T2D Dataset Loader
============================
Loads and windows the Shanghai dataset (98 T2D patients, 15-min CGM,
meal/insulin/drug event flags, rich EHR static features).

Expected files in data/raw/shanghai/:
  cgm_data.csv    — columns: patient_id, timestamp, glucose_value
                    (or per-patient files matching *cgm*.csv / *glucose*.csv)
  events.csv      — columns: patient_id, timestamp, meal_event, insulin_event, drug_event
                    (binary 0/1 flags; one row per 15-min interval OR sparse event rows)
  static.csv      — columns: patient_id, age, sex, bmi, hba1c, fasting_glucose,
                    total_cholesterol, ldl, hdl, triglycerides, creatinine, egfr,
                    c_peptide, hypertension, macrovascular, microvascular,
                    dm_years, on_insulin, on_metformin

Output: data/processed/shanghai_windows.npz
  X_seq    : (N, 24, 4)  float32  [glucose_norm, meal_event, insulin_event, drug_event]
  X_static : (N, 15)     float32  one row per window (same static for all windows of patient)
  y        : (N,)        int8     spike label  (glucose > 180 mg/dL within next 2 h)
  group    : (N,)        object   patient_id   (for grouped cross-validation)
  g_last   : (N,)        float32  glucose mg/dL at window end
  channels : (4,)        str      channel names for X_seq

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
STEP_MINUTES   = 15          # Shanghai CGM cadence
HISTORY_STEPS  = 24          # 6 h at 15-min steps
HORIZON_STEPS  = 8           # 2 h at 15-min steps
STRIDE_STEPS   = 4           # 1-hour stride between windows
SPIKE_THRESH   = 180.0       # mg/dL

SEQ_CHANNELS   = ["glucose_norm", "meal_event", "insulin_event", "drug_event"]
STATIC_COLS    = [
    "age", "sex_male", "bmi", "hba1c", "fasting_glucose",
    "total_cholesterol", "ldl", "hdl", "triglycerides",
    "creatinine", "egfr", "c_peptide",
    "hypertension", "macrovascular", "microvascular",
    "dm_years",  # diabetes duration in years
    "on_insulin", "on_metformin",
]

# Normalisation reference (approx population mean/std for diabetic patients)
GLUCOSE_MEAN = 160.0
GLUCOSE_STD  = 50.0


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_all(data_dir: Path | str) -> Dict[str, pd.DataFrame]:
    """
    Load Shanghai dataset from *data_dir*.

    Returns dict with keys: ``cgm``, ``events``, ``static``.
    Falls back to demo data if real files are missing.
    """
    data_dir = Path(data_dir)
    cgm_df    = _load_cgm(data_dir)
    events_df = _load_events(data_dir)
    static_df = _load_static(data_dir)

    is_real = not cgm_df.empty and "is_demo" not in cgm_df.columns
    if not is_real:
        logger.warning("Shanghai: no real data found — using demo data")
        cgm_df, events_df, static_df = _create_demo_shanghai(n_patients=98, seed=42)
    else:
        logger.info("Shanghai: loaded real data — %d CGM rows, %d patients",
                    len(cgm_df), cgm_df["patient_id"].nunique())

    return {"cgm": cgm_df, "events": events_df, "static": static_df}


def build_npz(
    data: Dict[str, pd.DataFrame],
    output_path: Path | str = "data/processed/shanghai_windows.npz",
) -> Path:
    """
    Build windowed arrays and save as .npz.

    Returns path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cgm_df    = data["cgm"]
    events_df = data["events"]
    static_df = data["static"]

    static_lookup = _build_static_lookup(static_df)

    X_seq_list: List[np.ndarray] = []
    X_static_list: List[np.ndarray] = []
    y_list: List[int] = []
    group_list: List[str] = []
    g_last_list: List[float] = []

    patient_ids = sorted(cgm_df["patient_id"].unique())
    total_windows = 0

    for pid in patient_ids:
        pid_cgm = (
            cgm_df[cgm_df["patient_id"] == pid]
            .set_index("timestamp")
            .sort_index()
        )
        pid_events = (
            events_df[events_df["patient_id"] == pid]
            .set_index("timestamp")
            .sort_index()
            if not events_df.empty
            else pd.DataFrame()
        )

        static_vec = static_lookup.get(str(pid), np.full(len(STATIC_COLS), np.nan))

        if len(pid_cgm) < HISTORY_STEPS + HORIZON_STEPS:
            logger.debug("Shanghai: skipping %s (too few CGM rows)", pid)
            continue

        t_min = pid_cgm.index.min() + pd.Timedelta(minutes=STEP_MINUTES * HISTORY_STEPS)
        t_max = pid_cgm.index.max() - pd.Timedelta(minutes=STEP_MINUTES * HORIZON_STEPS)
        current = t_min
        stride  = pd.Timedelta(minutes=STEP_MINUTES * STRIDE_STEPS)
        hist_td = pd.Timedelta(minutes=STEP_MINUTES * HISTORY_STEPS)
        hor_td  = pd.Timedelta(minutes=STEP_MINUTES * HORIZON_STEPS)

        while current <= t_max:
            w_start = current - hist_td
            w_end   = current
            t_end   = current + hor_td

            hist_mask = (pid_cgm.index >= w_start) & (pid_cgm.index < w_end)
            tgt_mask  = (pid_cgm.index >= w_end)   & (pid_cgm.index < t_end)

            glc_hist = pid_cgm.loc[hist_mask, "glucose_mg_dl"].values
            glc_tgt  = pid_cgm.loc[tgt_mask,  "glucose_mg_dl"].values

            glc_hist = _pad_or_truncate(glc_hist, HISTORY_STEPS)
            glc_tgt  = _pad_or_truncate(glc_tgt,  HORIZON_STEPS)

            if np.sum(~np.isnan(glc_hist)) < HISTORY_STEPS // 2:
                current += stride
                continue

            # Event channels (15-min aligned)
            meal_hist, ins_hist, drug_hist = _extract_event_channels(
                pid_events, w_start, w_end, HISTORY_STEPS, STEP_MINUTES
            )

            # Build sequence array (T, 4)
            glc_norm = (glc_hist - GLUCOSE_MEAN) / GLUCOSE_STD
            seq = np.stack([glc_norm, meal_hist, ins_hist, drug_hist], axis=1)  # (24, 4)

            # Targets
            valid_tgt = glc_tgt[~np.isnan(glc_tgt)]
            spike = int(len(valid_tgt) > 0 and np.max(valid_tgt) > SPIKE_THRESH)
            g_last = float(glc_hist[~np.isnan(glc_hist)][-1]) if np.any(~np.isnan(glc_hist)) else np.nan

            X_seq_list.append(seq.astype(np.float32))
            X_static_list.append(static_vec.astype(np.float32))
            y_list.append(spike)
            group_list.append(str(pid))
            g_last_list.append(g_last)
            total_windows += 1

            current += stride

    if total_windows == 0:
        logger.error("Shanghai: no windows built — check data")
        return output_path

    np.savez_compressed(
        output_path,
        X_seq    = np.stack(X_seq_list,    axis=0),
        X_static = np.stack(X_static_list, axis=0),
        y        = np.array(y_list,     dtype=np.int8),
        group    = np.array(group_list, dtype=object),
        g_last   = np.array(g_last_list, dtype=np.float32),
        channels = np.array(SEQ_CHANNELS),
        static_cols = np.array(STATIC_COLS),
    )
    logger.info(
        "Shanghai: saved %d windows (%d patients) → %s",
        total_windows, len(patient_ids), output_path
    )
    positive_rate = np.mean(y_list)
    logger.info("Shanghai: positive rate (spike) = %.1f%%", 100 * positive_rate)
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  File loaders (real data)
# ═════════════════════════════════════════════════════════════════════════════

def _load_cgm(data_dir: Path) -> pd.DataFrame:
    """
    Try to load CGM data from common Shanghai file patterns.

    Supported formats:
      - cgm_data.csv with columns: patient_id, timestamp, glucose_value
      - per-patient files: *cgm*.csv or *glucose*.csv
      - combined with mmol/L — auto-converted to mg/dL
    """
    # Pattern 1: single combined file
    for pattern in ["cgm_data.csv", "CGM_data.csv", "cgm.csv", "glucose_data.csv"]:
        p = data_dir / pattern
        if p.exists():
            df = pd.read_csv(p)
            return _normalise_cgm_df(df)

    # Pattern 2: per-patient files in subdirectory
    cgm_files = list(data_dir.glob("**/cgm*.csv")) + list(data_dir.glob("**/glucose*.csv"))
    if cgm_files:
        parts = []
        for f in sorted(cgm_files):
            df = pd.read_csv(f)
            # infer patient_id from filename if missing
            if "patient_id" not in df.columns:
                pid = re.sub(r"[^0-9]", "", f.stem) or f.stem
                df["patient_id"] = pid
            parts.append(df)
        combined = pd.concat(parts, ignore_index=True)
        return _normalise_cgm_df(combined)

    return pd.DataFrame()


def _normalise_cgm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and convert units."""
    # Flexible column mapping
    col_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "_").replace("-", "_")
        if any(x in lc for x in ["patient", "subject", "id"]) and "patient_id" not in col_map.values():
            col_map[c] = "patient_id"
        elif any(x in lc for x in ["time", "date", "stamp"]) and "timestamp" not in col_map.values():
            col_map[c] = "timestamp"
        elif any(x in lc for x in ["glucose", "cgm", "cbg", "bg"]) and "glucose_raw" not in col_map.values():
            col_map[c] = "glucose_raw"

    df = df.rename(columns=col_map)
    if "patient_id" not in df.columns or "timestamp" not in df.columns or "glucose_raw" not in df.columns:
        logger.warning("Shanghai CGM: could not map required columns. Found: %s", list(df.columns))
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "glucose_raw"])

    # Convert mmol/L → mg/dL if values look like mmol/L (< 30)
    df["glucose_mg_dl"] = pd.to_numeric(df["glucose_raw"], errors="coerce")
    median_g = df["glucose_mg_dl"].median()
    if median_g < 30:
        df["glucose_mg_dl"] = (df["glucose_mg_dl"] * 18.016).round(1)
        logger.info("Shanghai CGM: converted mmol/L → mg/dL (median was %.1f)", median_g)

    df["patient_id"] = df["patient_id"].astype(str)
    return df[["patient_id", "timestamp", "glucose_mg_dl"]].reset_index(drop=True)


def _load_events(data_dir: Path) -> pd.DataFrame:
    """Load meal/insulin/drug event flags."""
    for pattern in ["events.csv", "event_log.csv", "meal_insulin_events.csv"]:
        p = data_dir / pattern
        if p.exists():
            df = pd.read_csv(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["patient_id"] = df["patient_id"].astype(str)
            for col in ["meal_event", "insulin_event", "drug_event"]:
                if col not in df.columns:
                    df[col] = 0
            return df[["patient_id", "timestamp", "meal_event", "insulin_event", "drug_event"]]
    return pd.DataFrame()


def _load_static(data_dir: Path) -> pd.DataFrame:
    """Load static patient demographics / labs."""
    for pattern in ["static.csv", "demographics.csv", "patient_info.csv", "baseline.csv"]:
        p = data_dir / pattern
        if p.exists():
            df = pd.read_csv(p)
            df["patient_id"] = df["patient_id"].astype(str) if "patient_id" in df.columns else df.iloc[:, 0].astype(str)
            return df
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
#  Demo data (used when real files not downloaded)
# ═════════════════════════════════════════════════════════════════════════════

def _create_demo_shanghai(n_patients: int = 98, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic Shanghai data.
    Mimics 15-min CGM + event flags + EHR for n_patients T2D patients.
    """
    rng = np.random.default_rng(seed)
    all_cgm, all_events, all_static = [], [], []

    study_duration_days = 14

    for i in range(n_patients):
        pid = f"SHA-{i+1:03d}"
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=rng.integers(0, 30))
        timestamps = pd.date_range(start, periods=study_duration_days * 24 * 4, freq="15min")
        n = len(timestamps)

        # Simulate CGM with diurnal pattern + meal spikes
        base     = rng.uniform(130, 200)
        glucose  = base + 20 * np.sin(np.linspace(0, 2 * np.pi * study_duration_days, n))
        glucose += rng.normal(0, 8, n)
        # Add meal spikes (3 meals/day roughly)
        meal_indices = [j for j in range(n) if (j % 96) in [28, 52, 76]]  # ~7am, 1pm, 7pm
        for mi in meal_indices:
            peak = rng.uniform(30, 70)
            for offset in range(min(8, n - mi)):
                glucose[mi + offset] += peak * np.exp(-0.4 * offset)
        glucose = np.clip(glucose, 40, 400)

        cgm_df = pd.DataFrame({
            "patient_id":   pid,
            "timestamp":    timestamps,
            "glucose_mg_dl": glucose.round(1),
            "is_demo":      True,
        })
        all_cgm.append(cgm_df)

        # Event flags (sparse)
        meal_ts   = [timestamps[j] for j in meal_indices if j < n]
        ins_ts    = [timestamps[j + 2] for j in meal_indices if j + 2 < n and rng.random() > 0.3]
        drug_ts   = [timestamps[j * 96] for j in range(study_duration_days) if j < n // 96]

        evs = []
        for ts in meal_ts:
            evs.append({"patient_id": pid, "timestamp": ts, "meal_event": 1, "insulin_event": 0, "drug_event": 0})
        for ts in ins_ts:
            evs.append({"patient_id": pid, "timestamp": ts, "meal_event": 0, "insulin_event": 1, "drug_event": 0})
        for ts in drug_ts:
            evs.append({"patient_id": pid, "timestamp": ts, "meal_event": 0, "insulin_event": 0, "drug_event": 1})
        if evs:
            all_events.append(pd.DataFrame(evs))

        # Static EHR
        t2d_severe = rng.random() > 0.4
        all_static.append({
            "patient_id":       pid,
            "age":              int(rng.uniform(45, 75)),
            "sex":              "M" if rng.random() > 0.45 else "F",
            "bmi":              round(rng.uniform(22, 34), 1),
            "hba1c":            round(rng.uniform(7.0, 11.0) if t2d_severe else rng.uniform(6.0, 8.5), 1),
            "fasting_glucose":  round(rng.uniform(120, 220), 0),
            "total_cholesterol":int(rng.uniform(150, 250)),
            "ldl":              int(rng.uniform(70, 160)),
            "hdl":              int(rng.uniform(30, 65)),
            "triglycerides":    int(rng.uniform(100, 300)),
            "creatinine":       round(rng.uniform(0.7, 1.8), 2),
            "egfr":             int(rng.uniform(40, 110)),
            "c_peptide":        round(rng.uniform(0.3, 2.0), 2),
            "hypertension":     int(rng.random() > 0.4),
            "macrovascular":    int(rng.random() > 0.6),
            "microvascular":    int(rng.random() > 0.5),
            "dm_years":         int(rng.uniform(1, 20)),
            "on_insulin":       int(t2d_severe and rng.random() > 0.4),
            "on_metformin":     int(rng.random() > 0.2),
        })

    cgm_out    = pd.concat(all_cgm, ignore_index=True)
    events_out = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    static_out = pd.DataFrame(all_static)

    logger.info("Shanghai demo: created %d patients, %d CGM rows", n_patients, len(cgm_out))
    return cgm_out, events_out, static_out


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_static_lookup(static_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Map patient_id → normalised static feature vector."""
    lookup: Dict[str, np.ndarray] = {}
    if static_df.empty:
        return lookup
    for _, row in static_df.iterrows():
        pid = str(row.get("patient_id", row.iloc[0]))
        vec = []
        for col in STATIC_COLS:
            if col == "sex_male":
                val = 1.0 if str(row.get("sex", "")).upper().startswith("M") else 0.0
            else:
                raw = row.get(col, np.nan)
                val = float(raw) if raw is not None else np.nan
            vec.append(val)
        lookup[pid] = np.array(vec, dtype=np.float32)
    return lookup


def _extract_event_channels(
    events_df: pd.DataFrame,
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    n_steps: int,
    step_minutes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (meal, insulin, drug) binary arrays of length n_steps."""
    meal = np.zeros(n_steps, dtype=np.float32)
    ins  = np.zeros(n_steps, dtype=np.float32)
    drug = np.zeros(n_steps, dtype=np.float32)

    if events_df.empty:
        return meal, ins, drug

    window_ev = events_df[
        (events_df.index >= w_start) & (events_df.index < w_end)
    ]
    for ts, row in window_ev.iterrows():
        idx = int((ts - w_start).total_seconds() / 60 // step_minutes)
        if 0 <= idx < n_steps:
            meal[idx] = float(row.get("meal_event", 0))
            ins[idx]  = float(row.get("insulin_event", 0))
            drug[idx] = float(row.get("drug_event", 0))

    return meal, ins, drug


def _pad_or_truncate(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target]
    return np.concatenate([arr, np.full(target - len(arr), np.nan)])
