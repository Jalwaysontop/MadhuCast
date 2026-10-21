"""
BIG IDEAs Dataset Loader
=========================
Loads and windows the BIG IDEAs Lab Glycemic Variability and Wearable Device
dataset (16 participants, 5-min Dexcom CGM, Fitbit HR, Empatica RMSSD + ACC,
self-reported meals via CGMacros-style food logs).

PhysioNet: https://physionet.org/content/big-ideas-lab-glycemic-variability-and-wearable-device-data/

Expected structure in data/raw/bigideas/:
  Per-participant folders  OR  flat files:
    P001/
      P001_CGM.csv          — Dexcom export (or Abbott Libre)
      P001_HR.csv           — Fitbit HR intraday
      P001_IBI.csv          — Empatica inter-beat interval (for RMSSD)
      P001_ACC.csv          — Empatica 3-axis accelerometer
      P001_Food.csv         — Self-reported meals (CGMacros Food_Log format)
    demographics.csv        — participant static info (optional, 2 fields)

Output: data/processed/bigideas_windows.npz
  X_seq    : (N, 72, 9)  float32  9 channels (see SEQ_CHANNELS)
  X_static : (N, 2)      float32  [age, sex_male]  (only 2 static fields available)
  y        : (N,)        int8     spike label (glucose > 180 mg/dL within 2h)
  group    : (N,)        object   participant_id for grouped CV
  g_last   : (N,)        float32  glucose mg/dL at window end
  channels : (9,)        str      channel names

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
STEP_MINUTES  = 5           # BIG IDEAs CGM cadence
HISTORY_STEPS = 72          # 6 h at 5-min steps
HORIZON_STEPS = 24          # 2 h at 5-min steps
STRIDE_STEPS  = 12          # 1-hour stride
SPIKE_THRESH  = 180.0       # mg/dL

SEQ_CHANNELS = [
    "glucose_norm",          # normalised CGM
    "hr_norm",               # heart rate (normalised)
    "rmssd_norm",            # RMSSD (HRV proxy, normalised)
    "acc_norm",              # accelerometer magnitude (normalised)
    "carbs_g",               # grams of carbs (from food log)
    "protein_g",             # grams of protein
    "fat_g",                 # grams of fat
    "fiber_g",               # grams of dietary fibre
    "rmssd_missing",         # binary: 1 if RMSSD unavailable this step
]
STATIC_COLS = ["age", "sex_male"]  # only 2 fields reliably available in BIG IDEAs

# Normalisation refs
GLUCOSE_MEAN = 120.0
GLUCOSE_STD  = 40.0
HR_MEAN      = 72.0
HR_STD       = 15.0
RMSSD_MEAN   = 35.0
RMSSD_STD    = 20.0
ACC_MEAN     = 1.0           # g (gravity units)
ACC_STD      = 0.5


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_all(data_dir: Path | str) -> Dict[str, pd.DataFrame]:
    """
    Load BIG IDEAs data from *data_dir*.

    Returns dict: ``cgm``, ``hr``, ``rmssd``, ``acc``, ``meals``, ``static``.
    Falls back to demo data when real files are missing.
    """
    data_dir = Path(data_dir)
    result = _load_real(data_dir)

    if result is None:
        logger.warning("BIG IDEAs: no real data found — using demo data")
        result = _create_demo_bigideas(n_participants=16, seed=42)
    else:
        n_pids = result["cgm"]["participant_id"].nunique() if not result["cgm"].empty else 0
        logger.info("BIG IDEAs: loaded real data — %d CGM rows, %d participants",
                    len(result["cgm"]), n_pids)

    return result


def build_npz(
    data: Dict[str, pd.DataFrame],
    output_path: Path | str = "data/processed/bigideas_windows.npz",
) -> Path:
    """
    Build windowed arrays and save as .npz for Person 2.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cgm_df   = data["cgm"]
    hr_df    = data.get("hr", pd.DataFrame())
    rmssd_df = data.get("rmssd", pd.DataFrame())
    acc_df   = data.get("acc", pd.DataFrame())
    meals_df = data.get("meals", pd.DataFrame())
    static_df = data.get("static", pd.DataFrame())

    static_lookup = _build_static_lookup(static_df)

    X_seq_list: List[np.ndarray] = []
    X_static_list: List[np.ndarray] = []
    y_list: List[int] = []
    group_list: List[str] = []
    g_last_list: List[float] = []

    participant_ids = sorted(cgm_df["participant_id"].unique())
    total_windows = 0
    hist_td = pd.Timedelta(minutes=STEP_MINUTES * HISTORY_STEPS)
    hor_td  = pd.Timedelta(minutes=STEP_MINUTES * HORIZON_STEPS)
    stride  = pd.Timedelta(minutes=STEP_MINUTES * STRIDE_STEPS)

    for pid in participant_ids:
        pid_cgm = (
            cgm_df[cgm_df["participant_id"] == pid]
            .set_index("timestamp").sort_index()
        )
        pid_hr = (
            hr_df[hr_df["participant_id"] == pid].set_index("timestamp").sort_index()
            if not hr_df.empty else pd.DataFrame()
        )
        pid_rmssd = (
            rmssd_df[rmssd_df["participant_id"] == pid].set_index("timestamp").sort_index()
            if not rmssd_df.empty else pd.DataFrame()
        )
        pid_acc = (
            acc_df[acc_df["participant_id"] == pid].set_index("timestamp").sort_index()
            if not acc_df.empty else pd.DataFrame()
        )
        pid_meals = (
            meals_df[meals_df["participant_id"] == pid]
            if not meals_df.empty else pd.DataFrame()
        )

        static_vec = static_lookup.get(str(pid), np.array([np.nan, np.nan], dtype=np.float32))

        if len(pid_cgm) < HISTORY_STEPS + HORIZON_STEPS:
            logger.debug("BIG IDEAs: skipping %s (too few CGM rows)", pid)
            continue

        t_min = pid_cgm.index.min() + hist_td
        t_max = pid_cgm.index.max() - hor_td
        current = t_min

        while current <= t_max:
            w_start = current - hist_td
            w_end   = current
            t_end   = current + hor_td

            hist_mask = (pid_cgm.index >= w_start) & (pid_cgm.index < w_end)
            tgt_mask  = (pid_cgm.index >= w_end)   & (pid_cgm.index < t_end)

            glc_hist = _pad_or_truncate(pid_cgm.loc[hist_mask, "glucose_mg_dl"].values, HISTORY_STEPS)
            glc_tgt  = _pad_or_truncate(pid_cgm.loc[tgt_mask,  "glucose_mg_dl"].values, HORIZON_STEPS)

            if np.sum(~np.isnan(glc_hist)) < HISTORY_STEPS // 2:
                current += stride
                continue

            # HR channel
            hr_arr = _resample_to_steps(pid_hr, "heart_rate_bpm", w_start, w_end,
                                        HISTORY_STEPS, STEP_MINUTES)
            # RMSSD channel + missing flag
            rmssd_arr = _resample_to_steps(pid_rmssd, "rmssd", w_start, w_end,
                                           HISTORY_STEPS, STEP_MINUTES)
            rmssd_missing = np.isnan(rmssd_arr).astype(np.float32)
            rmssd_arr = np.where(np.isnan(rmssd_arr), RMSSD_MEAN, rmssd_arr)

            # Accelerometer channel
            acc_arr = _resample_to_steps(pid_acc, "acc_mag", w_start, w_end,
                                         HISTORY_STEPS, STEP_MINUTES)

            # Meal channels
            carbs_arr, prot_arr, fat_arr, fiber_arr = _extract_meal_channels(
                pid_meals, w_start, w_end, HISTORY_STEPS, STEP_MINUTES
            )

            # Normalise
            glc_norm   = (glc_hist  - GLUCOSE_MEAN) / GLUCOSE_STD
            hr_norm    = (np.where(np.isnan(hr_arr),    HR_MEAN,    hr_arr)    - HR_MEAN)    / HR_STD
            rmssd_norm = (rmssd_arr - RMSSD_MEAN) / RMSSD_STD
            acc_norm   = (np.where(np.isnan(acc_arr),   ACC_MEAN,   acc_arr)   - ACC_MEAN)   / ACC_STD

            # Stack (72, 9)
            seq = np.stack([
                glc_norm, hr_norm, rmssd_norm, acc_norm,
                carbs_arr, prot_arr, fat_arr, fiber_arr, rmssd_missing
            ], axis=1).astype(np.float32)

            valid_tgt = glc_tgt[~np.isnan(glc_tgt)]
            spike = int(len(valid_tgt) > 0 and np.max(valid_tgt) > SPIKE_THRESH)
            g_last = float(glc_hist[~np.isnan(glc_hist)][-1]) if np.any(~np.isnan(glc_hist)) else np.nan

            X_seq_list.append(seq)
            X_static_list.append(static_vec)
            y_list.append(spike)
            group_list.append(str(pid))
            g_last_list.append(g_last)
            total_windows += 1
            current += stride

    if total_windows == 0:
        logger.error("BIG IDEAs: no windows built — check data")
        return output_path

    np.savez_compressed(
        output_path,
        X_seq       = np.stack(X_seq_list,    axis=0),
        X_static    = np.stack(X_static_list, axis=0),
        y           = np.array(y_list,        dtype=np.int8),
        group       = np.array(group_list,    dtype=object),
        g_last      = np.array(g_last_list,   dtype=np.float32),
        channels    = np.array(SEQ_CHANNELS),
        static_cols = np.array(STATIC_COLS),
    )
    logger.info(
        "BIG IDEAs: saved %d windows (%d participants) → %s",
        total_windows, len(participant_ids), output_path
    )
    logger.info("BIG IDEAs: positive rate (spike) = %.1f%%", 100 * np.mean(y_list))
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  Real file loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_real(data_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Attempt to load real BIG IDEAs files.
    Returns None if no CGM data found.
    """
    # Look for per-participant folders (P001, P002, ...) or flat files
    participant_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"P\d+", d.name, re.IGNORECASE)
    ]) if data_dir.exists() else []

    flat_cgm_files = list(data_dir.glob("*CGM*.csv")) + list(data_dir.glob("*Dexcom*.csv")) \
        if data_dir.exists() else []

    if not participant_dirs and not flat_cgm_files:
        return None

    all_cgm, all_hr, all_rmssd, all_acc, all_meals = [], [], [], [], []

    sources = participant_dirs if participant_dirs else [data_dir]

    for src in sources:
        pid = src.name if src.is_dir() else "BIG-001"

        # CGM
        for pat in ["*CGM*.csv", "*Dexcom*.csv", "*glucose*.csv", "*cgm*.csv"]:
            matches = list(src.glob(pat)) if src.is_dir() else list(data_dir.glob(pat))
            if matches:
                df = pd.read_csv(matches[0])
                df = _parse_dexcom_cgm(df, pid)
                if not df.empty:
                    all_cgm.append(df)
                break

        # HR
        for pat in ["*HR*.csv", "*heart*.csv", "*hr*.csv"]:
            matches = list(src.glob(pat)) if src.is_dir() else []
            if matches:
                df = _parse_hr(pd.read_csv(matches[0]), pid)
                if not df.empty:
                    all_hr.append(df)
                break

        # IBI → RMSSD
        for pat in ["*IBI*.csv", "*ibi*.csv", "*RR*.csv"]:
            matches = list(src.glob(pat)) if src.is_dir() else []
            if matches:
                df = _parse_ibi_to_rmssd(pd.read_csv(matches[0]), pid, STEP_MINUTES)
                if not df.empty:
                    all_rmssd.append(df)
                break

        # ACC
        for pat in ["*ACC*.csv", "*acc*.csv", "*accelerom*.csv"]:
            matches = list(src.glob(pat)) if src.is_dir() else []
            if matches:
                df = _parse_acc(pd.read_csv(matches[0]), pid, STEP_MINUTES)
                if not df.empty:
                    all_acc.append(df)
                break

        # Food log (CGMacros format)
        for pat in ["*Food*.csv", "*food*.csv", "*meal*.csv"]:
            matches = list(src.glob(pat)) if src.is_dir() else []
            if matches:
                df = _parse_food_log(pd.read_csv(matches[0]), pid)
                if not df.empty:
                    all_meals.append(df)
                break

    if not all_cgm:
        return None

    # Static demographics
    static_path = data_dir / "demographics.csv"
    static_df = pd.read_csv(static_path) if static_path.exists() else pd.DataFrame()

    return {
        "cgm":    pd.concat(all_cgm,   ignore_index=True),
        "hr":     pd.concat(all_hr,    ignore_index=True) if all_hr    else pd.DataFrame(),
        "rmssd":  pd.concat(all_rmssd, ignore_index=True) if all_rmssd else pd.DataFrame(),
        "acc":    pd.concat(all_acc,   ignore_index=True) if all_acc   else pd.DataFrame(),
        "meals":  pd.concat(all_meals, ignore_index=True) if all_meals else pd.DataFrame(),
        "static": static_df,
    }


def _parse_dexcom_cgm(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Parse Dexcom export format CGM data."""
    # Dexcom format: 'Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)'
    ts_col = next((c for c in df.columns if "timestamp" in c.lower() or "time" in c.lower()), None)
    glu_col = next((c for c in df.columns if "glucose" in c.lower() or "cgm" in c.lower() or "cbg" in c.lower()), None)
    if ts_col is None or glu_col is None:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["glucose_mg_dl"] = pd.to_numeric(df[glu_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "glucose_mg_dl"])
    df["participant_id"] = pid
    # Filter to EGV (estimated glucose value) rows if event_type column exists
    if "event type" in [c.lower() for c in df.columns]:
        ev_col = next(c for c in df.columns if c.lower() == "event type")
        df = df[df[ev_col].str.contains("EGV", na=False)]
    return df[["participant_id", "timestamp", "glucose_mg_dl"]].reset_index(drop=True)


def _parse_hr(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    ts_col  = next((c for c in df.columns if "time" in c.lower()), None)
    hr_col  = next((c for c in df.columns if "heart" in c.lower() or "hr" in c.lower() or "bpm" in c.lower()), None)
    if ts_col is None or hr_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "participant_id":  pid,
        "timestamp":       pd.to_datetime(df[ts_col], errors="coerce"),
        "heart_rate_bpm":  pd.to_numeric(df[hr_col], errors="coerce"),
    }).dropna()
    return out


def _parse_ibi_to_rmssd(df: pd.DataFrame, pid: str, window_minutes: int) -> pd.DataFrame:
    """
    Convert raw IBI (inter-beat interval) to RMSSD per window_minutes.
    IBI column in seconds. Timestamp column for alignment.
    """
    ts_col  = next((c for c in df.columns if "time" in c.lower()), None)
    ibi_col = next((c for c in df.columns if "ibi" in c.lower() or "rr" in c.lower() or "interval" in c.lower()), None)
    if ts_col is None or ibi_col is None:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["ibi"] = pd.to_numeric(df[ibi_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "ibi"])
    df = df.set_index("timestamp").sort_index()

    # Resample: compute RMSSD in each window_minutes bin
    def rmssd(x: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        diffs = np.diff(x.values)
        return float(np.sqrt(np.mean(diffs ** 2)))

    rmssd_series = df["ibi"].resample(f"{window_minutes}min").apply(rmssd)
    out = rmssd_series.reset_index()
    out.columns = ["timestamp", "rmssd"]
    out["participant_id"] = pid
    return out[["participant_id", "timestamp", "rmssd"]].dropna()


def _parse_acc(df: pd.DataFrame, pid: str, window_minutes: int) -> pd.DataFrame:
    """Parse 3-axis ACC → magnitude → resample per window_minutes."""
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    x_col  = next((c for c in df.columns if c.lower() in ["x", "acc_x", "x_axis"]), None)
    y_col  = next((c for c in df.columns if c.lower() in ["y", "acc_y", "y_axis"]), None)
    z_col  = next((c for c in df.columns if c.lower() in ["z", "acc_z", "z_axis"]), None)

    if ts_col is None:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    if x_col and y_col and z_col:
        df["acc_mag"] = np.sqrt(
            pd.to_numeric(df[x_col], errors="coerce") ** 2 +
            pd.to_numeric(df[y_col], errors="coerce") ** 2 +
            pd.to_numeric(df[z_col], errors="coerce") ** 2
        )
    else:
        # Single magnitude column
        mag_col = next((c for c in df.columns if "mag" in c.lower() or "acc" in c.lower()), None)
        if mag_col is None:
            return pd.DataFrame()
        df["acc_mag"] = pd.to_numeric(df[mag_col], errors="coerce")

    df = df.dropna(subset=["timestamp", "acc_mag"]).set_index("timestamp").sort_index()
    resampled = df["acc_mag"].resample(f"{window_minutes}min").mean().reset_index()
    resampled["participant_id"] = pid
    return resampled[["participant_id", "timestamp", "acc_mag"]].dropna()


def _parse_food_log(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Parse CGMacros-style food log CSV."""
    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if ts_col is None:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["participant_id"] = pid
    col_map = {
        "carbs_g":    next((c for c in df.columns if "carb" in c.lower()), None),
        "protein_g":  next((c for c in df.columns if "protein" in c.lower()), None),
        "fat_g":      next((c for c in df.columns if "fat" in c.lower()), None),
        "fiber_g":    next((c for c in df.columns if "fib" in c.lower()), None),
    }
    for target, src in col_map.items():
        df[target] = pd.to_numeric(df[src], errors="coerce") if src else 0.0
    cols = ["participant_id", "timestamp"] + list(col_map.keys())
    return df[cols].dropna(subset=["timestamp"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Demo data
# ═════════════════════════════════════════════════════════════════════════════

def _create_demo_bigideas(n_participants: int = 16, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic BIG IDEAs data (5-min streams, 10 days each)."""
    rng = np.random.default_rng(seed)
    all_cgm, all_hr, all_rmssd, all_acc, all_meals, all_static = [], [], [], [], [], []

    for i in range(n_participants):
        pid = f"BI-{i+1:03d}"
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        days = 10
        ts_5min = pd.date_range(start, periods=days * 24 * 12, freq="5min")
        n = len(ts_5min)

        # CGM
        base = rng.uniform(80, 160)
        glucose = base + 15 * np.sin(np.linspace(0, 2 * np.pi * days, n))
        glucose += rng.normal(0, 5, n)
        for meal_i in [j for j in range(n) if (j % 288) in [84, 156, 228]]:
            peak = rng.uniform(20, 60)
            for off in range(min(24, n - meal_i)):
                glucose[meal_i + off] += peak * np.exp(-0.3 * off)
        glucose = np.clip(glucose, 40, 400)
        all_cgm.append(pd.DataFrame({
            "participant_id": pid, "timestamp": ts_5min, "glucose_mg_dl": glucose.round(1)
        }))

        # HR
        hr = rng.uniform(58, 82) + 8 * np.sin(np.linspace(0, 2 * np.pi * days, n))
        hr += rng.normal(0, 3, n)
        all_hr.append(pd.DataFrame({
            "participant_id": pid, "timestamp": ts_5min, "heart_rate_bpm": np.clip(hr, 40, 140).round(0)
        }))

        # RMSSD (5-min windows)
        rmssd = rng.uniform(20, 60, n) + rng.normal(0, 5, n)
        rmssd_missing = rng.random(n) < 0.1
        rmssd = np.where(rmssd_missing, np.nan, rmssd)
        valid_mask = ~np.isnan(rmssd)
        if valid_mask.any():
            all_rmssd.append(pd.DataFrame({
                "participant_id": pid,
                "timestamp": ts_5min[valid_mask],
                "rmssd": rmssd[valid_mask].round(2),
            }))

        # ACC magnitude
        acc = rng.uniform(0.8, 1.2, n) + 0.3 * np.abs(rng.normal(0, 1, n))
        all_acc.append(pd.DataFrame({
            "participant_id": pid, "timestamp": ts_5min, "acc_mag": acc.round(3)
        }))

        # Meals (3/day)
        meal_rows = []
        for d in range(days):
            for hour in [7, 13, 19]:
                ts_meal = start + pd.Timedelta(days=d, hours=hour, minutes=int(rng.uniform(0, 20)))
                n_items = rng.integers(1, 4)
                for _ in range(n_items):
                    meal_rows.append({
                        "participant_id": pid,
                        "timestamp": ts_meal,
                        "carbs_g":   round(float(rng.uniform(10, 60)), 1),
                        "protein_g": round(float(rng.uniform(5, 30)), 1),
                        "fat_g":     round(float(rng.uniform(3, 25)), 1),
                        "fiber_g":   round(float(rng.uniform(1, 8)), 1),
                    })
        all_meals.append(pd.DataFrame(meal_rows))

        # Static
        all_static.append({
            "participant_id": pid,
            "age": int(rng.uniform(22, 65)),
            "sex": "M" if rng.random() > 0.5 else "F",
        })

    return {
        "cgm":    pd.concat(all_cgm,   ignore_index=True),
        "hr":     pd.concat(all_hr,    ignore_index=True),
        "rmssd":  pd.concat(all_rmssd, ignore_index=True) if all_rmssd else pd.DataFrame(),
        "acc":    pd.concat(all_acc,   ignore_index=True),
        "meals":  pd.concat(all_meals, ignore_index=True),
        "static": pd.DataFrame(all_static),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_static_lookup(static_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    if static_df.empty:
        return lookup
    pid_col = "participant_id" if "participant_id" in static_df.columns else static_df.columns[0]
    for _, row in static_df.iterrows():
        pid = str(row[pid_col])
        age = float(row.get("age", np.nan))
        sex_male = 1.0 if str(row.get("sex", "")).upper().startswith("M") else 0.0
        lookup[pid] = np.array([age, sex_male], dtype=np.float32)
    return lookup


def _resample_to_steps(
    df: pd.DataFrame, col: str, w_start: pd.Timestamp,
    w_end: pd.Timestamp, n_steps: int, step_minutes: int
) -> np.ndarray:
    """Average *col* into n_steps bins aligned to [w_start, w_end)."""
    out = np.full(n_steps, np.nan, dtype=np.float32)
    if df.empty or col not in df.columns:
        return out
    mask = (df.index >= w_start) & (df.index < w_end)
    sub = df.loc[mask, col]
    if sub.empty:
        return out
    for step in range(n_steps):
        step_start = w_start + pd.Timedelta(minutes=step * step_minutes)
        step_end   = step_start + pd.Timedelta(minutes=step_minutes)
        vals = sub[(sub.index >= step_start) & (sub.index < step_end)]
        if not vals.empty:
            out[step] = float(vals.mean())
    return out


def _extract_meal_channels(
    meals_df: pd.DataFrame, w_start: pd.Timestamp,
    w_end: pd.Timestamp, n_steps: int, step_minutes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (carbs, protein, fat, fiber) arrays of length n_steps."""
    carbs  = np.zeros(n_steps, dtype=np.float32)
    prot   = np.zeros(n_steps, dtype=np.float32)
    fat    = np.zeros(n_steps, dtype=np.float32)
    fiber  = np.zeros(n_steps, dtype=np.float32)
    if meals_df.empty:
        return carbs, prot, fat, fiber
    mask = (meals_df["timestamp"] >= w_start) & (meals_df["timestamp"] < w_end)
    for _, row in meals_df[mask].iterrows():
        idx = int((row["timestamp"] - w_start).total_seconds() / 60 // step_minutes)
        if 0 <= idx < n_steps:
            carbs[idx] += float(row.get("carbs_g", 0) or 0)
            prot[idx]  += float(row.get("protein_g", 0) or 0)
            fat[idx]   += float(row.get("fat_g", 0) or 0)
            fiber[idx] += float(row.get("fiber_g", 0) or 0)
    return carbs, prot, fat, fiber


def _pad_or_truncate(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target]
    return np.concatenate([arr, np.full(target - len(arr), np.nan)])
