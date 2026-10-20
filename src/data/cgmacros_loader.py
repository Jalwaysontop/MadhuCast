"""
CGMacros Data Loader
====================
Loads and cleans CGMacros dataset from PhysioNet.

CGMacros contains ~45 participants (15 healthy, 16 prediabetes, 14 T2D)
with 10 days of data each:
  - Continuous Glucose Monitor (CGM) readings at ~5-min intervals
  - Fitbit heart rate (intraday)
  - Fitbit activity/steps (intraday)
  - Fitbit sleep logs
  - Self-reported meal logs with macronutrients
  - Baseline labs (HbA1c, fasting glucose, lipids)
  - Demographics (age, sex, BMI)

Data source : https://www.physionet.org/content/cgmacros/
Licence     : Verify at PhysioNet — appears CC BY-NC-SA 4.0 but confirm.
Person      : Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Expected directory layout inside data/raw/cgmacros/ ──────────────────────
# Adjust these constants if the actual PhysioNet download uses different names.
# The loader is written to be resilient to minor naming variations.

DEFAULT_RAW_DIR = Path("data/raw/cgmacros")

# Sub-folder / file pattern expectations (case-insensitive glob)
# Multiple patterns per stream — the loader tries all of them.
FILE_PATTERNS = {
    "cgm":          ["*cgm*.csv"],
    "heart_rate":   ["*heart*rate*.csv"],
    "steps":        ["*step*.csv"],
    "activity":     ["*activit*.csv"],
    "sleep":        ["*sleep*.csv"],
    "meals":        ["*meal*.csv", "*food*log*.csv", "*Food_Log*.csv"],
    "demographics": ["*demo*.csv"],
    "labs":         ["*lab*.csv"],
}

# ── Diabetes status mapping ──────────────────────────────────────────────────
DIABETES_STATUS_MAP = {
    "healthy":      0,
    "control":      0,
    "normal":       0,
    "prediabetes":  1,
    "pre-diabetes": 1,
    "prediabetic":  1,
    "t2d":          2,
    "type2":        2,
    "type 2":       2,
    "diabetic":     2,
    "diabetes":     2,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Low-level file finders
# ═════════════════════════════════════════════════════════════════════════════

def _find_files(raw_dir: Path, pattern) -> List[Path]:
    """Recursively find files matching *pattern* (case-insensitive).

    Parameters
    ----------
    pattern : str or list[str]
        A single glob pattern or a list of glob patterns to try.
    """
    if isinstance(pattern, str):
        patterns = [pattern]
    else:
        patterns = list(pattern)

    matches: List[Path] = []
    for pat in patterns:
        matches.extend(sorted(raw_dir.rglob(pat)))

    if not matches:
        # Fallback: case-insensitive token matching
        all_tokens = []
        for pat in patterns:
            all_tokens.extend(pat.replace("*", "").replace(".", "").split())
        matches = [
            p for p in raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".csv"
            and any(tok in p.stem.lower() for tok in all_tokens if tok)
        ]

    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in matches:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique)


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} with any known encoding.")


# ═════════════════════════════════════════════════════════════════════════════
#  Individual stream loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_cgm(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load CGM glucose readings.

    Returns
    -------
    DataFrame with columns:
        participant_id : str
        timestamp      : datetime64[ns]
        glucose_mg_dl  : float
    """
    files = _find_files(raw_dir, FILE_PATTERNS["cgm"])
    if not files:
        logger.warning("No CGM files found in %s", raw_dir)
        return pd.DataFrame(columns=["participant_id", "timestamp", "glucose_mg_dl"])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Identify key columns by partial match
        id_col = _find_column(df, ["participant", "subject", "id", "pid"])
        ts_col = _find_column(df, ["timestamp", "time", "datetime", "date"])
        gl_col = _find_column(df, ["glucose", "cgm", "bg", "blood_glucose", "gl"])

        if id_col and ts_col and gl_col:
            chunk = df[[id_col, ts_col, gl_col]].copy()
            chunk.columns = ["participant_id", "timestamp", "glucose_mg_dl"]
            frames.append(chunk)
        else:
            logger.warning("Skipping %s — could not identify columns (id=%s, ts=%s, gl=%s)",
                           f.name, id_col, ts_col, gl_col)

    if not frames:
        return pd.DataFrame(columns=["participant_id", "timestamp", "glucose_mg_dl"])

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["glucose_mg_dl"] = pd.to_numeric(result["glucose_mg_dl"], errors="coerce")
    result.dropna(subset=["timestamp", "glucose_mg_dl"], inplace=True)

    # Basic physiological bounds: 20–500 mg/dL
    result = result[(result["glucose_mg_dl"] >= 20) & (result["glucose_mg_dl"] <= 500)]
    result.sort_values(["participant_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded CGM: %d rows, %d participants",
                len(result), result["participant_id"].nunique())
    return result


def load_heart_rate(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load Fitbit intraday heart rate.

    Returns
    -------
    DataFrame with columns:
        participant_id : str
        timestamp      : datetime64[ns]
        heart_rate_bpm : float
    """
    files = _find_files(raw_dir, FILE_PATTERNS["heart_rate"])
    if not files:
        logger.warning("No heart-rate files found in %s", raw_dir)
        return pd.DataFrame(columns=["participant_id", "timestamp", "heart_rate_bpm"])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        id_col = _find_column(df, ["participant", "subject", "id", "pid"])
        ts_col = _find_column(df, ["timestamp", "time", "datetime"])
        hr_col = _find_column(df, ["heart", "hr", "bpm", "pulse"])

        if id_col and ts_col and hr_col:
            chunk = df[[id_col, ts_col, hr_col]].copy()
            chunk.columns = ["participant_id", "timestamp", "heart_rate_bpm"]
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["participant_id", "timestamp", "heart_rate_bpm"])

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["heart_rate_bpm"] = pd.to_numeric(result["heart_rate_bpm"], errors="coerce")
    result.dropna(subset=["timestamp", "heart_rate_bpm"], inplace=True)
    result = result[(result["heart_rate_bpm"] >= 30) & (result["heart_rate_bpm"] <= 220)]
    result.sort_values(["participant_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded HR: %d rows, %d participants",
                len(result), result["participant_id"].nunique())
    return result


def load_steps(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load Fitbit intraday step counts.

    Returns
    -------
    DataFrame with columns:
        participant_id : str
        timestamp      : datetime64[ns]
        steps          : int
    """
    files = _find_files(raw_dir, FILE_PATTERNS["steps"])
    if not files:
        logger.warning("No step files found in %s", raw_dir)
        return pd.DataFrame(columns=["participant_id", "timestamp", "steps"])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        id_col = _find_column(df, ["participant", "subject", "id", "pid"])
        ts_col = _find_column(df, ["timestamp", "time", "datetime"])
        st_col = _find_column(df, ["step", "steps", "count"])

        if id_col and ts_col and st_col:
            chunk = df[[id_col, ts_col, st_col]].copy()
            chunk.columns = ["participant_id", "timestamp", "steps"]
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["participant_id", "timestamp", "steps"])

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["steps"] = pd.to_numeric(result["steps"], errors="coerce").fillna(0).astype(int)
    result.dropna(subset=["timestamp"], inplace=True)
    result.sort_values(["participant_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded steps: %d rows, %d participants",
                len(result), result["participant_id"].nunique())
    return result


def load_sleep(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load Fitbit sleep logs.

    Returns
    -------
    DataFrame with columns:
        participant_id    : str
        sleep_start       : datetime64[ns]
        sleep_end         : datetime64[ns]
        duration_minutes  : float
        efficiency        : float   (0–100 or NaN)
    """
    files = _find_files(raw_dir, FILE_PATTERNS["sleep"])
    if not files:
        logger.warning("No sleep files found in %s", raw_dir)
        return pd.DataFrame(columns=[
            "participant_id", "sleep_start", "sleep_end",
            "duration_minutes", "efficiency"
        ])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        id_col = _find_column(df, ["participant", "subject", "id", "pid"])
        start_col = _find_column(df, ["start", "begin", "sleep_start", "bedtime"])
        end_col = _find_column(df, ["end", "wake", "sleep_end", "wakeup"])
        dur_col = _find_column(df, ["duration", "minutes", "total_sleep", "sleep_duration"])
        eff_col = _find_column(df, ["efficiency", "quality", "score"])

        if id_col and (start_col or dur_col):
            cols_map = {"participant_id": id_col}
            if start_col:
                cols_map["sleep_start"] = start_col
            if end_col:
                cols_map["sleep_end"] = end_col
            if dur_col:
                cols_map["duration_minutes"] = dur_col
            if eff_col:
                cols_map["efficiency"] = eff_col

            chunk = df[[v for v in cols_map.values()]].copy()
            chunk.columns = list(cols_map.keys())
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=[
            "participant_id", "sleep_start", "sleep_end",
            "duration_minutes", "efficiency"
        ])

    result = pd.concat(frames, ignore_index=True)
    for col in ("sleep_start", "sleep_end"):
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors="coerce")
    if "duration_minutes" in result.columns:
        result["duration_minutes"] = pd.to_numeric(
            result["duration_minutes"], errors="coerce"
        )
    # Fill missing columns
    for col in ("sleep_start", "sleep_end", "duration_minutes", "efficiency"):
        if col not in result.columns:
            result[col] = np.nan

    result.sort_values(["participant_id", "sleep_start"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded sleep: %d rows, %d participants",
                len(result), result["participant_id"].nunique())
    return result


def _extract_participant_id_from_filename(filepath: Path) -> Optional[str]:
    """Extract participant ID from CGMacros Food_Log filenames.

    Examples
    --------
    Food_Log_001.csv  → '001'
    Food_Log_010.csv  → '010'
    """
    import re
    stem = filepath.stem  # e.g. 'Food_Log_001'
    match = re.search(r'(\d{2,4})$', stem)
    if match:
        return match.group(1)
    return None


def _infer_meal_type_from_time(hour: float) -> str:
    """Infer meal type from hour of day.

    CGMacros Food_Log files don't have a meal_type column, so we
    classify by time of day using standard conventions.
    """
    if 5 <= hour < 11:
        return "breakfast"
    elif 11 <= hour < 15:
        return "lunch"
    elif 15 <= hour < 17.5:
        return "snack"
    elif 17.5 <= hour < 22:
        return "dinner"
    else:
        return "snack"  # Late-night eating classified as snack


def load_meals(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load self-reported meal logs with macronutrient data.

    Handles two formats:
    1. **Standard** — CSV with participant_id, timestamp, carbs columns
    2. **CGMacros Food_Log** — per-participant files (Food_Log_001.csv)
       where the participant ID is encoded in the filename and columns
       are: date, time, time_begin, logged_food, calorie, total_carb,
       dietary_fiber, sugar, protein, total_fat.

    Returns
    -------
    DataFrame with columns:
        participant_id : str
        timestamp      : datetime64[ns]
        meal_type      : str          (breakfast / lunch / dinner / snack)
        food_name      : str          (logged food name, if available)
        calories_kcal  : float
        carbs_g        : float
        protein_g      : float
        fat_g          : float
        fibre_g        : float        (may be NaN if not available)
    """
    files = _find_files(raw_dir, FILE_PATTERNS["meals"])
    if not files:
        logger.warning("No meal files found in %s", raw_dir)
        return pd.DataFrame(columns=[
            "participant_id", "timestamp", "meal_type", "food_name",
            "calories_kcal", "carbs_g", "protein_g", "fat_g", "fibre_g"
        ])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # ── Detect if this is a CGMacros Food_Log file ────────────────────
        is_food_log = (
            "logged_food" in df.columns or "searched_food" in df.columns
        )

        if is_food_log:
            chunk = _parse_food_log(df, f)
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
            continue

        # ── Standard format (has participant_id column) ───────────────────
        id_col   = _find_column(df, ["participant", "subject", "id", "pid"])
        ts_col   = _find_column(df, ["timestamp", "time", "datetime", "date"])
        type_col = _find_column(df, ["meal_type", "type", "meal", "category"])
        cal_col  = _find_column(df, ["calorie", "calories", "kcal", "energy"])
        carb_col = _find_column(df, ["carb", "carbs", "carbohydrate", "total_carb"])
        prot_col = _find_column(df, ["protein", "prot"])
        fat_col  = _find_column(df, ["fat", "lipid", "total_fat"])
        fib_col  = _find_column(df, ["fibre", "fiber", "dietary_fiber"])

        if id_col and ts_col and carb_col:
            col_map = {
                "participant_id": id_col,
                "timestamp": ts_col,
            }
            if type_col:
                col_map["meal_type"] = type_col
            if cal_col:
                col_map["calories_kcal"] = cal_col
            col_map["carbs_g"] = carb_col
            if prot_col:
                col_map["protein_g"] = prot_col
            if fat_col:
                col_map["fat_g"] = fat_col
            if fib_col:
                col_map["fibre_g"] = fib_col

            chunk = df[[v for v in col_map.values()]].copy()
            chunk.columns = list(col_map.keys())
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=[
            "participant_id", "timestamp", "meal_type", "food_name",
            "calories_kcal", "carbs_g", "protein_g", "fat_g", "fibre_g"
        ])

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for num_col in ("calories_kcal", "carbs_g", "protein_g", "fat_g", "fibre_g"):
        if num_col in result.columns:
            result[num_col] = pd.to_numeric(result[num_col], errors="coerce")
        else:
            result[num_col] = np.nan

    if "meal_type" not in result.columns:
        result["meal_type"] = "unknown"
    result["meal_type"] = result["meal_type"].fillna("unknown").str.strip().str.lower()

    if "food_name" not in result.columns:
        result["food_name"] = ""
    result["food_name"] = result["food_name"].fillna("").astype(str)

    result.dropna(subset=["timestamp", "carbs_g"], inplace=True)
    result.sort_values(["participant_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded meals: %d rows, %d participants",
                len(result), result["participant_id"].nunique())
    return result


def _parse_food_log(df: pd.DataFrame, filepath: Path) -> Optional[pd.DataFrame]:
    """Parse a CGMacros Food_Log_XXX.csv into our standard meal format.

    CGMacros Food_Log columns:
        date, time, time_begin, time_end, logged_food, amount, unit,
        searched_food, calorie, total_carb, dietary_fiber, sugar,
        protein, total_fat

    Participant ID comes from the filename (Food_Log_001 → '001').
    Meal type is inferred from time of day.
    """
    pid = _extract_participant_id_from_filename(filepath)
    if pid is None:
        logger.warning("Cannot extract participant ID from %s — skipping", filepath)
        return None

    # ── Build timestamp from 'time_begin' or 'date' + 'time' ─────────────
    if "time_begin" in df.columns:
        ts = pd.to_datetime(df["time_begin"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce")
    else:
        logger.warning("No timestamp column in %s — skipping", filepath)
        return None

    # ── Extract food name ─────────────────────────────────────────────────
    food_name = df.get("logged_food", df.get("searched_food", pd.Series(dtype=str)))

    # ── Extract macros ────────────────────────────────────────────────────
    calorie = pd.to_numeric(df.get("calorie", np.nan), errors="coerce")
    carbs   = pd.to_numeric(df.get("total_carb", np.nan), errors="coerce")
    protein = pd.to_numeric(df.get("protein", np.nan), errors="coerce")
    fat     = pd.to_numeric(df.get("total_fat", np.nan), errors="coerce")
    fibre   = pd.to_numeric(df.get("dietary_fiber", np.nan), errors="coerce")

    # ── Infer meal type from hour ─────────────────────────────────────────
    hours = ts.dt.hour + ts.dt.minute / 60.0
    meal_type = hours.apply(
        lambda h: _infer_meal_type_from_time(h) if pd.notna(h) else "unknown"
    )

    chunk = pd.DataFrame({
        "participant_id": pid,
        "timestamp":      ts,
        "meal_type":      meal_type,
        "food_name":      food_name,
        "calories_kcal":  calorie,
        "carbs_g":        carbs,
        "protein_g":      protein,
        "fat_g":          fat,
        "fibre_g":        fibre,
    })

    # Drop rows with no usable data
    chunk.dropna(subset=["timestamp"], inplace=True)

    logger.info("  Parsed Food_Log for participant %s: %d meal entries", pid, len(chunk))
    return chunk


def load_demographics(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load baseline demographics.

    Returns
    -------
    DataFrame with columns:
        participant_id  : str
        age             : float
        sex             : str   ('M' or 'F')
        bmi             : float
        diabetes_status : int   (0=healthy, 1=prediabetes, 2=T2D)
    """
    files = _find_files(raw_dir, FILE_PATTERNS["demographics"])
    if not files:
        logger.warning("No demographics files found in %s", raw_dir)
        return pd.DataFrame(columns=[
            "participant_id", "age", "sex", "bmi", "diabetes_status"
        ])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        id_col   = _find_column(df, ["participant", "subject", "id", "pid"])
        age_col  = _find_column(df, ["age"])
        sex_col  = _find_column(df, ["sex", "gender"])
        bmi_col  = _find_column(df, ["bmi", "body_mass"])
        stat_col = _find_column(df, ["status", "diabetes", "group", "category", "diagnosis"])

        if id_col:
            col_map = {"participant_id": id_col}
            if age_col:
                col_map["age"] = age_col
            if sex_col:
                col_map["sex"] = sex_col
            if bmi_col:
                col_map["bmi"] = bmi_col
            if stat_col:
                col_map["diabetes_status_raw"] = stat_col

            chunk = df[[v for v in col_map.values()]].copy()
            chunk.columns = list(col_map.keys())
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=[
            "participant_id", "age", "sex", "bmi", "diabetes_status"
        ])

    result = pd.concat(frames, ignore_index=True)
    result["age"] = pd.to_numeric(result.get("age"), errors="coerce")
    result["bmi"] = pd.to_numeric(result.get("bmi"), errors="coerce")

    if "sex" in result.columns:
        result["sex"] = result["sex"].astype(str).str.strip().str.upper().str[0]
        result["sex"] = result["sex"].map(lambda s: s if s in ("M", "F") else np.nan)

    # Map diabetes status string → integer code
    if "diabetes_status_raw" in result.columns:
        result["diabetes_status"] = (
            result["diabetes_status_raw"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(DIABETES_STATUS_MAP)
        )
        result.drop(columns=["diabetes_status_raw"], inplace=True)
    elif "diabetes_status" not in result.columns:
        result["diabetes_status"] = np.nan

    # Ensure all expected columns exist
    for col in ("age", "sex", "bmi", "diabetes_status"):
        if col not in result.columns:
            result[col] = np.nan

    result.drop_duplicates(subset=["participant_id"], keep="first", inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded demographics: %d participants", len(result))
    return result


def load_labs(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load baseline lab results.

    Returns
    -------
    DataFrame with columns:
        participant_id   : str
        hba1c            : float   (%)
        fasting_glucose  : float   (mg/dL)
        total_cholesterol: float   (mg/dL, may be NaN)
        ldl              : float   (mg/dL, may be NaN)
        hdl              : float   (mg/dL, may be NaN)
        triglycerides    : float   (mg/dL, may be NaN)
    """
    files = _find_files(raw_dir, FILE_PATTERNS["labs"])
    if not files:
        logger.warning("No lab files found in %s", raw_dir)
        return pd.DataFrame(columns=[
            "participant_id", "hba1c", "fasting_glucose",
            "total_cholesterol", "ldl", "hdl", "triglycerides"
        ])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        id_col   = _find_column(df, ["participant", "subject", "id", "pid"])
        hba_col  = _find_column(df, ["hba1c", "a1c", "glycated"])
        fg_col   = _find_column(df, ["fasting_glucose", "fasting", "fbg", "fpg"])
        tc_col   = _find_column(df, ["total_cholesterol", "cholesterol", "tc"])
        ldl_col  = _find_column(df, ["ldl"])
        hdl_col  = _find_column(df, ["hdl"])
        tg_col   = _find_column(df, ["triglyceride", "tg", "trigs"])

        if id_col:
            col_map = {"participant_id": id_col}
            for name, col in [
                ("hba1c", hba_col), ("fasting_glucose", fg_col),
                ("total_cholesterol", tc_col), ("ldl", ldl_col),
                ("hdl", hdl_col), ("triglycerides", tg_col),
            ]:
                if col:
                    col_map[name] = col

            chunk = df[[v for v in col_map.values()]].copy()
            chunk.columns = list(col_map.keys())
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=[
            "participant_id", "hba1c", "fasting_glucose",
            "total_cholesterol", "ldl", "hdl", "triglycerides"
        ])

    result = pd.concat(frames, ignore_index=True)
    for col in ("hba1c", "fasting_glucose", "total_cholesterol",
                "ldl", "hdl", "triglycerides"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[col] = np.nan

    result.drop_duplicates(subset=["participant_id"], keep="first", inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info("Loaded labs: %d participants", len(result))
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  High-level loader: assemble everything
# ═════════════════════════════════════════════════════════════════════════════

def load_all(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Load all CGMacros streams in one call.

    Returns
    -------
    dict with keys:
        'cgm', 'heart_rate', 'steps', 'sleep',
        'meals', 'demographics', 'labs'
    Each value is a cleaned DataFrame.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CGMacros raw directory not found: {raw_dir}\n"
            "Download from https://www.physionet.org/content/cgmacros/ "
            "and place files under data/raw/cgmacros/"
        )

    data = {
        "cgm":          load_cgm(raw_dir),
        "heart_rate":   load_heart_rate(raw_dir),
        "steps":        load_steps(raw_dir),
        "sleep":        load_sleep(raw_dir),
        "meals":        load_meals(raw_dir),
        "demographics": load_demographics(raw_dir),
        "labs":         load_labs(raw_dir),
    }

    _log_summary(data)
    return data


def get_participant_ids(data: Dict[str, pd.DataFrame]) -> List[str]:
    """Return sorted list of participant IDs present in all *required* streams.

    CGM is mandatory — we cannot build glucose forecast windows without it.
    If CGM is absent the function returns an empty list so the caller knows
    to fall back to demo data.
    """
    # CGM is non-negotiable
    if "cgm" not in data or data["cgm"].empty:
        logger.warning("CGM stream is empty — cannot identify complete participants.")
        return []

    sets = [set(data["cgm"]["participant_id"].unique())]
    for key in ("demographics",):
        if key in data and not data[key].empty:
            sets.append(set(data[key]["participant_id"].unique()))

    common = sorted(sets[0].intersection(*sets[1:]))
    logger.info("Participants with complete data: %d", len(common))
    return common


def resample_cgm(
    cgm_df: pd.DataFrame,
    freq: str = "5min",
    method: str = "linear",
) -> pd.DataFrame:
    """
    Resample CGM to a uniform time grid per participant.

    Parameters
    ----------
    cgm_df : DataFrame with participant_id, timestamp, glucose_mg_dl
    freq   : Target frequency (default '5min')
    method : Interpolation method (default 'linear')

    Returns
    -------
    DataFrame on a regular grid; gaps > 30 min are left as NaN.
    """
    MAX_GAP = pd.Timedelta("30min")
    resampled_parts: List[pd.DataFrame] = []

    for pid, grp in cgm_df.groupby("participant_id"):
        grp = grp.set_index("timestamp").sort_index()
        # Detect large gaps
        diffs = grp.index.to_series().diff()

        # Create uniform index
        new_idx = pd.date_range(grp.index.min(), grp.index.max(), freq=freq)
        uniform = grp["glucose_mg_dl"].reindex(
            grp.index.union(new_idx)
        ).interpolate(method=method).reindex(new_idx)

        # Null out values in large gaps
        orig_times = grp.index
        for i, t in enumerate(new_idx):
            distances = np.abs((orig_times - t).total_seconds())
            if distances.min() > MAX_GAP.total_seconds():
                uniform.iloc[i] = np.nan

        part_df = pd.DataFrame({
            "participant_id": pid,
            "timestamp": new_idx,
            "glucose_mg_dl": uniform.values,
        })
        resampled_parts.append(part_df)

    result = pd.concat(resampled_parts, ignore_index=True)
    logger.info("Resampled CGM to %s: %d rows", freq, len(result))
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first column in *df* whose name contains any of *candidates*."""
    for col in df.columns:
        col_lower = col.lower()
        for cand in candidates:
            if cand in col_lower:
                return col
    return None


def _log_summary(data: Dict[str, pd.DataFrame]) -> None:
    """Log a one-line summary per stream."""
    for key, df in data.items():
        if df.empty:
            logger.info("  %-15s : EMPTY", key)
        else:
            n_pid = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
            logger.info("  %-15s : %6d rows, %s participants", key, len(df), n_pid)
