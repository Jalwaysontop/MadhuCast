#!/usr/bin/env python3
"""
MadhuTwin — End-to-End Data Pipeline  (scripts/build_dataset.py)
================================================================
Reproducible dataset build with ONE command:

    python scripts/build_dataset.py

Steps:
  1.  Load & validate CGMacros data      (data/raw/cgmacros/)
  2.  Generate synthetic EHR cohort      (Synthea or Python fallback)
  3.  Pair EHR ↔ wearable               (nearest-neighbour)
  4.  Build Indian meal engine           (validate food DB)
  5.  Augment meals                      (Indian meal substitution)
  6.  Build feature windows              (4h history → 2h horizon)
  7.  Create participant-level splits    (train/val/test)
  8.  Save CGMacros windows.pkl          (Person 2 interface)
  9.  Build Shanghai windows .npz        (data/raw/shanghai/)
  10. Build BIG IDEAs windows .npz       (data/raw/bigideas/)
  11. Save CGMacros .npz                 (uniform format for Person 2)

Outputs (interface contracts → Person 1 & Person 2):
  data/processed/splits.json               →  Person 2
  data/processed/windows.pkl               →  Person 2  (CGMacros)
  data/processed/cgmacros_windows.npz      →  Person 2  (uniform NPZ)
  data/processed/shanghai_windows.npz      →  Person 2
  data/processed/bigideas_windows.npz      →  Person 2
  data/indian_food_db.csv                  →  Person 1  (already exists)
  data/processed/patient_profiles/         →  Person 1
  data/pairing_log.csv                     →  Audit trail

Seeds are fixed at every step for reproducibility.

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.cgmacros_loader import (
    load_all as load_cgmacros,
    get_participant_ids,
    resample_cgm,
)
from src.data.synthea_generator import (
    _generate_synthetic_ehr,
    parse_fhir_bundles,
    export_patient_profiles,
)
from src.data.pairing import (
    pair_ehr_to_wearable,
    save_pairing_log,
    build_paired_dataset,
)
from src.data.meal_engine import IndianFoodDB, run_sensitivity_test
from src.data.augmentation import (
    augment_meals,
    compare_distributions,
    mark_augmented_windows,
)
from src.data.feature_builder import (
    create_participant_splits,
    save_splits,
    build_windows,
    save_windows,
    windows_to_dataframe,
)
# Multi-dataset loaders (Steps 9, 10, 11)
from src.data.shanghai_loader import load_all as load_shanghai, build_npz as build_shanghai_npz
from src.data.bigideas_loader import load_all as load_bigideas, build_npz as build_bigideas_npz

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")

# ── Default paths ─────────────────────────────────────────────────────────────
CGMACROS_DIR       = PROJECT_ROOT / "data" / "raw" / "cgmacros"
SHANGHAI_DIR       = PROJECT_ROOT / "data" / "raw" / "shanghai"
BIGIDEAS_DIR       = PROJECT_ROOT / "data" / "raw" / "bigideas"
SYNTHEA_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "synthea"
FOOD_DB_PATH       = PROJECT_ROOT / "data" / "indian_food_db.csv"
PROCESSED_DIR      = PROJECT_ROOT / "data" / "processed"
PAIRING_LOG_PATH   = PROJECT_ROOT / "data" / "pairing_log.csv"
PROFILES_DIR       = PROCESSED_DIR / "patient_profiles"
SPLITS_PATH        = PROCESSED_DIR / "splits.json"
WINDOWS_PATH       = PROCESSED_DIR / "windows.pkl"
CGMACROS_NPZ_PATH  = PROCESSED_DIR / "cgmacros_windows.npz"
SHANGHAI_NPZ_PATH  = PROCESSED_DIR / "shanghai_windows.npz"
BIGIDEAS_NPZ_PATH  = PROCESSED_DIR / "bigideas_windows.npz"


def main(
    seed: int = 42,
    num_synthetic_patients: int = 100,
    augmentation_fraction: float = 0.30,
    skip_cgmacros: bool = False,
    stride_minutes: int = 15,
):
    """Run the full pipeline."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("MadhuTwin Data Pipeline — Starting")
    logger.info("  Seed: %d", seed)
    logger.info("  Synthetic patients: %d", num_synthetic_patients)
    logger.info("  Augmentation fraction: %.0f%%", augmentation_fraction * 100)
    logger.info("  Window stride: %d min", stride_minutes)
    logger.info("=" * 70)

    # ── Step 1: Load CGMacros ─────────────────────────────────────────────
    logger.info("\n[Step 1/8] Loading CGMacros data…")

    # Check if real Food_Log files exist (even if we don't have CGM/HR/etc.)
    from src.data.cgmacros_loader import load_meals as _load_meals_standalone
    real_meals_df = None
    if CGMACROS_DIR.exists():
        _test_meals = _load_meals_standalone(CGMACROS_DIR)
        if not _test_meals.empty:
            real_meals_df = _test_meals
            logger.info("  Found %d real meal entries from %d participants (Food_Log files)",
                        len(real_meals_df), real_meals_df["participant_id"].nunique())

    if CGMACROS_DIR.exists() and not skip_cgmacros:
        cgmacros_data = load_cgmacros(CGMACROS_DIR)
        cgm_participants = get_participant_ids(cgmacros_data)

        if not cgm_participants:
            logger.warning("No complete participants found (missing CGM/demographics). Using demo mode.")
            cgmacros_data, cgm_participants = _create_demo_cgmacros(seed)
        else:
            # Resample CGM to uniform 5-min grid (only if CGM data present)
            if not cgmacros_data["cgm"].empty:
                cgmacros_data["cgm"] = resample_cgm(cgmacros_data["cgm"])
    else:
        logger.info("CGMacros not found at %s — generating demo data.", CGMACROS_DIR)
        cgmacros_data, cgm_participants = _create_demo_cgmacros(seed)

    # If we have real Food_Log files, overlay them onto the demo data.
    # KEY FIX: map each real participant 1-to-1 to a unique demo participant
    # (no reuse) so meals don't stack — then align timestamps to that demo
    # participant's CGM date range so meal features are physically plausible.
    if real_meals_df is not None and not real_meals_df.empty:
        real_pids = sorted(real_meals_df["participant_id"].unique().tolist())
        demo_pids = sorted(cgm_participants)

        # Strict 1:1 mapping — only as many real pids as we have demo pids
        n_map = min(len(real_pids), len(demo_pids))
        pid_map = {real_pids[i]: demo_pids[i] for i in range(n_map)}
        # Drop real participants that have no demo slot
        unmapped = [p for p in real_pids if p not in pid_map]
        if unmapped:
            logger.info("  Dropping %d excess real participants (no demo slot): %s",
                        len(unmapped), unmapped)

        real_meals_df = real_meals_df[real_meals_df["participant_id"].isin(pid_map)].copy()
        real_meals_df["original_participant_id"] = real_meals_df["participant_id"]
        real_meals_df["participant_id"] = real_meals_df["participant_id"].map(pid_map)

        # Align each real participant's meal timestamps to the matched demo
        # participant's CGM window so meals fall within the CGM date range.
        cgm_by_pid = cgmacros_data["cgm"].groupby("participant_id")["timestamp"]
        aligned_parts = []
        for rpid, dpid in pid_map.items():
            mask = real_meals_df["original_participant_id"] == rpid
            chunk = real_meals_df[mask].copy()
            if chunk.empty:
                continue

            # Get demo CGM window for this participant
            if dpid in cgm_by_pid.groups:
                demo_cgm_start = cgm_by_pid.get_group(dpid).min()
                demo_cgm_end   = cgm_by_pid.get_group(dpid).max()
            else:
                demo_cgm_start = cgmacros_data["cgm"]["timestamp"].min()
                demo_cgm_end   = cgmacros_data["cgm"]["timestamp"].max()

            # Shift meals so their start aligns with demo CGM start
            meal_start = chunk["timestamp"].min()
            if pd.notna(meal_start) and pd.notna(demo_cgm_start):
                offset = demo_cgm_start - meal_start
                chunk["timestamp"] = chunk["timestamp"] + offset

            # Clip to demo CGM window
            chunk = chunk[
                (chunk["timestamp"] >= demo_cgm_start) &
                (chunk["timestamp"] <= demo_cgm_end)
            ]
            aligned_parts.append(chunk)

        if aligned_parts:
            real_meals_aligned = pd.concat(aligned_parts, ignore_index=True)
            cgmacros_data["meals"] = real_meals_aligned
            logger.info("  ✓ Overlaid %d real meal entries (1:1 per participant, timestamp-aligned)",
                        len(real_meals_aligned))
        else:
            logger.warning("  No meal entries survived timestamp alignment.")


    logger.info("  Participants with complete data: %d", len(cgm_participants))

    # ── Step 2: Generate synthetic EHR ────────────────────────────────────
    logger.info("\n[Step 2/8] Generating synthetic EHR cohort…")
    SYNTHEA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _generate_synthetic_ehr(
        num_patients=num_synthetic_patients,
        output_dir=SYNTHEA_OUTPUT_DIR,
        seed=seed,
    )
    ehr_df = parse_fhir_bundles(SYNTHEA_OUTPUT_DIR)
    logger.info("  Generated %d synthetic EHR records", len(ehr_df))

    # ── Step 3: Pair EHR ↔ wearable ──────────────────────────────────────
    logger.info("\n[Step 3/8] Pairing EHR to CGMacros participants…")
    pairing_df = pair_ehr_to_wearable(
        ehr_df=ehr_df,
        cgmacros_demographics=cgmacros_data["demographics"],
        cgmacros_labs=cgmacros_data["labs"],
        seed=seed,
    )
    save_pairing_log(pairing_df, PAIRING_LOG_PATH)
    logger.info("  Pairing log saved: %s", PAIRING_LOG_PATH)

    # Build paired dataset (EHR + wearable)
    paired_data = build_paired_dataset(pairing_df, ehr_df, cgmacros_data)

    # Export patient profiles for Person 1
    export_patient_profiles(ehr_df, PROFILES_DIR)
    logger.info("  Patient profiles exported: %s", PROFILES_DIR)

    # ── Step 4: Validate Indian food DB ───────────────────────────────────
    logger.info("\n[Step 4/8] Loading & validating Indian Food Database…")
    food_db = IndianFoodDB(FOOD_DB_PATH)
    logger.info("  Food DB: %d items, %s", len(food_db), food_db.get_categories())

    sensitivity = run_sensitivity_test(food_db)
    n_pass = sensitivity["monotonic"].sum()
    logger.info("  Sensitivity test: %d/%d passed monotonicity", n_pass, len(sensitivity))
    sensitivity.to_csv(PROCESSED_DIR / "sensitivity_test.csv", index=False)

    # ── Step 5: Augment meals ─────────────────────────────────────────────
    logger.info("\n[Step 5/8] Augmenting meals with Indian substitutions…")
    # Use cgmacros_data["meals"] — NOT paired_data["meals"].
    # paired_data duplicates each meal row once per matched synthetic patient
    # (~25× for 100 EHR → 4 wearable pids), producing impossible feature values.
    # cgmacros_data["meals"] holds the 1:1 real-aligned meal rows set at Step 1.
    original_meals = cgmacros_data["meals"]

    if not original_meals.empty:
        augmented_meals = augment_meals(
            original_meals, food_db,
            augmentation_fraction=augmentation_fraction,
            seed=seed,
        )

        # Distribution comparison for the report
        dist_comparison = compare_distributions(original_meals, augmented_meals)
        dist_comparison.to_csv(PROCESSED_DIR / "augmentation_distribution.csv", index=False)
        logger.info("  Distribution comparison saved")
        logger.info("\n%s", dist_comparison.to_string(index=False))
    else:
        augmented_meals = original_meals
        logger.warning("  No meals to augment.")

    # Keep cgmacros_data["meals"] in sync (already correct, but be explicit)
    cgmacros_data["meals"] = augmented_meals


    # ── Step 6: Build feature windows ─────────────────────────────────────
    logger.info("\n[Step 6/8] Building feature windows (4h history → 2h horizon)…")

    # Use paired_patient_id if available, otherwise participant_id
    cgm_for_windows = paired_data.get("cgm", cgmacros_data["cgm"])
    hr_for_windows  = paired_data.get("heart_rate", cgmacros_data["heart_rate"])
    steps_for_windows = paired_data.get("steps", cgmacros_data["steps"])
    sleep_for_windows = paired_data.get("sleep", cgmacros_data["sleep"])
    # Meals must NOT come from paired_data: build_paired_dataset duplicates
    # each meal row once per synthetic patient (100 EHR → 4 pids = ~25× rows),
    # giving impossible meal_count_4h values. augmented_meals has the correct
    # 1:1 real rows keyed by wearable participant_id.
    meals_for_windows = augmented_meals


    # Determine participant IDs to window
    window_pids = sorted(cgm_for_windows["participant_id"].unique())

    # ── KEY FIX: build EHR lookup keyed by CGMacros participant_id ─────────
    # The EHR records use SYN-XXXX IDs; the pairing_log maps them to
    # DEMO-XXX participant IDs. We rebuild ehr_df indexed by the wearable
    # participant ID so build_windows() can look them up correctly.
    ehr_for_windows = ehr_df.copy()
    if not pairing_df.empty and "cgmacros_participant_id" in pairing_df.columns:
        # Take one EHR record per wearable participant (best-match)
        pid_to_syn = (
            pairing_df
            .sort_values("match_distance")
            .drop_duplicates(subset="cgmacros_participant_id", keep="first")
            .set_index("cgmacros_participant_id")["synthetic_patient_id"]
        )
        # Build a lookup: for each wearable pid, find its EHR row
        ehr_rows = []
        for wpid in window_pids:
            syn_pid = pid_to_syn.get(wpid)
            if syn_pid is not None:
                match = ehr_df[ehr_df["patient_id"] == syn_pid]
                if not match.empty:
                    row = match.iloc[0].copy()
                    row["participant_id"] = wpid  # re-key by wearable ID
                    ehr_rows.append(row)
        if ehr_rows:
            ehr_for_windows = pd.DataFrame(ehr_rows).reset_index(drop=True)
            logger.info("  Built EHR lookup for %d wearable participants", len(ehr_for_windows))
        else:
            logger.warning("  No EHR rows matched wearable participant IDs — static features will be NaN")

    windows = build_windows(
        cgm_df=cgm_for_windows,
        hr_df=hr_for_windows,
        steps_df=steps_for_windows,
        sleep_df=sleep_for_windows,
        meals_df=meals_for_windows,
        ehr_df=ehr_for_windows,      # re-keyed by wearable participant_id
        participant_ids=window_pids,
        stride_minutes=stride_minutes,
    )

    # Mark windows that contain augmented meals
    windows = mark_augmented_windows(windows, augmented_meals)
    logger.info("  Total windows: %d", len(windows))

    # ── Step 7: Create splits ─────────────────────────────────────────────
    logger.info("\n[Step 7/8] Creating participant-level train/val/test splits…")
    splits = create_participant_splits(window_pids, seed=seed)
    save_splits(splits, SPLITS_PATH)

    # Log split statistics
    for split_name, pids in splits.items():
        n_windows = sum(1 for w in windows if w["participant_id"] in pids)
        logger.info("  %s: %d participants, %d windows", split_name, len(pids), n_windows)

    # ── Step 8: Save CGMacros windows.pkl ────────────────────────────────
    logger.info("\n[Step 8/11] Saving CGMacros windows…")
    save_windows(windows, WINDOWS_PATH)

    # Flat CSV summary for quick inspection
    windows_summary = windows_to_dataframe(windows)
    windows_summary.to_csv(PROCESSED_DIR / "windows_summary.csv", index=False)
    logger.info("  Windows summary: %s", PROCESSED_DIR / "windows_summary.csv")

    # ── Step 9: Shanghai dataset → NPZ ───────────────────────────────────
    logger.info("\n[Step 9/11] Building Shanghai dataset windows…")
    SHANGHAI_DIR.mkdir(parents=True, exist_ok=True)
    shanghai_data = load_shanghai(SHANGHAI_DIR)
    n_sha = shanghai_data["cgm"]["patient_id"].nunique() if not shanghai_data["cgm"].empty else 0
    is_sha_real = "is_demo" not in shanghai_data["cgm"].columns
    logger.info("  Shanghai: %d patients (%s)", n_sha, "real" if is_sha_real else "demo fallback")
    sha_path = build_shanghai_npz(shanghai_data, SHANGHAI_NPZ_PATH)
    logger.info("  Saved → %s", sha_path)

    # ── Step 10: BIG IDEAs dataset → NPZ ─────────────────────────────────
    logger.info("\n[Step 10/11] Building BIG IDEAs dataset windows…")
    BIGIDEAS_DIR.mkdir(parents=True, exist_ok=True)
    bigideas_data = load_bigideas(BIGIDEAS_DIR)
    n_bi = bigideas_data["cgm"]["participant_id"].nunique() if not bigideas_data["cgm"].empty else 0
    is_bi_real = "is_demo" not in bigideas_data["cgm"].columns
    logger.info("  BIG IDEAs: %d participants (%s)", n_bi, "real" if is_bi_real else "demo fallback")
    bi_path = build_bigideas_npz(bigideas_data, BIGIDEAS_NPZ_PATH)
    logger.info("  Saved → %s", bi_path)

    # ── Step 11: CGMacros .npz (uniform format for Person 2) ─────────────
    logger.info("\n[Step 11/11] Saving CGMacros windows as .npz (uniform format)…")
    _save_cgmacros_npz(windows, splits, CGMACROS_NPZ_PATH)
    logger.info("  Saved → %s", CGMACROS_NPZ_PATH)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 70)
    logger.info("Outputs:")
    logger.info("  %-50s  (→ Person 2)", SPLITS_PATH)
    logger.info("  %-50s  (→ Person 2)", WINDOWS_PATH)
    logger.info("  %-50s  (→ Person 2, CGMacros NPZ)", CGMACROS_NPZ_PATH)
    logger.info("  %-50s  (→ Person 2, Shanghai NPZ)", SHANGHAI_NPZ_PATH)
    logger.info("  %-50s  (→ Person 2, BIG IDEAs NPZ)", BIGIDEAS_NPZ_PATH)
    logger.info("  %-50s  (→ Person 1)", PROFILES_DIR)
    logger.info("  %-50s  (→ Person 1)", FOOD_DB_PATH)
    logger.info("  %-50s  (audit)", PAIRING_LOG_PATH)
    logger.info("  %-50s  (report)", PROCESSED_DIR / "augmentation_distribution.csv")
    logger.info("  %-50s  (report)", PROCESSED_DIR / "sensitivity_test.csv")
    logger.info("=" * 70)



# ═════════════════════════════════════════════════════════════════════════════
#  CGMacros NPZ exporter  (uniform format for Person 2)
# ═════════════════════════════════════════════════════════════════════════════

def _save_cgmacros_npz(windows: list, splits: dict, output_path: Path) -> None:
    """
    Convert CGMacros windows (list of dicts from feature_builder) into the
    same .npz format as shanghai_windows.npz and bigideas_windows.npz so that
    Person 2 can load all three datasets identically.

    X_seq    : (N, 48, 6)  float32  [glucose_norm, hr_norm, steps_norm,
                                      carbs_g, protein_g, fat_g]
    X_static : (N, 16)     float32  EHR static vector
    y        : (N,)        int8     spike label (glucose >180 within 2h)
    group    : (N,)        object   participant_id
    g_last   : (N,)        float32  glucose at window end
    channels : (6,)        str
    split    : (N,)        str      "train" | "val" | "test"
    """
    SEQ_CH = ["glucose_norm", "hr_norm", "steps_norm", "carbs_g", "protein_g", "fat_g"]
    STATIC_CH = [
        "age", "sex_male", "bmi", "hba1c", "fasting_glucose",
        "total_cholesterol", "ldl", "hdl", "triglycerides",
        "genetic_risk_score", "diabetes_duration_years",
        "comorbidity_hypertension", "comorbidity_ckd", "comorbidity_cvd",
        "comorbidity_retinopathy", "comorbidity_neuropathy",
    ]
    GLUCOSE_MEAN, GLUCOSE_STD = 150.0, 50.0
    HR_MEAN, HR_STD           = 72.0, 15.0
    STEPS_MEAN, STEPS_STD     = 40.0, 30.0

    # Build split lookup
    pid_to_split: dict = {}
    for split_name, pids in splits.items():
        for p in pids:
            pid_to_split[p] = split_name

    X_seq_list, X_static_list, y_list, group_list, g_last_list, split_list = [], [], [], [], [], []

    for w in windows:
        pid = w.get("participant_id", "")

        # Build 6-channel sequence from stored scalar features
        # (windows store 5-min scalar aggregates; reshape to (T, 6))
        # We reconstruct a flat sequence using the stored history arrays if present,
        # otherwise use the scalar summary values duplicated across T steps.
        glucose_hist = np.array(w.get("glucose_history", []), dtype=np.float32)
        hr_hist      = np.array(w.get("hr_history",      []), dtype=np.float32)
        steps_hist   = np.array(w.get("steps_history",   []), dtype=np.float32)

        T = max(len(glucose_hist), 48)  # target: 4h @5min = 48
        if len(glucose_hist) == 0:
            # Fall back to scalar summary duplicated across T
            g_val = float(w.get("glucose_mean_4h", 150.0))
            h_val = float(w.get("hr_mean_4h",      72.0))
            s_val = float(w.get("steps_4h",         0.0))
            glucose_hist = np.full(T, g_val, dtype=np.float32)
            hr_hist      = np.full(T, h_val, dtype=np.float32)
            steps_hist   = np.full(T, s_val / T, dtype=np.float32)

        # Pad / truncate to T=48
        def _fit(a: np.ndarray) -> np.ndarray:
            if len(a) >= 48:
                return a[:48]
            return np.pad(a, (48 - len(a), 0), constant_values=np.nan)

        g_norm = (_fit(glucose_hist) - GLUCOSE_MEAN) / GLUCOSE_STD
        h_norm = (_fit(hr_hist)      - HR_MEAN)      / HR_STD
        s_norm = (_fit(steps_hist)   - STEPS_MEAN)   / STEPS_STD

        # Meal channels — distribute carbs/protein/fat into 48 steps at the
        # recorded meal time index (best effort from scalar features)
        carbs_ch   = np.zeros(48, dtype=np.float32)
        protein_ch = np.zeros(48, dtype=np.float32)
        fat_ch     = np.zeros(48, dtype=np.float32)
        meal_step  = max(0, 48 - int(w.get("time_since_last_meal_min", 60) / 5))
        if 0 <= meal_step < 48:
            carbs_ch[meal_step]   = float(w.get("recent_meal_carbs_g",  0.0) or 0.0)
            protein_ch[meal_step] = float(w.get("recent_meal_calories", 0.0) or 0.0) * 0.05
            fat_ch[meal_step]     = float(w.get("recent_meal_gl",       0.0) or 0.0) * 0.02

        seq = np.stack([g_norm, h_norm, s_norm, carbs_ch, protein_ch, fat_ch], axis=1)  # (48,6)

        # Static vector
        static_vec = np.array(
            [float(w.get(k, np.nan) or np.nan) for k in STATIC_CH],
            dtype=np.float32
        )

        # Target
        spike = int(w.get("label_spike_2h", 0) or 0)
        g_last = float(w.get("glucose_last", w.get("glucose_mean_4h", np.nan)) or np.nan)

        X_seq_list.append(seq.astype(np.float32))
        X_static_list.append(static_vec)
        y_list.append(spike)
        group_list.append(str(pid))
        g_last_list.append(g_last)
        split_list.append(pid_to_split.get(pid, "unknown"))

    if not X_seq_list:
        logger.warning("CGMacros NPZ: no windows to save")
        return

    np.savez_compressed(
        output_path,
        X_seq       = np.stack(X_seq_list,    axis=0),
        X_static    = np.stack(X_static_list, axis=0),
        y           = np.array(y_list,        dtype=np.int8),
        group       = np.array(group_list,    dtype=object),
        g_last      = np.array(g_last_list,   dtype=np.float32),
        split       = np.array(split_list,    dtype=object),
        channels    = np.array(SEQ_CH),
        static_cols = np.array(STATIC_CH),
    )
    logger.info(
        "CGMacros NPZ: saved %d windows → %s  (shapes: X_seq=%s, X_static=%s)",
        len(X_seq_list), output_path,
        np.stack(X_seq_list).shape, np.stack(X_static_list).shape,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Demo data generator  (when CGMacros isn't downloaded yet)
# ═════════════════════════════════════════════════════════════════════════════


def _create_demo_cgmacros(seed: int = 42):
    """
    Create a small synthetic CGMacros-like dataset for development
    and testing before the real PhysioNet data is downloaded.

    Generates 10 synthetic participants with 3 days of data each.
    """
    rng = np.random.default_rng(seed)
    n_participants = 10
    days_per_participant = 3

    logger.info("Generating demo CGMacros data (%d participants, %d days each)…",
                n_participants, days_per_participant)

    # Demographics
    statuses = [0] * 3 + [1] * 3 + [2] * 4  # 3 healthy, 3 prediabetes, 4 T2D
    demo_rows = []
    lab_rows = []
    all_cgm = []
    all_hr = []
    all_steps = []
    all_sleep = []
    all_meals = []

    for i in range(n_participants):
        pid = f"DEMO-{i+1:03d}"
        status = statuses[i]
        age = int(rng.integers(30, 65))
        sex = rng.choice(["M", "F"])
        bmi = round(float(rng.normal(26, 4)), 1)

        # Labs based on diabetes status
        if status == 0:
            hba1c = round(float(rng.normal(5.2, 0.3)), 1)
            fg = round(float(rng.normal(90, 8)), 0)
        elif status == 1:
            hba1c = round(float(rng.normal(6.0, 0.3)), 1)
            fg = round(float(rng.normal(108, 10)), 0)
        else:
            hba1c = round(float(rng.normal(8.0, 1.2)), 1)
            fg = round(float(rng.normal(150, 35)), 0)

        demo_rows.append({
            "participant_id": pid, "age": age, "sex": sex,
            "bmi": bmi, "diabetes_status": status,
        })
        lab_rows.append({
            "participant_id": pid, "hba1c": hba1c, "fasting_glucose": fg,
            "total_cholesterol": round(float(rng.normal(200, 35)), 0),
            "ldl": round(float(rng.normal(120, 30)), 0),
            "hdl": round(float(rng.normal(45, 10)), 0),
            "triglycerides": round(float(rng.normal(160, 50)), 0),
        })

        # Generate time-series data
        base_glucose = fg  # Baseline around fasting glucose
        for day in range(days_per_participant):
            day_start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)

            # CGM: 5-min intervals, 288 readings per day
            for minute in range(0, 1440, 5):
                ts = day_start + pd.Timedelta(minutes=minute)
                hour = minute / 60

                # Simulate meal spikes
                meal_effect = 0
                if 7 <= hour <= 9:    # Breakfast
                    meal_effect = 40 * np.exp(-0.5 * (hour - 8) ** 2)
                elif 12 <= hour <= 14:  # Lunch
                    meal_effect = 50 * np.exp(-0.5 * (hour - 13) ** 2)
                elif 19 <= hour <= 21:  # Dinner
                    meal_effect = 45 * np.exp(-0.5 * (hour - 20) ** 2)

                # Dawn phenomenon
                dawn_effect = 10 * np.exp(-0.5 * (hour - 5) ** 2) if status >= 1 else 0

                glucose = base_glucose + meal_effect + dawn_effect + rng.normal(0, 8)
                glucose = np.clip(glucose, 50, 350)

                all_cgm.append({
                    "participant_id": pid, "timestamp": ts,
                    "glucose_mg_dl": round(float(glucose), 1),
                })

            # Heart rate: 1-min intervals
            for minute in range(0, 1440, 5):
                ts = day_start + pd.Timedelta(minutes=minute)
                base_hr = 72 if sex == "M" else 76
                hr = base_hr + rng.normal(0, 8) + (10 if 600 <= minute <= 660 else 0)
                all_hr.append({
                    "participant_id": pid, "timestamp": ts,
                    "heart_rate_bpm": round(float(np.clip(hr, 45, 180)), 0),
                })

            # Steps: 15-min buckets
            for minute in range(0, 1440, 15):
                ts = day_start + pd.Timedelta(minutes=minute)
                hour = minute / 60
                if 6 <= hour <= 22:
                    steps = max(0, int(rng.poisson(50)))
                else:
                    steps = 0
                all_steps.append({
                    "participant_id": pid, "timestamp": ts, "steps": steps,
                })

            # Sleep: one entry per day
            sleep_start = day_start + pd.Timedelta(hours=23)
            sleep_end   = day_start + pd.Timedelta(hours=30)  # next day 6 AM
            all_sleep.append({
                "participant_id": pid,
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "duration_minutes": float(rng.normal(420, 45)),
                "efficiency": round(float(rng.normal(85, 8)), 0),
            })

            # Meals: 3 meals + 1 snack per day
            for meal_time, meal_type in [(8, "breakfast"), (13, "lunch"), (16, "snack"), (20, "dinner")]:
                ts = day_start + pd.Timedelta(hours=meal_time, minutes=int(rng.integers(-15, 15)))
                carbs = round(float(rng.normal(50, 20)), 0) if meal_type != "snack" else round(float(rng.normal(20, 10)), 0)
                all_meals.append({
                    "participant_id": pid, "timestamp": ts,
                    "meal_type": meal_type,
                    "calories_kcal": round(float(carbs * 4 + rng.normal(200, 80)), 0),
                    "carbs_g": max(5, carbs),
                    "protein_g": round(float(rng.normal(15, 5)), 0),
                    "fat_g": round(float(rng.normal(12, 5)), 0),
                    "fibre_g": round(float(rng.normal(4, 2)), 0),
                })

    data = {
        "cgm":          pd.DataFrame(all_cgm),
        "heart_rate":   pd.DataFrame(all_hr),
        "steps":        pd.DataFrame(all_steps),
        "sleep":        pd.DataFrame(all_sleep),
        "meals":        pd.DataFrame(all_meals),
        "demographics": pd.DataFrame(demo_rows),
        "labs":         pd.DataFrame(lab_rows),
    }

    # Convert timestamps
    for key in ("cgm", "heart_rate", "steps", "meals"):
        data[key]["timestamp"] = pd.to_datetime(data[key]["timestamp"])
    for col in ("sleep_start", "sleep_end"):
        data["sleep"][col] = pd.to_datetime(data["sleep"][col])

    pids = sorted(data["demographics"]["participant_id"].unique().tolist())
    logger.info("Demo data generated: %d participants", len(pids))

    return data, pids


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MadhuTwin — Build the complete dataset with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--num-patients", type=int, default=100,
                        help="Number of synthetic EHR patients (default: 100)")
    parser.add_argument("--augmentation-fraction", type=float, default=0.30,
                        help="Fraction of meals to augment (default: 0.30)")
    parser.add_argument("--stride", type=int, default=15,
                        help="Window stride in minutes (default: 15)")
    parser.add_argument("--skip-cgmacros", action="store_true",
                        help="Skip CGMacros loading (use demo data)")

    args = parser.parse_args()

    main(
        seed=args.seed,
        num_synthetic_patients=args.num_patients,
        augmentation_fraction=args.augmentation_fraction,
        skip_cgmacros=args.skip_cgmacros,
        stride_minutes=args.stride,
    )
