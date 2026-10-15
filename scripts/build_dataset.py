#!/usr/bin/env python3
"""
MadhuTwin — End-to-End Data Pipeline  (scripts/build_dataset.py)
================================================================
Reproducible dataset build with ONE command:

    python scripts/build_dataset.py

Steps:
  1. Load & validate CGMacros data  (data/raw/cgmacros/)
  2. Generate synthetic EHR cohort   (Synthea or Python fallback)
  3. Pair EHR ↔ wearable             (nearest-neighbour)
  4. Build Indian meal engine         (validate food DB)
  5. Augment meals                    (Indian meal substitution)
  6. Build feature windows            (4h history → 2h horizon)
  7. Create participant-level splits  (train/val/test)
  8. Save everything                  (interface contracts honoured)

Outputs (interface contracts → Person 1 & Person 2):
  data/processed/splits.json           →  Person 2
  data/processed/windows.pkl           →  Person 2
  data/indian_food_db.csv              →  Person 1  (already exists)
  data/processed/patient_profiles/     →  Person 1
  data/pairing_log.csv                 →  Audit trail

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

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")

# ── Default paths ─────────────────────────────────────────────────────────────
CGMACROS_DIR      = PROJECT_ROOT / "data" / "raw" / "cgmacros"
SYNTHEA_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "synthea"
FOOD_DB_PATH       = PROJECT_ROOT / "data" / "indian_food_db.csv"
PROCESSED_DIR      = PROJECT_ROOT / "data" / "processed"
PAIRING_LOG_PATH   = PROJECT_ROOT / "data" / "pairing_log.csv"
PROFILES_DIR       = PROCESSED_DIR / "patient_profiles"
SPLITS_PATH        = PROCESSED_DIR / "splits.json"
WINDOWS_PATH       = PROCESSED_DIR / "windows.pkl"


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
    if CGMACROS_DIR.exists() and not skip_cgmacros:
        cgmacros_data = load_cgmacros(CGMACROS_DIR)
        cgm_participants = get_participant_ids(cgmacros_data)

        if not cgm_participants:
            logger.warning("No complete participants found. Using demo mode.")
            cgmacros_data, cgm_participants = _create_demo_cgmacros(seed)
        else:
            # Resample CGM to uniform 5-min grid
            cgmacros_data["cgm"] = resample_cgm(cgmacros_data["cgm"])
    else:
        logger.info("CGMacros not found at %s — generating demo data.", CGMACROS_DIR)
        cgmacros_data, cgm_participants = _create_demo_cgmacros(seed)

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
    original_meals = paired_data.get("meals", cgmacros_data["meals"])

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

    # Update the meals in paired_data for windowing
    if "meals" in paired_data:
        paired_data["meals"] = augmented_meals
    cgmacros_data["meals"] = augmented_meals

    # ── Step 6: Build feature windows ─────────────────────────────────────
    logger.info("\n[Step 6/8] Building feature windows (4h history → 2h horizon)…")

    # Use paired_patient_id if available, otherwise participant_id
    cgm_for_windows = paired_data.get("cgm", cgmacros_data["cgm"])
    hr_for_windows  = paired_data.get("heart_rate", cgmacros_data["heart_rate"])
    steps_for_windows = paired_data.get("steps", cgmacros_data["steps"])
    sleep_for_windows = paired_data.get("sleep", cgmacros_data["sleep"])
    meals_for_windows = paired_data.get("meals", augmented_meals)

    # Determine participant IDs to window
    window_pids = sorted(cgm_for_windows["participant_id"].unique())

    windows = build_windows(
        cgm_df=cgm_for_windows,
        hr_df=hr_for_windows,
        steps_df=steps_for_windows,
        sleep_df=sleep_for_windows,
        meals_df=meals_for_windows,
        ehr_df=ehr_df,
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

    # ── Step 8: Save everything ───────────────────────────────────────────
    logger.info("\n[Step 8/8] Saving outputs…")
    save_windows(windows, WINDOWS_PATH)

    # Also save a flat CSV summary for quick inspection
    windows_summary = windows_to_dataframe(windows)
    windows_summary.to_csv(PROCESSED_DIR / "windows_summary.csv", index=False)
    logger.info("  Windows summary: %s", PROCESSED_DIR / "windows_summary.csv")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 70)
    logger.info("Outputs:")
    logger.info("  %-45s  (→ Person 2)", SPLITS_PATH)
    logger.info("  %-45s  (→ Person 2)", WINDOWS_PATH)
    logger.info("  %-45s  (→ Person 1)", PROFILES_DIR)
    logger.info("  %-45s  (→ Person 1)", FOOD_DB_PATH)
    logger.info("  %-45s  (audit)", PAIRING_LOG_PATH)
    logger.info("  %-45s  (report)", PROCESSED_DIR / "augmentation_distribution.csv")
    logger.info("  %-45s  (report)", PROCESSED_DIR / "sensitivity_test.csv")
    logger.info("=" * 70)


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
