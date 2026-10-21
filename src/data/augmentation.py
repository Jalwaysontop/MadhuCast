"""
Indian Meal Augmentation
========================
Substitutes Indian meals into CGMacros meal slots to create augmented
training data that bridges US-sourced wearable data with Indian dietary
patterns.

Design (from spec Section 6.1):
  - Replace a fraction of CGMacros meals with Indian meals from the
    nutrient mapping, keeping carbohydrate load distribution realistic.
  - Label EVERY augmented sample as synthetic.
  - Run robustness experiments: results with vs. without augmentation.

Key constraint:
  The paired EHR and Indian-meal responses are synthetic.  This is stated
  plainly in the dataset card and every evaluation table.

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.meal_engine import IndianFoodDB, compute_nutrient_vector, classify_gl

logger = logging.getLogger(__name__)

# ── Augmentation parameters ──────────────────────────────────────────────────
DEFAULT_AUGMENTATION_FRACTION = 0.30   # Replace 30% of meals
DEFAULT_SEED = 42

# Meal-type → typical Indian meal patterns (food_id lists)
TYPICAL_INDIAN_MEALS = {
    "breakfast": [
        [("BF003", 2), ("D010", 1)],                            # 2 idlis + sambar
        [("BF001", 1)],                                          # poha
        [("BF002", 1)],                                          # upma
        [("B020", 2), ("AC003", 1)],                             # 2 dosas + chutney
        [("B001", 2), ("DA002", 1)],                             # 2 rotis + curd
        [("BF009", 1)],                                          # misal pav
        [("BF007", 1)],                                          # chole bhature
        [("BF008", 1)],                                          # paratha + curd
        [("BF015", 1), ("FR001", 1)],                            # oats + banana
        [("B023", 1), ("AC003", 1)],                             # uttapam + chutney
    ],
    "lunch": [
        [("R001", 1), ("D001", 1), ("V001", 1), ("DA002", 1)],  # rice + dal + sabzi + curd
        [("B001", 3), ("D007", 1), ("DA004", 1)],                # 3 rotis + chole + raita
        [("R001", 1), ("D010", 1), ("V012", 1), ("AC008", 1)],  # rice + sambar + beans + papad
        [("R006", 1), ("DA003", 1)],                             # veg pulao + raita
        [("B001", 2), ("V004", 1), ("D002", 1)],                 # 2 rotis + palak paneer + dal
        [("R001", 1), ("NV001", 1), ("DA002", 1)],              # rice + chicken curry + curd
        [("B001", 2), ("D009", 1), ("V002", 1)],                 # rotis + dal makhani + bhindi
        [("R001", 1), ("NV007", 1), ("V018", 1)],               # rice + fish curry + aviyal
        [("R011", 1), ("AC008", 1)],                             # khichdi + papad
        [("B001", 2), ("V006", 1), ("D003", 1)],                 # rotis + aloo matar + masoor dal
    ],
    "dinner": [
        [("B001", 2), ("D001", 1), ("V003", 1)],                 # rotis + dal + baingan
        [("R001", 1), ("D002", 1), ("V007", 1)],                 # rice + moong dal + mixed veg
        [("B001", 3), ("NV005", 1)],                              # 3 rotis + egg curry
        [("R001", 1), ("D008", 1), ("V008", 1)],                 # rice + dal tadka + lauki
        [("B001", 2), ("V005", 1), ("D011", 1)],                 # rotis + matar paneer + rasam
        [("B015", 2), ("V014", 1)],                               # bajra roti + sarson saag
        [("B001", 2), ("NV009", 1), ("DA004", 1)],               # rotis + mutton + raita
        [("R001", 1), ("D013", 1), ("V002", 1)],                 # rice + kadhi + bhindi
        [("R005", 1)],                                            # curd rice (light dinner)
        [("B001", 2), ("D006", 1), ("V011", 1)],                 # rotis + rajma + cabbage
    ],
    "snack": [
        [("SN001", 1), ("BV001", 1)],                            # samosa + chai
        [("SN010", 2), ("BV002", 1)],                            # dhokla + chai
        [("FR001", 1)],                                           # banana
        [("SN004", 1)],                                           # bhel puri
        [("SN002", 1), ("BV001", 1)],                            # pakora + chai
        [("FR004", 1)],                                           # guava
        [("SN007", 1), ("BV001", 1)],                            # aloo tikki + chai
        [("FR005", 1)],                                           # apple
        [("BV004", 1)],                                           # sweet lassi
        [("SN005", 1)],                                           # pani puri
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
#  Meal matching (macronutrient-profile based)
# ═════════════════════════════════════════════════════════════════════════════

def find_matching_indian_meal(
    original_carbs: float,
    original_calories: float,
    meal_type: str,
    food_db: IndianFoodDB,
    rng: np.random.Generator,
    carb_tolerance: float = 0.40,
) -> Optional[Dict[str, Any]]:
    """
    Find an Indian meal that approximates the macronutrient profile of the
    original meal.

    Parameters
    ----------
    original_carbs    : Carbs (g) of the original meal.
    original_calories : Calories of the original meal.
    meal_type         : 'breakfast', 'lunch', 'dinner', or 'snack'.
    food_db           : IndianFoodDB instance.
    rng               : Numpy random generator.
    carb_tolerance    : Acceptable relative carb difference (default ±40%).

    Returns
    -------
    dict with the Indian meal composition, or None if no match found.
    """
    meal_type_key = _normalize_meal_type(meal_type)
    candidates = TYPICAL_INDIAN_MEALS.get(meal_type_key, TYPICAL_INDIAN_MEALS["lunch"])

    # Score each candidate by carb proximity
    scored: List[Tuple[float, int, List]] = []
    for i, meal_items in enumerate(candidates):
        total_carbs = 0.0
        total_cal = 0.0
        valid = True

        for food_id, portions in meal_items:
            item = food_db.lookup(food_id)
            if item is None:
                valid = False
                break
            nvec = compute_nutrient_vector(item, portions)
            total_carbs += nvec["carbs_g"]
            total_cal += nvec["calories_kcal"]

        if not valid:
            continue

        # Relative carb difference
        if original_carbs > 0:
            carb_diff = abs(total_carbs - original_carbs) / original_carbs
        else:
            carb_diff = 1.0

        if carb_diff <= carb_tolerance:
            scored.append((carb_diff, i, meal_items))

    if not scored:
        # Fallback: pick a random meal from the category, scale portions
        fallback_idx = rng.integers(0, len(candidates))
        return _build_augmented_meal(
            candidates[fallback_idx], food_db,
            target_carbs=original_carbs,
        )

    # Pick from top-3 closest by carb match (with randomness)
    scored.sort(key=lambda x: x[0])
    top_n = min(3, len(scored))
    choice = scored[rng.integers(0, top_n)]

    return _build_augmented_meal(choice[2], food_db)


def _build_augmented_meal(
    meal_items: List[Tuple[str, float]],
    food_db: IndianFoodDB,
    target_carbs: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a full augmented meal dict from food item list.

    If target_carbs is provided, scale all portions proportionally.
    """
    total_carbs = 0.0
    total_protein = 0.0
    total_fat = 0.0
    total_fibre = 0.0
    total_cal = 0.0
    total_gl = 0.0
    items_detail = []

    for food_id, portions in meal_items:
        item = food_db.lookup(food_id)
        if item is None:
            continue
        nvec = compute_nutrient_vector(item, portions)
        total_carbs   += nvec["carbs_g"]
        total_protein += nvec["protein_g"]
        total_fat     += nvec["fat_g"]
        total_fibre   += nvec["fibre_g"]
        total_cal     += nvec["calories_kcal"]
        total_gl      += nvec["glycemic_load"]
        items_detail.append({
            "food_id": food_id,
            "food_name": item["food_name"],
            "portions": portions,
            **nvec,
        })

    # Scale if target carbs provided
    scale = 1.0
    if target_carbs and total_carbs > 0:
        scale = target_carbs / total_carbs

    return {
        "carbs_g":        round(total_carbs * scale, 1),
        "protein_g":      round(total_protein * scale, 1),
        "fat_g":          round(total_fat * scale, 1),
        "fibre_g":        round(total_fibre * scale, 1),
        "calories_kcal":  round(total_cal * scale, 0),
        "glycemic_load":  round(total_gl * scale, 1),
        "items_detail":   items_detail,
        "is_indian_meal": True,
        "is_augmented":   True,
        "augmentation_source": "indian_meal_substitution",
    }


def _normalize_meal_type(meal_type: str) -> str:
    """Map various meal type strings to our canonical categories."""
    mt = str(meal_type).strip().lower()
    if any(k in mt for k in ["break", "morning"]):
        return "breakfast"
    elif any(k in mt for k in ["lunch", "midday", "afternoon"]):
        return "lunch"
    elif any(k in mt for k in ["dinner", "evening", "supper", "night"]):
        return "dinner"
    elif any(k in mt for k in ["snack", "tea", "between", "misc"]):
        return "snack"
    return "lunch"  # Default


# ═════════════════════════════════════════════════════════════════════════════
#  Main augmentation pipeline
# ═════════════════════════════════════════════════════════════════════════════

def augment_meals(
    meals_df: pd.DataFrame,
    food_db: IndianFoodDB,
    augmentation_fraction: float = DEFAULT_AUGMENTATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Augment a meal DataFrame by replacing a fraction of meals with
    Indian equivalents matched by macronutrient profile.

    Parameters
    ----------
    meals_df : Original CGMacros meal DataFrame with columns:
               participant_id, timestamp, meal_type, calories_kcal,
               carbs_g, protein_g, fat_g, fibre_g.
    food_db  : IndianFoodDB instance.
    augmentation_fraction : Fraction of meals to replace (default 0.30).
    seed : Random seed.

    Returns
    -------
    DataFrame with same structure plus additional columns:
        is_augmented           : bool
        augmentation_source    : str or NaN
        original_carbs_g       : float (original value before substitution)
        original_calories_kcal : float
        indian_meal_items      : str (JSON of item details) or NaN
    """
    rng = np.random.default_rng(seed)
    result = meals_df.copy()

    # Add augmentation tracking columns
    result["is_augmented"]        = False
    result["augmentation_source"] = pd.NA
    result["original_carbs_g"]    = result["carbs_g"]
    result["original_calories_kcal"] = result.get("calories_kcal", np.nan)
    result["indian_meal_items"]   = pd.NA
    result["glycemic_load"]       = np.nan

    n_meals = len(result)
    n_to_augment = int(n_meals * augmentation_fraction)

    # Select random meals to augment (stratified by meal_type if possible)
    augment_indices = rng.choice(n_meals, size=n_to_augment, replace=False)
    augment_indices.sort()

    n_success = 0
    for idx in augment_indices:
        row = result.iloc[idx]
        original_carbs = float(row.get("carbs_g", 0))
        original_cal   = float(row.get("calories_kcal", 0))
        meal_type      = str(row.get("meal_type", "lunch"))

        indian_meal = find_matching_indian_meal(
            original_carbs, original_cal, meal_type, food_db, rng
        )

        if indian_meal is None:
            continue

        # Substitute the macros
        result.iloc[idx, result.columns.get_loc("carbs_g")]       = indian_meal["carbs_g"]
        result.iloc[idx, result.columns.get_loc("protein_g")]     = indian_meal["protein_g"]
        result.iloc[idx, result.columns.get_loc("fat_g")]         = indian_meal["fat_g"]
        result.iloc[idx, result.columns.get_loc("fibre_g")]       = indian_meal["fibre_g"]
        if "calories_kcal" in result.columns:
            result.iloc[idx, result.columns.get_loc("calories_kcal")] = indian_meal["calories_kcal"]
        result.iloc[idx, result.columns.get_loc("glycemic_load")] = indian_meal["glycemic_load"]
        result.iloc[idx, result.columns.get_loc("is_augmented")]  = True
        result.iloc[idx, result.columns.get_loc("augmentation_source")] = "indian_meal_substitution"

        # Store item details as string
        import json
        items_str = json.dumps([
            {"food_id": d["food_id"], "food_name": d["food_name"], "portions": d["portions"]}
            for d in indian_meal["items_detail"]
        ])
        result.iloc[idx, result.columns.get_loc("indian_meal_items")] = items_str

        n_success += 1

    logger.info(
        "Augmented %d/%d meals (%.1f%%) with Indian meal substitutions. "
        "Target was %d (%.0f%%).",
        n_success, n_meals, 100 * n_success / n_meals,
        n_to_augment, 100 * augmentation_fraction,
    )

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Augment feature windows  (post-windowing augmentation)
# ═════════════════════════════════════════════════════════════════════════════

def mark_augmented_windows(
    windows: List[Dict[str, Any]],
    augmented_meals_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Mark windows that include augmented meals.

    A window is marked as augmented if any meal within its history window
    was substituted with an Indian meal.
    """
    if augmented_meals_df.empty or "is_augmented" not in augmented_meals_df.columns:
        return windows

    aug_times = set()
    aug_meals = augmented_meals_df[augmented_meals_df["is_augmented"] == True]
    for _, row in aug_meals.iterrows():
        pid = row["participant_id"]
        ts  = row["timestamp"]
        aug_times.add((pid, ts))

    n_marked = 0
    for w in windows:
        pid = w["participant_id"]
        w_start = w["window_start"]
        w_end   = w["window_end"]

        # Check if any augmented meal falls within this window
        for (meal_pid, meal_ts) in aug_times:
            if meal_pid == pid and w_start <= meal_ts <= w_end:
                w["is_augmented"] = True
                n_marked += 1
                break

    logger.info("Marked %d/%d windows as augmented", n_marked, len(windows))
    return windows


# ═════════════════════════════════════════════════════════════════════════════
#  Distribution analysis  (for the robustness experiment)
# ═════════════════════════════════════════════════════════════════════════════

def compare_distributions(
    original_meals: pd.DataFrame,
    augmented_meals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare macronutrient distributions before and after augmentation.

    Returns a summary DataFrame for the dataset card / evaluation report.
    """
    metrics = ["carbs_g", "protein_g", "fat_g", "fibre_g", "calories_kcal"]
    rows = []

    for m in metrics:
        if m not in original_meals.columns or m not in augmented_meals.columns:
            continue
        orig_vals = original_meals[m].dropna()
        aug_vals  = augmented_meals[m].dropna()

        rows.append({
            "metric": m,
            "original_mean":   round(orig_vals.mean(), 1),
            "original_std":    round(orig_vals.std(), 1),
            "augmented_mean":  round(aug_vals.mean(), 1),
            "augmented_std":   round(aug_vals.std(), 1),
            "mean_shift_pct":  round(100 * (aug_vals.mean() - orig_vals.mean()) / orig_vals.mean(), 1) if orig_vals.mean() != 0 else 0,
        })

    return pd.DataFrame(rows)
