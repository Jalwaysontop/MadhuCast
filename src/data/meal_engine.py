"""
Indian Meal Glycemic Load Engine  (Feature F2)
===============================================
Turns a meal entry like "2 rotis, 1 katori dal, 1 katori rice, sabzi"
into an estimated nutrient vector (carbs, protein, fat, fibre) and
glycemic load (GL).

Core design:
  - A curated lookup of ~200 common Indian foods with Indian portion
    units (katori, piece, glass, tablespoon, etc.).
  - GL = (GI × available_carbs_per_serving) / 100
  - The model consumes the nutrient vector and GL, NOT food names,
    so it trains on CGMacros (macronutrients) and accepts Indian foods
    at inference through this mapping layer.
  - The mapping is small, open, rule-based, and human-auditable.

Differs from:  Fitterfly (37 k+ proprietary items), January AI
               (millions of items).  We can't compete on size —
               our value is transparency and auditability.

Evidence to show:
  - Mapping table published in the repo (data/indian_food_db.csv).
  - Sensitivity test: model output changes monotonically as carb load rises.

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FOOD_DB_PATH = Path("data/indian_food_db.csv")

# ── Preparation modifiers ────────────────────────────────────────────────────
# Literature-based GI multipliers for common Indian cooking methods.
# These are flagged as assumptions in the UI.
# Sources: Foster-Powell et al. (2002), Atkinson et al. (2008).
PREPARATION_GI_MODIFIERS = {
    "raw":           1.00,
    "boiled":        1.00,
    "steamed":       0.95,
    "slow_cooked":   0.90,
    "cooked":        1.00,
    "tempered":      0.98,   # tadka — brief frying
    "mixed":         0.95,
    "griddle":       1.00,
    "dry_roasted":   0.95,
    "puffed":        1.05,
    "shallow_fried": 0.98,
    "deep_fried":    1.05,
    "tandoor":       1.00,
    "fried_soaked":  0.90,
    "assembled":     1.00,
    "brewed":        1.00,
    "blended":       1.00,
    "churned":       1.00,
    "pressed":       1.05,
    "fermented":     0.85,
    "ground":        1.00,
    "clarified":     1.00,
    "soaked":        0.90,
    "cold":          1.00,
    "layered":       1.00,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Food Database
# ═════════════════════════════════════════════════════════════════════════════

class IndianFoodDB:
    """
    In-memory Indian food database loaded from CSV.

    Attributes
    ----------
    df : pd.DataFrame  — the raw food database
    """

    def __init__(self, db_path: Path | str = DEFAULT_FOOD_DB_PATH):
        self.db_path = Path(db_path)
        self.df = self._load()
        self._index = self._build_index()

    def _load(self) -> pd.DataFrame:
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Indian food database not found at {self.db_path}. "
                "Ensure data/indian_food_db.csv exists."
            )
        df = pd.read_csv(self.db_path)
        logger.info("Loaded Indian food DB: %d items from %s", len(df), self.db_path)
        return df

    def _build_index(self) -> Dict[str, int]:
        """Build food_id → row-index lookup."""
        idx: Dict[str, int] = {}
        for i, row in self.df.iterrows():
            idx[str(row["food_id"]).strip().upper()] = i
            # Also index by lowercase food_name for convenience
            idx[str(row["food_name"]).strip().lower()] = i
        return idx

    def lookup(self, query: str) -> Optional[pd.Series]:
        """
        Look up a food item by food_id or food_name (case-insensitive).

        Returns a Series with all columns, or None if not found.
        """
        key = query.strip()
        # Try exact ID match
        row_idx = self._index.get(key.upper())
        if row_idx is None:
            # Try name match
            row_idx = self._index.get(key.lower())
        if row_idx is None:
            # Try fuzzy substring match
            matches = self.df[
                self.df["food_name"].str.lower().str.contains(key.lower(), na=False)
            ]
            if len(matches) == 1:
                return matches.iloc[0]
            elif len(matches) > 1:
                logger.debug("Ambiguous lookup '%s' → %d matches", key, len(matches))
                return matches.iloc[0]  # Return first match
            return None
        return self.df.iloc[row_idx]

    def search(
        self,
        query: str = "",
        category: Optional[str] = None,
        vegetarian_only: bool = False,
        max_results: int = 20,
    ) -> pd.DataFrame:
        """Search the food DB with optional filters."""
        mask = pd.Series(True, index=self.df.index)
        if query:
            mask &= self.df["food_name"].str.lower().str.contains(query.lower(), na=False)
        if category:
            mask &= self.df["category"].str.lower() == category.lower()
        if vegetarian_only:
            mask &= self.df["is_vegetarian"].astype(bool)
        return self.df[mask].head(max_results)

    def get_categories(self) -> List[str]:
        """Return sorted list of food categories."""
        return sorted(self.df["category"].unique().tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"IndianFoodDB({len(self)} items, {len(self.get_categories())} categories)"


# ═════════════════════════════════════════════════════════════════════════════
#  Nutrient vector & GL computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_nutrient_vector(
    food_item: pd.Series,
    num_portions: float = 1.0,
    preparation: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute the nutrient vector for a food item at a given portion count.

    Parameters
    ----------
    food_item    : A row from the food DB (pd.Series).
    num_portions : Number of standard portions (e.g. 2 rotis = 2.0).
    preparation  : Override preparation modifier (e.g. 'deep_fried').
                   If None, uses the value from the DB.

    Returns
    -------
    dict with keys: carbs_g, protein_g, fat_g, fibre_g, calories_kcal,
                    glycemic_index, glycemic_load, portion_grams_total
    """
    prep = preparation or str(food_item.get("preparation_modifier", "cooked"))
    gi_modifier = PREPARATION_GI_MODIFIERS.get(prep, 1.0)

    base_gi = float(food_item.get("glycemic_index", 50))
    adjusted_gi = base_gi * gi_modifier

    carbs    = float(food_item.get("carbs_g", 0)) * num_portions
    protein  = float(food_item.get("protein_g", 0)) * num_portions
    fat      = float(food_item.get("fat_g", 0)) * num_portions
    fibre    = float(food_item.get("fibre_g", 0)) * num_portions
    calories = float(food_item.get("calories_kcal", 0)) * num_portions
    portion_g = float(food_item.get("portion_grams", 0)) * num_portions

    # Available carbs (carbs minus fibre, but not below zero)
    available_carbs = max(0, carbs - fibre)

    # GL = (GI × available_carbs) / 100
    gl = round((adjusted_gi * available_carbs) / 100, 1)

    return {
        "carbs_g": round(carbs, 1),
        "protein_g": round(protein, 1),
        "fat_g": round(fat, 1),
        "fibre_g": round(fibre, 1),
        "calories_kcal": round(calories, 0),
        "glycemic_index": round(adjusted_gi, 0),
        "glycemic_load": gl,
        "portion_grams_total": round(portion_g, 0),
    }


def compute_meal_vector(
    items: List[Tuple[str, float]],
    food_db: IndianFoodDB,
    preparation_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compute the combined nutrient vector for a full meal.

    Parameters
    ----------
    items : List of (food_id_or_name, num_portions).
            Example: [("White Rice (Steamed)", 1), ("Toor Dal", 1),
                      ("Bhindi Masala", 1), ("Wheat Roti", 2)]
    food_db : IndianFoodDB instance.
    preparation_overrides : Optional dict mapping food_id → preparation.

    Returns
    -------
    dict with keys:
        total_carbs_g, total_protein_g, total_fat_g, total_fibre_g,
        total_calories_kcal, total_glycemic_load, weighted_glycemic_index,
        item_count, items_detail (list of per-item dicts),
        is_valid (bool), missing_items (list of items not found)
    """
    preparation_overrides = preparation_overrides or {}
    total = {
        "total_carbs_g": 0.0,
        "total_protein_g": 0.0,
        "total_fat_g": 0.0,
        "total_fibre_g": 0.0,
        "total_calories_kcal": 0.0,
        "total_glycemic_load": 0.0,
    }
    items_detail: List[Dict] = []
    missing_items: List[str] = []
    weighted_gi_num = 0.0
    weighted_gi_den = 0.0

    for food_query, num_portions in items:
        food_item = food_db.lookup(food_query)
        if food_item is None:
            missing_items.append(food_query)
            logger.warning("Food item not found in DB: '%s'", food_query)
            continue

        prep = preparation_overrides.get(str(food_item.get("food_id", "")))
        nvec = compute_nutrient_vector(food_item, num_portions, prep)

        total["total_carbs_g"]       += nvec["carbs_g"]
        total["total_protein_g"]     += nvec["protein_g"]
        total["total_fat_g"]         += nvec["fat_g"]
        total["total_fibre_g"]       += nvec["fibre_g"]
        total["total_calories_kcal"] += nvec["calories_kcal"]
        total["total_glycemic_load"] += nvec["glycemic_load"]

        # For weighted GI: weight by available carbs
        avail_carbs = max(0, nvec["carbs_g"] - nvec["fibre_g"])
        weighted_gi_num += nvec["glycemic_index"] * avail_carbs
        weighted_gi_den += avail_carbs

        items_detail.append({
            "food_name": food_item["food_name"],
            "food_id": food_item["food_id"],
            "portions": num_portions,
            **nvec,
        })

    # Weighted average GI of the meal
    weighted_gi = round(weighted_gi_num / weighted_gi_den, 0) if weighted_gi_den > 0 else 0.0

    # Round totals
    for k in total:
        total[k] = round(total[k], 1)

    return {
        **total,
        "weighted_glycemic_index": weighted_gi,
        "item_count": len(items_detail),
        "items_detail": items_detail,
        "is_valid": len(missing_items) == 0,
        "missing_items": missing_items,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Portion conversion helpers
# ═════════════════════════════════════════════════════════════════════════════

# Standard Indian portion units → approximate grams
PORTION_UNIT_GRAMS = {
    "katori":       150,   # standard serving bowl
    "piece":         50,   # varies by food — DB has per-item values
    "glass":        200,
    "cup":          150,
    "tablespoon":    15,
    "teaspoon":       5,
    "serving":      150,
    "plate":        200,
    "bowl":         200,
    "ladle":        100,
    "slice":         30,
}


def portion_to_grams(
    amount: float,
    unit: str,
    food_item: Optional[pd.Series] = None,
) -> float:
    """
    Convert an Indian portion description to grams.

    If ``food_item`` is provided, uses its specific ``portion_grams``.
    Otherwise falls back to generic unit estimates.
    """
    if food_item is not None:
        base_grams = float(food_item.get("portion_grams", 0))
        if base_grams > 0:
            return round(amount * base_grams, 0)

    unit_lower = unit.strip().lower()
    base = PORTION_UNIT_GRAMS.get(unit_lower, 100)
    return round(amount * base, 0)


# ═════════════════════════════════════════════════════════════════════════════
#  GL classification  (for downstream alerting thresholds)
# ═════════════════════════════════════════════════════════════════════════════

def classify_gl(gl: float) -> str:
    """
    Classify a meal's glycemic load.

    Low:    GL ≤ 10
    Medium: 11 ≤ GL ≤ 19
    High:   GL ≥ 20
    """
    if gl <= 10:
        return "low"
    elif gl <= 19:
        return "medium"
    else:
        return "high"


def classify_meal_risk(meal_vector: Dict[str, Any]) -> str:
    """
    Simple risk classification for a meal based on GL and macros.

    Returns 'low', 'moderate', or 'high'.
    """
    gl = meal_vector.get("total_glycemic_load", 0)
    carbs = meal_vector.get("total_carbs_g", 0)
    fibre = meal_vector.get("total_fibre_g", 0)

    # Fibre-to-carb ratio provides some mitigation
    fibre_ratio = fibre / carbs if carbs > 0 else 0

    if gl >= 30 or (gl >= 20 and fibre_ratio < 0.1):
        return "high"
    elif gl >= 15 or (gl >= 10 and fibre_ratio < 0.05):
        return "moderate"
    else:
        return "low"


# ═════════════════════════════════════════════════════════════════════════════
#  Sensitivity test  (spec requires monotonicity check)
# ═════════════════════════════════════════════════════════════════════════════

def run_sensitivity_test(food_db: IndianFoodDB) -> pd.DataFrame:
    """
    Sensitivity test: verify GL increases monotonically with portion size.

    Picks representative foods from each category and checks that doubling
    the portion at least doubles the GL (within rounding tolerance).

    Returns a DataFrame with test results.
    """
    results: List[Dict] = []
    test_foods = ["White Rice (Steamed)", "Wheat Roti", "Toor Dal",
                  "Banana", "Samosa", "Gulab Jamun", "Idli", "Naan"]

    for food_name in test_foods:
        item = food_db.lookup(food_name)
        if item is None:
            continue

        gl_1 = compute_nutrient_vector(item, 1.0)["glycemic_load"]
        gl_2 = compute_nutrient_vector(item, 2.0)["glycemic_load"]
        gl_3 = compute_nutrient_vector(item, 3.0)["glycemic_load"]

        monotonic = (gl_1 <= gl_2 <= gl_3)
        proportional = abs(gl_2 - 2 * gl_1) < 0.5  # Within rounding

        results.append({
            "food_name": food_name,
            "gl_1_portion": gl_1,
            "gl_2_portions": gl_2,
            "gl_3_portions": gl_3,
            "monotonic": monotonic,
            "proportional": proportional,
        })

    df = pd.DataFrame(results)
    n_pass = df["monotonic"].sum()
    logger.info("Sensitivity test: %d/%d foods pass monotonicity", n_pass, len(df))
    return df
