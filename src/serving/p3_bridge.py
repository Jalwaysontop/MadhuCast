"""P1 bridge onto Person 3's meal engine (R3.4 / R3.5), without editing P3's files.

P3 delivered ``src/data/meal_engine.py`` (``IndianFoodDB`` + ``compute_nutrient_vector``) and
``data/indian_food_db.csv`` in their own schema. This module maps them onto the contract:

* ``food_db()``     -> FOOD_DB_COLUMNS, one row per food, keyed by the lower-cased P3 ``food_name``.
* ``resolve_meal()`` -> a validated ``Meal``; nutrition and GL come from P3's ``compute_nutrient_vector``
  (GI x available carbs / 100, with P3's preparation GI multiplier), so the numbers stay P3's.

Assumptions surfaced as ``assumption_flag`` (shown as ⚠ in the UI): any preparation GI multiplier
other than x1, and preparation methods P3's table has no multiplier for (P3 silently uses x1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.contracts import FOOD_DB_COLUMNS, ContractError, Meal, Provenance

ROOT = Path(__file__).resolve().parents[2]
FOOD_DB_PATH = ROOT / "data" / "indian_food_db.csv"
SOURCE = "P3 curated Indian food table (IFCT/INDB-derived values)"
LICENCE = "unverified: IFCT/INDB source licence not yet checked (see data/dataset_card.md)"


def _engine():
    try:
        from src.data import meal_engine
    except ImportError as e:  # pragma: no cover - exercised only when P3's module is broken
        raise ContractError("P3:meal_engine", f"src/data/meal_engine.py not importable: {e}") from e
    return meal_engine


def _flag(prep: str, modifiers: dict[str, float]) -> str:
    if prep not in modifiers:
        return f"no GI multiplier for '{prep}' preparation, assumed x1"
    m = modifiers[prep]
    return "" if m == 1.0 else f"GI x{m:g} for {prep.replace('_', ' ')} preparation (literature multiplier)"


@lru_cache(maxsize=4)
def _table(path: str) -> pd.DataFrame:
    eng = _engine()
    if not Path(path).exists():
        raise ContractError("P3:indian_food_db", f"{path} not found")
    try:
        db = eng.IndianFoodDB(path)
    except Exception as e:
        raise ContractError("P3:indian_food_db", f"IndianFoodDB could not load {Path(path).name}: {e}") from e
    rows = []
    for _, r in db.df.iterrows():
        unit = eng.compute_nutrient_vector(r, 1.0)
        prep = str(r.get("preparation_modifier", "cooked"))
        rows.append(
            {
                "food": str(r["food_name"]).strip().lower(),
                "portion_unit": str(r["portion_unit"]).strip().lower(),
                "grams_per_unit": float(r["portion_grams"]),
                "carbs_g": unit["carbs_g"],
                "protein_g": unit["protein_g"],
                "fat_g": unit["fat_g"],
                "fibre_g": unit["fibre_g"],
                "gi": unit["glycemic_index"],
                "gl_per_unit": unit["glycemic_load"],
                "source": SOURCE,
                "licence": LICENCE,
                "assumption_flag": _flag(prep, eng.PREPARATION_GI_MODIFIERS),
                "food_id": str(r["food_id"]),
                "category": str(r["category"]),
                "_prep": prep,
            }
        )
    df = pd.DataFrame(rows)
    dup = df["food"][df["food"].duplicated()].tolist()
    if dup:
        raise ContractError("P3:indian_food_db", f"duplicate food names after lower-casing: {dup}")
    return df


def food_db(path: Path = FOOD_DB_PATH) -> pd.DataFrame:
    """Contract view of P3's table (FOOD_DB_COLUMNS first, then food_id and category)."""
    df = _table(str(path))
    return df[[*FOOD_DB_COLUMNS, "food_id", "category"]].copy()


def resolve_meal(items: list[dict[str, Any]], path: Path = FOOD_DB_PATH) -> Meal:
    """R3.5 ``resolve_meal(items)`` on P3's engine. Callers overwrite ``t`` and ``label``.

    items: [{"food": "bajra roti", "portion": 2, "unit": "piece"}]. ``food`` is a name from ``food_db()``
    (case-insensitive) or a P3 ``food_id``; ``unit`` is the row's portion_unit or "g".
    Unknown foods/units and non-positive portions raise ValueError (no fuzzy matching).
    """
    if not items:
        raise ValueError("a meal needs at least one item")
    eng = _engine()
    df = _table(str(path))
    by_name = {f: i for i, f in enumerate(df["food"])}
    by_id = {f.upper(): i for i, f in enumerate(df["food_id"])}
    src = eng.IndianFoodDB(str(path)).df
    tot = dict(carbs_g=0.0, protein_g=0.0, fat_g=0.0, fibre_g=0.0, gl=0.0)
    flags, names, out_items = [], [], []
    for it in items:
        key = str(it.get("food", "")).strip()
        i = by_name.get(key.lower(), by_id.get(key.upper()))
        if i is None:
            raise ValueError(f"unknown food {key!r}")
        row = df.iloc[i]
        portion = float(it.get("portion", 1))
        if portion <= 0:
            raise ValueError(f"portion for {row['food']!r} must be positive")
        unit = str(it.get("unit") or row["portion_unit"]).strip().lower()
        if unit == "g":
            n_units = portion / float(row["grams_per_unit"])
        elif unit == row["portion_unit"]:
            n_units = portion
        else:
            raise ValueError(f"unit {unit!r} not valid for {row['food']!r} (use {row['portion_unit']!r} or 'g')")
        vec = eng.compute_nutrient_vector(src[src["food_id"] == row["food_id"]].iloc[0], n_units)
        for k in ("carbs_g", "protein_g", "fat_g", "fibre_g"):
            tot[k] += vec[k]
        tot["gl"] += vec["glycemic_load"]
        if row["assumption_flag"]:
            flags.append(f"{row['food']}: {row['assumption_flag']}")
        names.append(f"{portion:g} {unit} {row['food']}")
        out_items.append({"food": row["food"], "portion": portion, "unit": unit})
    return Meal(
        t=datetime(2026, 1, 1, tzinfo=UTC),
        label=" + ".join(names),
        items=out_items,
        assumption_flags=flags or None,
        provenance=Provenance.SYNTHETIC,
        **{k: round(v, 2) for k, v in tot.items()},
    )


def clear_cache() -> None:
    _table.cache_clear()
