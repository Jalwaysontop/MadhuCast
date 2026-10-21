"""Write the R3.1-R3.3 per-patient files from Person 3's pipeline (P1-owned glue, ADR-016).

P3's pipeline (``scripts/build_dataset.py``) ends in model-training windows (NPZ); the dashboard
contract needs per-patient files instead:

    data/processed/patient_profiles/P-###.json   PatientProfile
    data/processed/timeseries/P-###.parquet       5-min grid: t, glucose, hr, mets, sleep_proxy, missing_mask
    data/processed/meals/P-###.json               list[Meal]

This module reuses P3's own steps (CGMacros loader or its demo fallback, synthetic EHR, pairing,
Indian meal augmentation) and only reshapes their outputs. Every payload is validated against
``src.contracts`` before it is written.

Conversions that are P1 choices, not P3's (also in docs/decisions.md ADR-016):
* Wall-clock timestamps are read as IST, so a 08:00 breakfast shows at 08:00 IST.
* Glucose is averaged into 5-min bins and never interpolated; empty bins are ``missing_mask``.
* METs come from step cadence: 1.2 at rest + 2.3 per 100 steps/min (100 steps/min ~ 3.5 METs).
* ``sleep_proxy`` is 1 inside a logged sleep interval, else 0.
* Meals without a GI (every non-substituted CGMacros meal) get GL from GI 55, flagged.
* T2D wearers take their nearest synthetic EHR (P3's pairing). P3's EHR cohort is T2D-only, so
  healthy and prediabetes wearers keep their own demographics and labs (no pairing block).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import Meal, PatientProfile, PatientState, Provenance
from src.serving import p3_bridge
from src.serving import timeseries as ts

ROOT = Path(__file__).resolve().parents[2]
CGMACROS_DIR = ROOT / "data" / "raw" / "cgmacros"
OUT_DIR = ROOT / "data" / "processed"
STEP = f"{ts.STEP_MIN}min"
DEFAULT_GI = 55.0
REST_METS, METS_PER_100_SPM = 1.2, 2.3
COHORT = {0: "healthy", 1: "prediabetes", 2: "t2d"}
MATCH_ON = ["status", "age", "bmi", "hba1c"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Export:
    pid: str
    source_id: str
    profile: PatientProfile
    history: pd.DataFrame
    meals: list[Meal]


# --------------------------------------------------------------------------- P3 pipeline steps


def load_wearables(raw_dir: Path = CGMACROS_DIR, seed: int = 42) -> tuple[dict[str, pd.DataFrame], bool]:
    """(streams, is_real). Falls back to P3's demo cohort exactly as build_dataset.py does."""
    if raw_dir.exists():
        from src.data.cgmacros_loader import get_participant_ids, load_all

        data = load_all(raw_dir)
        if get_participant_ids(data):
            return data, True
        log.warning("CGMacros at %s has no complete participants; using P3's demo cohort", raw_dir)
    from scripts.build_dataset import _create_demo_cgmacros

    data, _ = _create_demo_cgmacros(seed)
    return data, False


def synthetic_ehr(n: int, seed: int) -> pd.DataFrame:
    from src.data.synthea_generator import _generate_synthetic_ehr, parse_fhir_bundles

    with tempfile.TemporaryDirectory() as tmp:
        _generate_synthetic_ehr(num_patients=n, output_dir=Path(tmp), seed=seed)
        return parse_fhir_bundles(Path(tmp))


def best_pairs(ehr: pd.DataFrame, data: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    """One synthetic EHR per wearable participant (smallest match distance), as build_dataset.py does."""
    from src.data.pairing import pair_ehr_to_wearable

    pairs = pair_ehr_to_wearable(
        ehr_df=ehr, cgmacros_demographics=data["demographics"], cgmacros_labs=data["labs"], seed=seed
    )
    if pairs.empty:
        return pairs
    pairs = pairs[pairs["diabetes_status_match"].astype(bool)]
    return (
        pairs.sort_values("match_distance")
        .drop_duplicates("cgmacros_participant_id")
        .set_index("cgmacros_participant_id")
    )


def augmented_meals(meals: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    from src.data.augmentation import augment_meals
    from src.data.meal_engine import IndianFoodDB

    if meals.empty:
        return meals.assign(is_augmented=False, indian_meal_items=pd.NA, glycemic_load=np.nan)
    return augment_meals(meals, IndianFoodDB(p3_bridge.FOOD_DB_PATH), augmentation_fraction=fraction, seed=seed)


# --------------------------------------------------------------------------- reshaping


def _utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s)
    if t.dt.tz is None:
        t = t.dt.tz_localize(ts.IST)
    return t.dt.tz_convert("UTC")


def _binned(df: pd.DataFrame, col: str, grid: pd.DatetimeIndex) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=grid)
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=_utc(df["timestamp"]))
    return s.resample(STEP).mean().reindex(grid)


def _mets(steps: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Step counts per logging bucket -> cadence -> METs on the 5-min grid. No steps logged -> NaN."""
    if steps.empty:
        return pd.Series(np.nan, index=grid)
    t = _utc(steps["timestamp"])
    n = pd.to_numeric(steps["steps"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(t.to_numpy())
    t, n = t.iloc[order], n[order]
    bucket = float(np.median(np.diff(t.to_numpy()) / np.timedelta64(1, "m"))) if len(t) > 1 else ts.STEP_MIN
    bucket = max(bucket, 1.0)
    cadence = pd.Series(n / bucket, index=t).groupby(level=0).sum()
    cadence = cadence.resample(STEP).mean().reindex(grid).ffill(limit=max(int(bucket // ts.STEP_MIN) - 1, 0))
    return (REST_METS + METS_PER_100_SPM * cadence / 100).round(2)


def _sleep(sleep: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(0.0, index=grid)
    for start, end in zip(_utc(sleep["sleep_start"]), _utc(sleep["sleep_end"])) if not sleep.empty else []:
        out[(grid >= start) & (grid < end)] = 1.0
    return out


def history(data: dict[str, pd.DataFrame], wid: str) -> pd.DataFrame:
    pick = {k: v[v["participant_id"] == wid] for k, v in data.items() if "participant_id" in v}
    cgm = pick["cgm"]
    t = _utc(cgm["timestamp"])
    grid = pd.date_range(t.min().floor(STEP), t.max().floor(STEP), freq=STEP)
    glucose = _binned(cgm, "glucose_mg_dl", grid).round(1)
    glucose = glucose.where(glucose.between(20, 600))
    return pd.DataFrame(
        {
            "t": grid,
            "glucose": glucose.to_numpy(),
            "hr": _binned(pick["heart_rate"], "heart_rate_bpm", grid).round(0).to_numpy(),
            "mets": _mets(pick["steps"], grid).to_numpy(),
            "sleep_proxy": _sleep(pick["sleep"], grid).to_numpy(),
            "missing_mask": glucose.isna().to_numpy(),
        }
    )


def _row_value(row: pd.Series | dict, key: str) -> float | None:
    v = row.get(key) if row is not None else None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def profile(
    pid: str,
    wid: str,
    data: dict[str, pd.DataFrame],
    pair: pd.Series | None,
    ehr: pd.DataFrame,
    real: bool,
    days: float,
) -> PatientProfile:
    demo = data["demographics"].set_index("participant_id").loc[wid]
    labs = data["labs"].set_index("participant_id")
    lab = labs.loc[wid] if wid in labs.index else {}
    worn = Provenance.REAL if real else Provenance.SYNTHETIC
    if pair is not None:
        rec = ehr.set_index("patient_id").loc[pair["synthetic_patient_id"]]
        src = {k: rec.get(k) for k in ("age", "sex", "bmi", "hba1c", "fasting_glucose")}
        lip = {"ldl": "ldl", "hdl": "hdl", "tg": "triglycerides", "chol": "total_cholesterol"}
        lipids = {k: _row_value(rec, c) for k, c in lip.items()}
        comorb = [c for c in ("hypertension", "ckd", "cvd", "retinopathy", "neuropathy") if rec.get(f"comorbidity_{c}")]
        grs = _row_value(rec, "genetic_risk_score")
        status = int(rec.get("diabetes_status", demo["diabetes_status"]))
        prov_ehr = Provenance.SYNTHETIC
        pairing: dict[str, Any] | None = {
            "cgmacros_id": wid,
            "synthetic_ehr_id": str(pair["synthetic_patient_id"]),
            "match_distance": round(float(pair["match_distance"]), 3),
            "match_on": MATCH_ON,
        }
    else:
        src = {"age": demo["age"], "sex": demo["sex"], "bmi": demo["bmi"]}
        src |= {"hba1c": _row_value(lab, "hba1c"), "fasting_glucose": _row_value(lab, "fasting_glucose")}
        lip = {"ldl": "ldl", "hdl": "hdl", "tg": "triglycerides", "chol": "total_cholesterol"}
        lipids = {k: _row_value(lab, c) for k, c in lip.items()}
        comorb, grs, pairing = [], None, None
        status = int(demo["diabetes_status"])
        prov_ehr = worn
    lipids = {k: round(v, 0) for k, v in lipids.items() if v is not None}
    payload = {
        "patient_id": pid,
        "cohort": COHORT[status],
        "age": int(round(float(src["age"]))),
        "sex": str(src["sex"]).upper()[:1],
        "bmi": round(float(src["bmi"]), 1),
        "hba1c": round(float(src["hba1c"]), 1),
        "fasting_glucose": round(float(src["fasting_glucose"]), 0),
        "lipids": lipids or None,
        "comorbidities": comorb,
        "genetic_risk_score": round(grs, 3) if grs is not None else None,
        "pairing": pairing,
        "history_days": round(days, 1),
        "provenance": {
            **{k: prov_ehr for k in ("age", "sex", "bmi", "hba1c", "fasting_glucose", "comorbidities")},
            **({"lipids": prov_ehr} if lipids else {}),
            **({"genetic_risk_score": Provenance.SYNTHETIC} if grs is not None else {}),
            "glucose": worn,
            "wearable": worn,
        },
    }
    return PatientProfile.model_validate(payload)


def meals(rows: pd.DataFrame, real: bool) -> list[Meal]:
    db = p3_bridge.food_db().set_index("food_id")
    out = []
    for _, r in rows.sort_values("timestamp").iterrows():
        t = _utc(pd.Series([r["timestamp"]])).iloc[0].to_pydatetime()
        kind = str(r.get("meal_type") or "meal").replace("_", " ").capitalize()
        macros = {k: max(0.0, _row_value(r, k) or 0.0) for k in ("carbs_g", "protein_g", "fat_g", "fibre_g")}
        items, flags = None, []
        raw_items = r.get("indian_meal_items")
        if bool(r.get("is_augmented")) and isinstance(raw_items, str):
            items = []
            for d in json.loads(raw_items):
                row = db.loc[d["food_id"]]
                items.append({"food": row["food"], "portion": float(d["portions"]), "unit": row["portion_unit"]})
                if row["assumption_flag"]:
                    flags.append(f"{row['food']}: {row['assumption_flag']}")
            flags.append("Indian meal substituted by carbohydrate match; glucose response is the original meal's")
            gl = _row_value(r, "glycemic_load") or 0.0
            prov = Provenance.SYNTHETIC
            label = f"{kind}: " + " + ".join(f"{i['portion']:g} {i['unit']} {i['food']}" for i in items)
        else:
            gl = DEFAULT_GI * max(macros["carbs_g"] - macros["fibre_g"], 0.0) / 100
            flags.append(f"GL estimated with GI {DEFAULT_GI:g}; the source meal log has no GI")
            prov = Provenance.REAL if real else Provenance.SYNTHETIC
            label = f"{kind}, {macros['carbs_g']:.0f} g carbs"
        out.append(
            Meal(
                t=t,
                label=label,
                gl=round(gl, 1),
                items=items,
                assumption_flags=flags or None,
                provenance=prov,
                **{k: round(v, 1) for k, v in macros.items()},
            )
        )
    return out


# --------------------------------------------------------------------------- driver


def build(
    raw_dir: Path = CGMACROS_DIR, seed: int = 42, n_ehr: int = 100, augmentation: float = 0.30
) -> tuple[list[Export], bool]:
    data, real = load_wearables(raw_dir, seed)
    ehr = synthetic_ehr(n_ehr, seed)
    pairs = best_pairs(ehr, data, seed)
    food = augmented_meals(data["meals"], augmentation, seed)
    wids = sorted(data["cgm"]["participant_id"].unique())
    wids = [w for w in wids if w in set(data["demographics"]["participant_id"])]
    if len(wids) > 999:
        raise ValueError("more than 999 participants do not fit the P-### pseudonym")
    out = []
    for i, wid in enumerate(wids, start=1):
        pid = f"P-{i:03d}"
        hist = history(data, wid)
        days = (hist["t"].iloc[-1] - hist["t"].iloc[0]).total_seconds() / 86400
        pair = pairs.loc[wid] if wid in pairs.index else None
        prof = profile(pid, wid, data, pair, ehr, real, days)
        ml = meals(food[food["participant_id"] == wid], real)
        # The same validation backend._real_state applies, so a bad export fails here, not in the UI.
        PatientState(profile=prof, history=hist, meals=ml, now=hist["t"].iloc[-1].to_pydatetime())
        out.append(Export(pid, wid, prof, hist, ml))
    return out, real


def write(exports: list[Export], out_dir: Path = OUT_DIR) -> None:
    dirs = {k: out_dir / k for k in ("patient_profiles", "timeseries", "meals")}
    for d in dirs.values():
        if d.exists():
            for f in [*d.glob("P-*.json"), *d.glob("P-*.parquet")]:
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    for e in exports:
        (dirs["patient_profiles"] / f"{e.pid}.json").write_text(e.profile.model_dump_json(indent=2))
        e.history.to_parquet(dirs["timeseries"] / f"{e.pid}.parquet", index=False)
        (dirs["meals"] / f"{e.pid}.json").write_text(json.dumps([m.model_dump(mode="json") for m in e.meals], indent=2))
