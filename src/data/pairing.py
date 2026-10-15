"""
EHR ↔ Wearable Pairing Script
==============================
Matches each synthetic EHR patient (from Synthea) to a CGMacros participant
using nearest-neighbour on:
  - diabetes status  (exact match preferred)
  - age band
  - BMI band
  - HbA1c band

The pairing rule and match distances are logged in ``data/pairing_log.csv``
for full transparency and reproducibility.

Interface contract:
    Output  →  data/pairing_log.csv
    Columns :  synthetic_patient_id, cgmacros_participant_id,
               match_distance, diabetes_status_match,
               age_diff, bmi_diff, hba1c_diff, seed

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ── Feature weights for distance computation ──────────────────────────────────
# Higher weight = more important for matching quality.
DEFAULT_WEIGHTS = {
    "diabetes_status": 10.0,   # Exact-match strongly preferred
    "age":              1.0,
    "bmi":              1.5,
    "hba1c":            2.0,
}

# ── Band definitions (for logging clarity) ────────────────────────────────────
AGE_BANDS   = [(0, 30, "young"), (30, 45, "middle-young"), (45, 60, "middle"), (60, 100, "senior")]
BMI_BANDS   = [(0, 23, "normal"), (23, 25, "overweight"), (25, 30, "obese-I"), (30, 100, "obese-II")]
HBA1C_BANDS = [(0, 5.7, "normal"), (5.7, 6.5, "prediabetes"), (6.5, 8, "controlled-T2D"), (8, 20, "uncontrolled-T2D")]


def _assign_band(value: float, bands: List[Tuple[float, float, str]]) -> str:
    """Assign a value to a named band."""
    for lo, hi, name in bands:
        if lo <= value < hi:
            return name
    return bands[-1][2]


# ═════════════════════════════════════════════════════════════════════════════
#  Pairing engine
# ═════════════════════════════════════════════════════════════════════════════

def pair_ehr_to_wearable(
    ehr_df: pd.DataFrame,
    cgmacros_demographics: pd.DataFrame,
    cgmacros_labs: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    allow_reuse: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Match each synthetic EHR patient to a CGMacros participant.

    Parameters
    ----------
    ehr_df : DataFrame from ``synthea_generator.parse_fhir_bundles()`` or
             ``_generate_synthetic_ehr()``.  Must have columns:
             patient_id, age, sex, bmi, diabetes_status, hba1c.
    cgmacros_demographics : DataFrame from ``cgmacros_loader.load_demographics()``.
             Must have: participant_id, age, sex, bmi, diabetes_status.
    cgmacros_labs : DataFrame from ``cgmacros_loader.load_labs()``.
             Must have: participant_id, hba1c.  If None, HbA1c is excluded
             from distance.
    weights : Feature-weight dict (default: DEFAULT_WEIGHTS).
    allow_reuse : If True, the same CGMacros participant can be matched to
                  multiple EHR patients (necessary when N_ehr > N_cgmacros).
    seed : Random seed used for tie-breaking and reproducibility.

    Returns
    -------
    DataFrame — the pairing log with one row per EHR patient.
    """
    weights = weights or DEFAULT_WEIGHTS.copy()
    rng = np.random.default_rng(seed)

    # ── Prepare CGMacros feature matrix ──────────────────────────────────────
    cgm_feat = cgmacros_demographics[["participant_id", "age", "bmi", "diabetes_status"]].copy()
    if cgmacros_labs is not None and not cgmacros_labs.empty:
        labs_sub = cgmacros_labs[["participant_id", "hba1c"]].copy()
        cgm_feat = cgm_feat.merge(labs_sub, on="participant_id", how="left")
    else:
        cgm_feat["hba1c"] = np.nan

    cgm_feat.dropna(subset=["age", "bmi"], inplace=True)
    if cgm_feat.empty:
        raise ValueError("No CGMacros participants with valid age/BMI for pairing.")

    # ── Prepare EHR feature matrix ───────────────────────────────────────────
    ehr_feat = ehr_df[["patient_id", "age", "bmi", "diabetes_status", "hba1c"]].copy()
    ehr_feat.dropna(subset=["age", "bmi"], inplace=True)
    if ehr_feat.empty:
        raise ValueError("No EHR patients with valid age/BMI for pairing.")

    # ── Build numeric arrays for distance computation ────────────────────────
    feature_cols = ["diabetes_status", "age", "bmi", "hba1c"]
    w = np.array([weights.get(c, 1.0) for c in feature_cols])

    # Fill missing HbA1c with mean so it doesn't dominate distance
    global_hba1c_mean = pd.concat([cgm_feat["hba1c"], ehr_feat["hba1c"]]).mean()
    cgm_feat["hba1c"].fillna(global_hba1c_mean, inplace=True)
    ehr_feat["hba1c"].fillna(global_hba1c_mean, inplace=True)

    # Z-score normalise each feature so weights are meaningful
    combined = pd.concat([
        cgm_feat[feature_cols],
        ehr_feat[feature_cols],
    ], ignore_index=True)
    means = combined.mean()
    stds  = combined.std().replace(0, 1)

    cgm_norm = ((cgm_feat[feature_cols] - means) / stds).values
    ehr_norm = ((ehr_feat[feature_cols] - means) / stds).values

    # Weighted Euclidean distance
    dist_matrix = cdist(ehr_norm * w, cgm_norm * w, metric="euclidean")

    # ── Greedy nearest-neighbour assignment ──────────────────────────────────
    used_count: Dict[int, int] = {}
    max_reuse = len(ehr_feat) if allow_reuse else 1

    pairs: List[Dict] = []

    # Randomise order for fair tie-breaking
    ehr_indices = list(range(len(ehr_feat)))
    rng.shuffle(ehr_indices)

    for ei in ehr_indices:
        row_dists = dist_matrix[ei].copy()

        # Prefer unused CGMacros participants
        sorted_cgm = np.argsort(row_dists)

        matched = False
        for ci in sorted_cgm:
            if used_count.get(ci, 0) < max_reuse:
                used_count[ci] = used_count.get(ci, 0) + 1
                matched = True

                ehr_row = ehr_feat.iloc[ei]
                cgm_row = cgm_feat.iloc[ci]

                pair = {
                    "synthetic_patient_id":     ehr_row["patient_id"],
                    "cgmacros_participant_id":  cgm_row["participant_id"],
                    "match_distance":           round(float(row_dists[ci]), 4),
                    "diabetes_status_match":    int(ehr_row["diabetes_status"]) == int(cgm_row["diabetes_status"]),
                    "age_diff":                 round(float(abs(ehr_row["age"] - cgm_row["age"])), 1),
                    "bmi_diff":                 round(float(abs(ehr_row["bmi"] - cgm_row["bmi"])), 1),
                    "hba1c_diff":               round(float(abs(ehr_row["hba1c"] - cgm_row["hba1c"])), 1),
                    "ehr_age_band":             _assign_band(ehr_row["age"], AGE_BANDS),
                    "ehr_bmi_band":             _assign_band(ehr_row["bmi"], BMI_BANDS),
                    "ehr_hba1c_band":           _assign_band(ehr_row["hba1c"], HBA1C_BANDS),
                    "cgm_age_band":             _assign_band(cgm_row["age"], AGE_BANDS),
                    "cgm_bmi_band":             _assign_band(cgm_row["bmi"], BMI_BANDS),
                    "cgm_hba1c_band":           _assign_band(cgm_row["hba1c"], HBA1C_BANDS),
                    "seed":                     seed,
                }
                pairs.append(pair)
                break

        if not matched:
            logger.warning(
                "Could not find a match for EHR patient %s",
                ehr_feat.iloc[ei]["patient_id"],
            )

    pairing_df = pd.DataFrame(pairs)
    pairing_df.sort_values("synthetic_patient_id", inplace=True)
    pairing_df.reset_index(drop=True, inplace=True)

    logger.info(
        "Paired %d EHR patients to %d unique CGMacros participants. "
        "Mean distance: %.3f, Diabetes-status exact match: %d/%d",
        len(pairing_df),
        pairing_df["cgmacros_participant_id"].nunique(),
        pairing_df["match_distance"].mean(),
        pairing_df["diabetes_status_match"].sum(),
        len(pairing_df),
    )

    return pairing_df


# ═════════════════════════════════════════════════════════════════════════════
#  Save / load pairing log
# ═════════════════════════════════════════════════════════════════════════════

def save_pairing_log(
    pairing_df: pd.DataFrame,
    output_path: Path | str = "data/pairing_log.csv",
) -> Path:
    """Write the pairing log to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairing_df.to_csv(output_path, index=False)
    logger.info("Pairing log saved to %s (%d pairs)", output_path, len(pairing_df))
    return output_path


def load_pairing_log(
    path: Path | str = "data/pairing_log.csv",
) -> pd.DataFrame:
    """Load a previously saved pairing log."""
    return pd.read_csv(path)


# ═════════════════════════════════════════════════════════════════════════════
#  Merge paired data  (convenience)
# ═════════════════════════════════════════════════════════════════════════════

def build_paired_dataset(
    pairing_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
    cgmacros_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Merge EHR records with their paired CGMacros wearable data.

    Returns a dict with keys matching cgmacros_data, but each DataFrame
    now has an additional ``paired_patient_id`` column linking to the
    synthetic EHR.
    """
    pair_map = dict(zip(
        pairing_df["cgmacros_participant_id"],
        pairing_df["synthetic_patient_id"],
    ))
    # For many-to-one, build reverse map
    reverse_map: Dict[str, List[str]] = {}
    for _, row in pairing_df.iterrows():
        cgm_pid = row["cgmacros_participant_id"]
        syn_pid = row["synthetic_patient_id"]
        reverse_map.setdefault(cgm_pid, []).append(syn_pid)

    paired: Dict[str, pd.DataFrame] = {}

    for key, df in cgmacros_data.items():
        if "participant_id" not in df.columns or df.empty:
            paired[key] = df.copy()
            continue

        rows: List[pd.DataFrame] = []
        for cgm_pid, syn_pids in reverse_map.items():
            subset = df[df["participant_id"] == cgm_pid]
            if subset.empty:
                continue
            for syn_pid in syn_pids:
                chunk = subset.copy()
                chunk["paired_patient_id"] = syn_pid
                rows.append(chunk)

        if rows:
            paired[key] = pd.concat(rows, ignore_index=True)
        else:
            paired[key] = df.copy()
            paired[key]["paired_patient_id"] = np.nan

    logger.info("Built paired dataset with %d streams", len(paired))
    return paired
