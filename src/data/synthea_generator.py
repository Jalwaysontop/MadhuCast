"""
Synthea EHR Generator
=====================
Generates synthetic Type 2 Diabetes patient EHR records using Synthea,
with calibration parameters shifted toward Indian demographics.

Synthea is an open-source (Apache 2.0) synthetic patient generator.
It outputs FHIR Bundles (JSON) that we parse into a flat DataFrame.

Download: https://github.com/synthetichealth/synthea
Requires: Java 11+ to run Synthea; Python fhir.resources to parse output.

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SYNTHEA_DIR = Path("tools/synthea")       # Where Synthea JAR lives
DEFAULT_OUTPUT_DIR  = Path("data/raw/synthea")     # Where FHIR Bundles land
DEFAULT_NUM_PATIENTS = 100

# ── India-calibrated T2D cohort parameters ────────────────────────────────────
# These override Synthea defaults to produce a cohort closer to an Indian
# Type 2 Diabetes population.  Sources: WHO India Diabetes Profile 2023,
# ICMR INDIAB study prevalence data.
#
# NOTE: Synthea is US-centric by design.  These settings shift the *output*
#       distributions but do NOT make the records clinically Indian.  This
#       limitation is stated plainly in the dataset card and README.

INDIA_CALIBRATION = {
    # Age range — T2D onset is earlier in South Asian populations
    "age_min": 25,
    "age_max": 70,
    "age_mean": 48,
    "age_std": 12,

    # BMI — Asian BMI thresholds are lower; overweight ≥ 23, obese ≥ 25
    "bmi_min": 20.0,
    "bmi_max": 38.0,
    "bmi_mean": 26.5,
    "bmi_std": 3.8,

    # HbA1c — T2D cohort
    "hba1c_min": 6.5,
    "hba1c_max": 12.0,
    "hba1c_mean": 8.2,
    "hba1c_std": 1.5,

    # Fasting glucose (mg/dL)
    "fasting_glucose_min": 100,
    "fasting_glucose_max": 300,
    "fasting_glucose_mean": 155,
    "fasting_glucose_std": 40,

    # Lipids (mg/dL)
    "total_cholesterol_mean": 195,
    "total_cholesterol_std": 35,
    "ldl_mean": 120,
    "ldl_std": 30,
    "hdl_mean": 42,
    "hdl_std": 10,
    "triglycerides_mean": 165,
    "triglycerides_std": 55,

    # Sex distribution (approximate)
    "male_fraction": 0.58,

    # Comorbidity flags — prevalence in Indian T2D
    "hypertension_prevalence": 0.50,
    "ckd_prevalence": 0.15,
    "cvd_prevalence": 0.20,
    "retinopathy_prevalence": 0.18,
    "neuropathy_prevalence": 0.25,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Option A:  Generate via Synthea CLI  (requires Java + Synthea download)
# ═════════════════════════════════════════════════════════════════════════════

def run_synthea(
    num_patients: int = DEFAULT_NUM_PATIENTS,
    synthea_dir: Path | str = DEFAULT_SYNTHEA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    state: str = "Massachusetts",   # Synthea default; we post-process anyway
) -> Path:
    """
    Run Synthea to generate FHIR Bundle JSON files.

    Parameters
    ----------
    num_patients : Number of patients to generate.
    synthea_dir  : Path to the Synthea installation (contains the JAR).
    output_dir   : Where to write output FHIR Bundles.
    seed         : Random seed for reproducibility.
    state        : US state for Synthea (we override demographics after).

    Returns
    -------
    Path to the output directory containing FHIR JSON files.
    """
    synthea_dir = Path(synthea_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the Synthea JAR
    jar_files = list(synthea_dir.glob("synthea*.jar"))
    if not jar_files:
        jar_files = list(synthea_dir.glob("**/*.jar"))

    if not jar_files:
        logger.warning(
            "Synthea JAR not found in %s. "
            "Falling back to synthetic generation without Synthea.",
            synthea_dir,
        )
        return _generate_synthetic_ehr(num_patients, output_dir, seed)

    jar = jar_files[0]
    logger.info("Using Synthea JAR: %s", jar)

    cmd = [
        "java", "-jar", str(jar),
        "-p", str(num_patients),
        "-s", str(seed),
        "--exporter.fhir.export", "true",
        "--exporter.hospital.fhir.export", "false",
        "--exporter.csv.export", "false",
        "--exporter.baseDirectory", str(output_dir),
        state,
    ]

    logger.info("Running Synthea: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("Synthea failed:\n%s", result.stderr)
        raise RuntimeError(f"Synthea exited with code {result.returncode}")

    logger.info("Synthea completed. Output in %s", output_dir)
    return output_dir


# ═════════════════════════════════════════════════════════════════════════════
#  Option B:  Generate synthetic EHR directly (no Synthea dependency)
# ═════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_ehr(
    num_patients: int = DEFAULT_NUM_PATIENTS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> Path:
    """
    Generate synthetic EHR records directly in Python, calibrated to Indian
    T2D demographics.  This is the fallback when Synthea is not installed.

    Outputs one JSON file per patient in FHIR-like structure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cal = INDIA_CALIBRATION

    logger.info("Generating %d synthetic EHR records (India-calibrated)…", num_patients)

    for i in range(num_patients):
        patient_id = f"SYN-{i+1:04d}"

        sex = "M" if rng.random() < cal["male_fraction"] else "F"
        age = int(np.clip(
            rng.normal(cal["age_mean"], cal["age_std"]),
            cal["age_min"], cal["age_max"]
        ))
        bmi = round(float(np.clip(
            rng.normal(cal["bmi_mean"], cal["bmi_std"]),
            cal["bmi_min"], cal["bmi_max"]
        )), 1)
        hba1c = round(float(np.clip(
            rng.normal(cal["hba1c_mean"], cal["hba1c_std"]),
            cal["hba1c_min"], cal["hba1c_max"]
        )), 1)
        fasting_glucose = round(float(np.clip(
            rng.normal(cal["fasting_glucose_mean"], cal["fasting_glucose_std"]),
            cal["fasting_glucose_min"], cal["fasting_glucose_max"]
        )), 0)

        # Lipids
        tc  = round(float(rng.normal(cal["total_cholesterol_mean"], cal["total_cholesterol_std"])), 0)
        ldl = round(float(rng.normal(cal["ldl_mean"], cal["ldl_std"])), 0)
        hdl = round(float(rng.normal(cal["hdl_mean"], cal["hdl_std"])), 0)
        tg  = round(float(rng.normal(cal["triglycerides_mean"], cal["triglycerides_std"])), 0)

        # Comorbidities
        comorbidities = []
        if rng.random() < cal["hypertension_prevalence"]:
            comorbidities.append("hypertension")
        if rng.random() < cal["ckd_prevalence"]:
            comorbidities.append("chronic_kidney_disease")
        if rng.random() < cal["cvd_prevalence"]:
            comorbidities.append("cardiovascular_disease")
        if rng.random() < cal["retinopathy_prevalence"]:
            comorbidities.append("diabetic_retinopathy")
        if rng.random() < cal["neuropathy_prevalence"]:
            comorbidities.append("diabetic_neuropathy")

        # Simulated genetic risk score (0–1 scale)
        genetic_risk = round(float(rng.beta(2, 3)), 3)

        # Duration of diabetes (years)
        diabetes_duration = max(0, int(rng.normal(6, 4)))

        record = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "gender": "male" if sex == "M" else "female",
                        "birthDate": f"{2026 - age}-01-01",
                        "extension": [
                            {"url": "synthetic_source", "valueString": "madhutwin_generator"},
                            {"url": "india_calibrated", "valueBoolean": True},
                        ]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"{patient_id}-demographics",
                        "code": {"text": "demographics"},
                        "component": [
                            {"code": {"text": "age"}, "valueQuantity": {"value": age, "unit": "years"}},
                            {"code": {"text": "sex"}, "valueQuantity": {"value": sex}},
                            {"code": {"text": "bmi"}, "valueQuantity": {"value": bmi, "unit": "kg/m2"}},
                            {"code": {"text": "diabetes_status"}, "valueQuantity": {"value": 2}},
                            {"code": {"text": "diabetes_duration_years"}, "valueQuantity": {"value": diabetes_duration, "unit": "years"}},
                        ]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"{patient_id}-labs",
                        "code": {"text": "laboratory_results"},
                        "component": [
                            {"code": {"text": "hba1c"}, "valueQuantity": {"value": hba1c, "unit": "%"}},
                            {"code": {"text": "fasting_glucose"}, "valueQuantity": {"value": fasting_glucose, "unit": "mg/dL"}},
                            {"code": {"text": "total_cholesterol"}, "valueQuantity": {"value": tc, "unit": "mg/dL"}},
                            {"code": {"text": "ldl"}, "valueQuantity": {"value": ldl, "unit": "mg/dL"}},
                            {"code": {"text": "hdl"}, "valueQuantity": {"value": hdl, "unit": "mg/dL"}},
                            {"code": {"text": "triglycerides"}, "valueQuantity": {"value": tg, "unit": "mg/dL"}},
                        ]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"{patient_id}-risk",
                        "code": {"text": "risk_factors"},
                        "component": [
                            {"code": {"text": "genetic_risk_score"}, "valueQuantity": {"value": genetic_risk}},
                            {"code": {"text": "comorbidities"}, "valueString": ",".join(comorbidities) if comorbidities else "none"},
                        ]
                    }
                },
            ]
        }

        out_path = output_dir / f"{patient_id}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

    logger.info("Generated %d synthetic EHR records in %s", num_patients, output_dir)
    return output_dir


# ═════════════════════════════════════════════════════════════════════════════
#  FHIR Bundle parser  →  flat DataFrame
# ═════════════════════════════════════════════════════════════════════════════

def parse_fhir_bundles(
    fhir_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Parse all FHIR Bundle JSON files into a single flat DataFrame.

    Returns
    -------
    DataFrame with columns:
        patient_id            : str
        age                   : float
        sex                   : str
        bmi                   : float
        diabetes_status       : int
        diabetes_duration_years: int
        hba1c                 : float
        fasting_glucose       : float
        total_cholesterol     : float
        ldl                   : float
        hdl                   : float
        triglycerides         : float
        genetic_risk_score    : float
        comorbidity_hypertension       : bool
        comorbidity_ckd                : bool
        comorbidity_cvd                : bool
        comorbidity_retinopathy        : bool
        comorbidity_neuropathy         : bool
        is_synthetic          : bool   (always True)
    """
    fhir_dir = Path(fhir_dir)
    json_files = sorted(fhir_dir.glob("*.json"))

    if not json_files:
        logger.warning("No FHIR JSON files found in %s", fhir_dir)
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    for jf in json_files:
        try:
            with open(jf) as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", jf.name, e)
            continue

        record = _extract_patient_record(bundle)
        if record:
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["is_synthetic"] = True

    # Ensure boolean comorbidity columns
    for comorb in ("hypertension", "ckd", "cvd", "retinopathy", "neuropathy"):
        col = f"comorbidity_{comorb}"
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(bool)

    logger.info("Parsed %d synthetic EHR records from FHIR Bundles", len(df))
    return df


def _extract_patient_record(bundle: Dict) -> Optional[Dict[str, Any]]:
    """Extract a flat record dict from a single FHIR Bundle."""
    record: Dict[str, Any] = {}

    entries = bundle.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        if rtype == "Patient":
            record["patient_id"] = resource.get("id", str(uuid.uuid4())[:8])
            gender = resource.get("gender", "")
            record["sex"] = "M" if gender == "male" else "F" if gender == "female" else ""

            birth = resource.get("birthDate", "")
            if birth:
                try:
                    birth_year = int(birth.split("-")[0])
                    record["age"] = 2026 - birth_year
                except (ValueError, IndexError):
                    pass

        elif rtype == "Observation":
            code_text = resource.get("code", {}).get("text", "").lower()
            components = resource.get("component", [])

            for comp in components:
                comp_code = comp.get("code", {}).get("text", "").lower()
                val_q = comp.get("valueQuantity", {})
                val_s = comp.get("valueString", "")
                value = val_q.get("value") if val_q else val_s

                if comp_code == "age" and value is not None:
                    record["age"] = float(value) if not isinstance(value, str) else record.get("age")
                elif comp_code == "sex":
                    record["sex"] = str(value)
                elif comp_code == "bmi":
                    record["bmi"] = float(value)
                elif comp_code == "diabetes_status":
                    record["diabetes_status"] = int(value)
                elif comp_code == "diabetes_duration_years":
                    record["diabetes_duration_years"] = int(value)
                elif comp_code == "hba1c":
                    record["hba1c"] = float(value)
                elif comp_code == "fasting_glucose":
                    record["fasting_glucose"] = float(value)
                elif comp_code == "total_cholesterol":
                    record["total_cholesterol"] = float(value)
                elif comp_code == "ldl":
                    record["ldl"] = float(value)
                elif comp_code == "hdl":
                    record["hdl"] = float(value)
                elif comp_code == "triglycerides":
                    record["triglycerides"] = float(value)
                elif comp_code == "genetic_risk_score":
                    record["genetic_risk_score"] = float(value)
                elif comp_code == "comorbidities":
                    comorbs = str(value).lower().split(",")
                    record["comorbidity_hypertension"] = "hypertension" in comorbs
                    record["comorbidity_ckd"] = "chronic_kidney_disease" in comorbs
                    record["comorbidity_cvd"] = "cardiovascular_disease" in comorbs
                    record["comorbidity_retinopathy"] = "diabetic_retinopathy" in comorbs
                    record["comorbidity_neuropathy"] = "diabetic_neuropathy" in comorbs

    return record if record.get("patient_id") else None


# ═════════════════════════════════════════════════════════════════════════════
#  Patient profile exporter  →  JSON files for Person 1 (dashboard)
# ═════════════════════════════════════════════════════════════════════════════

def export_patient_profiles(
    ehr_df: pd.DataFrame,
    output_dir: Path | str = "data/processed/patient_profiles",
) -> None:
    """
    Write per-patient EHR summary as JSON files.

    These are consumed by Person 1 for the dashboard patient card.

    Interface contract:
        data/processed/patient_profiles/<patient_id>.json
        {
          "patient_id": str,
          "age": int, "sex": str, "bmi": float,
          "hba1c": float, "fasting_glucose": float,
          "lipids": {"tc": float, "ldl": float, "hdl": float, "tg": float},
          "comorbidities": [str, ...],
          "genetic_risk_score": float,
          "diabetes_duration_years": int,
          "is_synthetic": bool
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in ehr_df.iterrows():
        pid = row.get("patient_id", row.get("participant_id"))
        if not pid:
            continue

        comorbidities = []
        for comorb in ("hypertension", "ckd", "cvd", "retinopathy", "neuropathy"):
            if row.get(f"comorbidity_{comorb}", False):
                comorbidities.append(comorb)

        profile = {
            "patient_id": str(pid),
            "age": int(row.get("age", 0)),
            "sex": str(row.get("sex", "")),
            "bmi": round(float(row.get("bmi", 0)), 1),
            "hba1c": round(float(row.get("hba1c", 0)), 1),
            "fasting_glucose": round(float(row.get("fasting_glucose", 0)), 0),
            "lipids": {
                "total_cholesterol": round(float(row.get("total_cholesterol", 0)), 0),
                "ldl": round(float(row.get("ldl", 0)), 0),
                "hdl": round(float(row.get("hdl", 0)), 0),
                "triglycerides": round(float(row.get("triglycerides", 0)), 0),
            },
            "comorbidities": comorbidities,
            "genetic_risk_score": round(float(row.get("genetic_risk_score", 0)), 3),
            "diabetes_duration_years": int(row.get("diabetes_duration_years", 0)),
            "diabetes_status": int(row.get("diabetes_status", 2)),
            "is_synthetic": bool(row.get("is_synthetic", True)),
        }

        out_path = output_dir / f"{pid}.json"
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)

    logger.info("Exported %d patient profiles to %s", len(ehr_df), output_dir)
