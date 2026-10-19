"""F9: FHIR export of a patient's recent CGM readings and the current forecast.

Built and validated with ``fhir.resources.R4B`` (ADR-002: R4B is wire-compatible with R4
for Patient, Observation, RiskAssessment and Bundle; we say "R4B" honestly).

Bundle (type ``collection``):
  * Patient: pseudonymous id only (no name/address), tagged as test/synthetic data.
  * Observation per CGM reading: LOINC 99504-3 (interstitial-fluid glucose, mass/volume,
    the code fixed by the HL7 CGM IG), UCUM mg/dL, category laboratory.
  * RiskAssessment: spike/hypo probabilities as predictions with probabilityDecimal and
    whenPeriod (a prediction is a risk, not a measurement; ADR-008).
  * Observation (status preliminary): the forecast curve, one component per quantile
    and horizon, method "MadhuTwin model estimate".

Run ``python -m src.serving.fhir_export`` to regenerate docs/samples/fhir_bundle_example.json.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fhir.resources.R4B.bundle import Bundle

from src.contracts import Forecast, PatientState

FHIR_VERSION = "4.3.0 (R4B)"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
CGM_LOINC = ("99504-3", "Glucose [Mass/volume] in Interstitial fluid")
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
ACT_REASON = "http://terminology.hl7.org/CodeSystem/v3-ActReason"
TAG_SYSTEM = "urn:madhutwin:tags"
NS = uuid.UUID("6f1c2b1e-6a43-4c55-9f0e-6d2d1a7a0b11")  # fixed namespace → reproducible ids
DISCLAIMER = (
    "Synthetic/open data only. Model estimate from observational data, not a proven causal "
    "effect. Research PoC, not a medical device. Not medical advice."
)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _urn(*parts: str) -> str:
    return f"urn:uuid:{uuid.uuid5(NS, '/'.join(parts))}"


def _mgdl(v: float) -> dict[str, Any]:
    return {"value": round(float(v), 1), "unit": "mg/dL", "system": UCUM, "code": "mg/dL"}


def build_bundle(state: PatientState, fc: Forecast, cgm_window_min: int = 240) -> dict[str, Any]:
    """Return a validated FHIR R4B Bundle as a JSON-ready dict."""
    pid = state.profile.patient_id
    now = state.now
    patient_url = _urn("Patient", pid)
    tags = [
        {"system": ACT_REASON, "code": "HTEST", "display": "test health data"},
        {"system": TAG_SYSTEM, "code": "synthetic", "display": "Synthetic / mock data, not a real person"},
    ]
    entries: list[dict[str, Any]] = [
        {
            "fullUrl": patient_url,
            "resource": {
                "resourceType": "Patient",
                "id": pid,
                "meta": {"tag": tags},
                "active": True,
                "identifier": [{"system": "urn:madhutwin:pseudonym", "value": pid}],
                "gender": {"M": "male", "F": "female"}[state.profile.sex],
            },
        }
    ]

    hist = state.history.copy()
    hist["t"] = pd.to_datetime(hist["t"], utc=True)
    win = hist[
        (hist["t"] > pd.Timestamp(now) - pd.Timedelta(minutes=cgm_window_min))
        & (hist["t"] <= pd.Timestamp(now))
        & ~hist["missing_mask"].astype(bool)
        & hist["glucose"].notna()
    ]
    for _, row in win.iterrows():
        t = row["t"].to_pydatetime()
        entries.append(
            {
                "fullUrl": _urn("Observation", pid, _iso(t)),
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "meta": {"tag": tags},
                    "category": [{"coding": [{"system": OBS_CATEGORY, "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {
                        "coding": [{"system": LOINC, "code": CGM_LOINC[0], "display": CGM_LOINC[1]}],
                        "text": "CGM interstitial glucose",
                    },
                    "subject": {"reference": patient_url},
                    "effectiveDateTime": _iso(t),
                    "valueQuantity": _mgdl(row["glucose"]),
                },
            }
        )

    model = f"MadhuTwin model estimate ({fc.model_version or 'unversioned'}; provenance {fc.provenance.value})"
    period = {"start": _iso(now), "end": _iso(now + timedelta(minutes=120))}
    entries.append(
        {
            "fullUrl": _urn("RiskAssessment", pid, _iso(now)),
            "resource": {
                "resourceType": "RiskAssessment",
                "status": "preliminary",
                "meta": {"tag": tags},
                "subject": {"reference": patient_url},
                "occurrenceDateTime": _iso(now),
                "method": {"text": model},
                "prediction": [
                    {
                        "outcome": {"text": "Glucose > 180 mg/dL (hyperglycaemic spike)"},
                        "probabilityDecimal": round(fc.spike_prob, 3),
                        "whenPeriod": period,
                    },
                    {
                        "outcome": {"text": "Glucose < 70 mg/dL (hypoglycaemia)"},
                        "probabilityDecimal": round(fc.hypo_prob, 3),
                        "whenPeriod": period,
                    },
                ],
                "note": [{"text": f"Model confidence {fc.confidence:.2f}. {DISCLAIMER}"}],
            },
        }
    )

    components = []
    for h in (30, 60, 120):
        for name, v in zip(("q10", "q50", "q90"), fc.horizon(h)):
            components.append({"code": {"text": f"{name} glucose at +{h} min"}, "valueQuantity": _mgdl(v)})
        if fc.quantiles_90 and str(h) in fc.quantiles_90:
            for name, v in zip(("q05", "q95"), fc.quantiles_90[str(h)]):
                components.append({"code": {"text": f"{name} glucose at +{h} min"}, "valueQuantity": _mgdl(v)})
    entries.append(
        {
            "fullUrl": _urn("Observation", pid, "forecast", _iso(now)),
            "resource": {
                "resourceType": "Observation",
                "status": "preliminary",
                "meta": {"tag": tags},
                "code": {"text": "MadhuTwin glucose forecast (interstitial, quantiles)"},
                "subject": {"reference": patient_url},
                "effectivePeriod": period,
                "issued": _iso(now),
                "method": {"text": model},
                "component": components,
                "note": [{"text": f"Feature library {fc.feature_lib_version or 'n/a'}. {DISCLAIMER}"}],
            },
        }
    )

    bundle = {
        "resourceType": "Bundle",
        "id": f"madhutwin-{pid}-{now.astimezone(UTC):%Y%m%dT%H%M}",
        "type": "collection",
        "timestamp": _iso(now),
        "entry": entries,
    }
    return json.loads(Bundle.model_validate(bundle).model_dump_json(exclude_none=True))


def validate(bundle: dict[str, Any]) -> Bundle:
    """Parse a bundle back with fhir.resources; raises on any schema violation."""
    return Bundle.model_validate(bundle)


def to_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, ensure_ascii=False)


def _write_sample(path: Path) -> Path:
    from src.serving import backend, mock_data

    state = backend.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    bundle = build_bundle(state, backend.predict(state), cgm_window_min=60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(bundle) + "\n")
    return path


if __name__ == "__main__":
    out = _write_sample(Path(__file__).resolve().parents[2] / "docs" / "samples" / "fhir_bundle_example.json")
    print(f"wrote {out}")
