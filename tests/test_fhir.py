import json
from pathlib import Path

from src.serving import backend, fhir_export, mock_data

SAMPLE = Path(__file__).resolve().parents[1] / "docs" / "samples" / "fhir_bundle_example.json"


def bundle():
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    return st, fhir_export.build_bundle(st, backend.predict(st))


def test_round_trip_validates():
    _, b = bundle()
    parsed = fhir_export.validate(json.loads(fhir_export.to_json(b)))
    assert parsed.type == "collection"
    kinds = [e.resource.get_resource_type() for e in parsed.entry]
    assert kinds[0] == "Patient" and "RiskAssessment" in kinds and kinds.count("Observation") > 40


def test_patient_is_pseudonymous_and_tagged():
    _, b = bundle()
    pat = b["entry"][0]["resource"]
    assert pat["id"] == "P-010" and "name" not in pat and "address" not in pat
    assert {t["code"] for t in pat["meta"]["tag"]} == {"HTEST", "synthetic"}


def test_cgm_observation_coding():
    _, b = bundle()
    obs = b["entry"][1]["resource"]
    assert obs["code"]["coding"][0] == {"system": "http://loinc.org", "code": "99504-3",
                                        "display": "Glucose [Mass/volume] in Interstitial fluid"}
    assert obs["valueQuantity"]["system"] == "http://unitsofmeasure.org" and obs["valueQuantity"]["code"] == "mg/dL"


def test_forecast_semantics():
    st, b = bundle()
    ra = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "RiskAssessment")
    assert len(ra["prediction"]) == 2 and all(0 <= p["probabilityDecimal"] <= 1 for p in ra["prediction"])
    assert "mock-0.1.0" in ra["method"]["text"]
    fobs = b["entry"][-1]["resource"]
    assert fobs["status"] == "preliminary" and len(fobs["component"]) == 15
    assert "not a medical device" in fobs["note"][0]["text"]


def test_missing_samples_are_not_exported():
    st = mock_data.get_state("P-007", mock_data.DEMO_NOW)
    b = fhir_export.build_bundle(st, backend.predict(st))
    n_obs = sum(e["resource"]["resourceType"] == "Observation" for e in b["entry"]) - 1
    assert n_obs == 48 - 9  # 4 h window minus the 45-min gap


def test_committed_sample_is_valid_and_reproducible():
    committed = json.loads(SAMPLE.read_text())
    fhir_export.validate(committed)
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    assert committed == fhir_export.build_bundle(st, backend.predict(st), cgm_window_min=60)
