"""F10 Patient card: static stream (EHR), pairing disclosure, twin status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import common, kit

common.init_session()
COLD_START_DAYS = 3

LABS = [  # key, label, unit, reference text, function(value, sex) -> status
    (
        "hba1c",
        "HbA1c",
        "%",
        "< 5.7 normal · 5.7–6.4 prediabetes · ≥ 6.5 diabetes (ADA)",
        lambda v, s: "normal" if v < 5.7 else "prediabetes range" if v < 6.5 else "diabetes range",
    ),
    (
        "fasting_glucose",
        "Fasting glucose",
        "mg/dL",
        "70–99 normal · 100–125 prediabetes · ≥ 126 diabetes",
        lambda v, s: "normal" if v < 100 else "prediabetes range" if v < 126 else "diabetes range",
    ),
    (
        "bmi",
        "BMI",
        "kg/m²",
        "18.5–22.9 normal · 23–24.9 overweight · ≥ 25 obese (WHO Asia-Pacific)",
        lambda v, s: "normal" if v < 23 else "overweight" if v < 25 else "obese",
    ),
]
LIPIDS = {
    "ldl": ("LDL cholesterol", "< 100", lambda v, s: "above target" if v >= 100 else "at target"),
    "hdl": ("HDL cholesterol", "> 40 (M) / > 50 (F)", lambda v, s: "low" if v < (40 if s == "M" else 50) else "ok"),
    "tg": ("Triglycerides", "< 150", lambda v, s: "high" if v >= 150 else "ok"),
}


def render() -> None:
    state = common.get_state()
    p = state.profile
    prov = p.provenance
    st.title(f"Patient {p.patient_id}")
    days = p.history_days
    cold = days is not None and days < COLD_START_DAYS
    pb = lambda f: common.prov_badge(prov[f]) if prov.get(f) else ""  # noqa: E731
    kit.readouts(
        [
            kit.Readout("Cohort", "T2D" if p.cohort == "t2d" else p.cohort.capitalize(), "", "Diagnosis band"),
            kit.Readout("Age, sex", f"{p.age}", "years", "Female" if p.sex == "F" else "Male", badge=pb("age")),
            kit.Readout("HbA1c", f"{p.hba1c:.1f}", "%", LABS[0][4](p.hba1c, p.sex).capitalize(), badge=pb("hba1c")),
            kit.Readout("BMI", f"{p.bmi:.1f}", "kg/m²", LABS[2][4](p.bmi, p.sex).capitalize(), badge=pb("bmi")),
            kit.Readout(
                "Twin status",
                "Cold start" if cold else "Personalised",
                "",
                (
                    f"{days:g} days of CGM" + (f", under {COLD_START_DAYS}: wider uncertainty" if cold else "")
                    if days is not None
                    else "History length not provided"
                ),
                foot_tone="warn" if cold else None,
            ),
        ]
    )
    st.write("")
    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(key="panel-labs"):
        st.subheader("Labs")
        rows = [
            {
                "Test": label,
                "Value": f"{getattr(p, key):g} {unit}",
                "Status": fn(getattr(p, key), p.sex),
                "Reference": ref,
                "Provenance": prov[key].value.capitalize() if prov.get(key) else "n/a",
            }
            for key, label, unit, ref, fn in LABS
        ]
        for key, v in (p.lipids or {}).items():
            label, ref, fn = LIPIDS.get(key, (key.upper(), "", lambda v, s: ""))
            rows.append(
                {
                    "Test": label,
                    "Value": f"{v:g} mg/dL",
                    "Status": fn(v, p.sex),
                    "Reference": f"{ref} mg/dL",
                    "Provenance": prov["lipids"].value.capitalize() if prov.get("lipids") else "n/a",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        kit.caption("Reference ranges are general adult guidance, shown for orientation only.")
        kit.md(
            kit.definition_list(
                [
                    ("Comorbidities", ", ".join(p.comorbidities) or "None recorded", ""),
                    ("Medications, display only", ", ".join(p.medications) or "None recorded", ""),
                ]
            )
        )
        kit.caption("Medications are never a what-if lever, by design.")
    with right, st.container(key="panel-identity"):
        st.subheader("Synthetic parts, disclosed")
        rows = []
        if p.genetic_risk_score is not None:
            rows.append(("Genetic risk score (0–1)", f"{p.genetic_risk_score:.2f}", common.prov_badge("SYNTHETIC")))
        kit.md(kit.definition_list(rows) if rows else "")
        kit.caption("No genotype data is used anywhere in this project.")
        pr = p.pairing or {}
        if pr:
            kit.disclaimer(
                f"EHR synthetically paired to CGMacros participant #{pr.get('cgmacros_id', '?')} by nearest "
                f"neighbour on {', '.join(pr.get('match_on', ['status', 'age', 'BMI', 'HbA1c']))} bands, "
                f"distance {pr.get('match_distance', '?')}. The EHR and the CGM trace are not from the same "
                "real person."
            )
        else:
            kit.empty_state("No pairing metadata.", "Person 3's profile JSON should include a pairing block.")
        kit.md(
            '<div style="font-size:14px;font-weight:600;margin:16px 0 6px">Field provenance</div>'
            + " ".join(
                f"<span class='mt-chip'>{kit.esc(k)}</span>{common.prov_badge(v)}" for k, v in sorted(prov.items())
            )
        )


common.guard(render)
