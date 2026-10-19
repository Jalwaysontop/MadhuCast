"""F10 Patient card: static stream (EHR), pairing disclosure, twin status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components import common

common.init_session()
COLD_START_DAYS = 3

LABS = [  # key, label, unit, reference text, function(value, sex) -> status
    ("hba1c", "HbA1c", "%", "< 5.7 normal · 5.7–6.4 prediabetes · ≥ 6.5 diabetes (ADA)",
     lambda v, s: "normal" if v < 5.7 else "prediabetes range" if v < 6.5 else "diabetes range"),
    ("fasting_glucose", "Fasting glucose", "mg/dL", "70–99 normal · 100–125 prediabetes · ≥ 126 diabetes",
     lambda v, s: "normal" if v < 100 else "prediabetes range" if v < 126 else "diabetes range"),
    ("bmi", "BMI", "kg/m²", "18.5–22.9 normal · 23–24.9 overweight · ≥ 25 obese (WHO Asia-Pacific)",
     lambda v, s: "normal" if v < 23 else "overweight" if v < 25 else "obese"),
]
LIPIDS = {"ldl": ("LDL cholesterol", "< 100", lambda v, s: "above target" if v >= 100 else "at target"),
          "hdl": ("HDL cholesterol", "> 40 (M) / > 50 (F)",
                  lambda v, s: "low" if v < (40 if s == "M" else 50) else "ok"),
          "tg": ("Triglycerides", "< 150", lambda v, s: "high" if v >= 150 else "ok")}


def render() -> None:
    state = common.get_state()
    p = state.profile
    prov = p.provenance
    st.title(f"Patient {p.patient_id}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohort", p.cohort.upper())
    c2.metric("Age / sex", f"{p.age} {p.sex}")
    c3.metric("HbA1c", f"{p.hba1c:.1f}%")
    c4.metric("BMI", f"{p.bmi:.1f}")

    days = p.history_days
    if days is not None and days < COLD_START_DAYS:
        st.warning(f"Twin status: **limited history** ({days:g} days of CGM, under {COLD_START_DAYS}). Cold start: "
                   "forecasts lean on population patterns and uncertainty is wider.", icon="🧊")
    else:
        st.success(f"Twin status: personalised ({days:g} days of history)" if days is not None
                   else "Twin status: history length not provided", icon="🧬")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Labs")
        rows = [{"Test": label, "Value": f"{getattr(p, key):g} {unit}", "Status": fn(getattr(p, key), p.sex),
                 "Reference": ref, "Provenance": (prov.get(key) or "n/a").value if prov.get(key) else "n/a"}
                for key, label, unit, ref, fn in LABS]
        for key, v in (p.lipids or {}).items():
            label, ref, fn = LIPIDS.get(key, (key.upper(), "", lambda v, s: ""))
            rows.append({"Test": label, "Value": f"{v:g} mg/dL", "Status": fn(v, p.sex), "Reference": f"{ref} mg/dL",
                         "Provenance": prov["lipids"].value if prov.get("lipids") else "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Reference ranges are general adult guidance, shown for orientation only.")

        st.markdown("#### Comorbidities")
        st.write(", ".join(p.comorbidities) or "None recorded")
        st.markdown("#### Medications *(display only)*")
        st.write(", ".join(p.medications) or "None recorded")
        st.caption("Medications are never a what-if lever, by design.")

    with right:
        st.markdown("#### Genetic risk score")
        if p.genetic_risk_score is not None:
            common.md(f"<b>{p.genetic_risk_score:.2f}</b> (0–1 scale) {common.prov_badge('SYNTHETIC')}")
            st.caption("Clearly synthetic. No genotype data is used anywhere in this project.")
        else:
            st.write("Not provided")
        st.markdown("#### Pairing disclosure")
        pr = p.pairing or {}
        if pr:
            st.info(f"EHR synthetically paired to CGMacros participant #{pr.get('cgmacros_id', '?')} by nearest "
                    f"neighbour on {'/'.join(pr.get('match_on', ['status', 'age', 'BMI', 'HbA1c']))} bands; "
                    f"distance {pr.get('match_distance', '?')}. The EHR and the CGM trace are not from the same "
                    "real person.", icon="🔗")
        else:
            st.write("No pairing metadata provided.")
        st.markdown("#### Field provenance")
        common.md(" ".join(f"<span class='mt-chip'>{k}</span>{common.prov_badge(v)}" for k, v in sorted(prov.items())))


common.guard(render)
