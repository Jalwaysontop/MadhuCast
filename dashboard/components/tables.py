"""Pure table helpers for the trust page (unit-tested; no Streamlit)."""

from __future__ import annotations

import pandas as pd

LOWER_IS_BETTER = ("rmse", "mae", "false_alarm")


def lower_is_better(metric: str) -> bool:
    return metric.startswith(LOWER_IS_BETTER)


def best_without_overlap(df: pd.DataFrame, metric: str) -> str | None:
    """The best variant for ``metric`` only if its CI overlaps no other variant's CI; else None."""
    d = df[df["metric"] == metric]
    if len(d) < 2:
        return None
    asc = lower_is_better(metric)
    d = d.sort_values("value", ascending=asc)
    best, rest = d.iloc[0], d.iloc[1:]
    overlaps = ((rest["ci_low"] <= best["ci_high"]) & (best["ci_low"] <= rest["ci_high"])).any()
    return None if overlaps else str(best["variant"])


def ablation_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """variant × metric with 'value [lo, hi]' strings, in the order given by the file."""
    df = df.copy()
    df["cell"] = df.apply(lambda r: f"{r['value']:.3g} [{r['ci_low']:.3g}, {r['ci_high']:.3g}]", axis=1)
    order = list(dict.fromkeys(df["variant"]))
    out = df.pivot(index="variant", columns="metric", values="cell").reindex(order)
    n = df.groupby("variant")["n_participants"].max().reindex(order)
    out.insert(0, "n_participants", n)
    return out.reset_index()


PROVENANCE_REGISTER = pd.DataFrame(
    [
        {
            "Dataset": "CGMacros (CGM, Fitbit HR/METs, meal macros)",
            "Role": "Dynamic stream",
            "Provenance": "REAL",
            "Licence": "VERIFY: one paper reports CC BY-NC-SA 4.0",
            "Caveat": "NonCommercial clause: confirm prize/commercial implications. ~45 participants.",
        },
        {
            "Dataset": "Synthea (simulated EHR)",
            "Role": "Static stream",
            "Provenance": "SYNTHETIC",
            "Licence": "Apache-2.0",
            "Caveat": "Simulated patients. Paired to CGMacros by nearest neighbour, not the same person.",
        },
        {
            "Dataset": "Genetic risk score",
            "Role": "Static feature",
            "Provenance": "SYNTHETIC",
            "Licence": "Generated in-project",
            "Caveat": "No genotype data used anywhere.",
        },
        {
            "Dataset": "Indian meal augmentation + food table (IFCT/INDB-derived)",
            "Role": "Meals / what-if",
            "Provenance": "SYNTHETIC",
            "Licence": "VERIFY IFCT/INDB terms",
            "Caveat": "150–250 items: table stakes, done openly. Assumption-flagged rows are marked ⚠.",
        },
        {
            "Dataset": "Sleep signal",
            "Role": "Dynamic feature",
            "Provenance": "SYNTHETIC",
            "Licence": "Derived",
            "Caveat": "Proxy from night METs and HR, not measured sleep.",
        },
    ]
)

LIMITATIONS = [
    "The pairing is synthetic: EHR records are matched to CGM participants, not observed together.",
    "45 participants means wide confidence intervals.",
    "The sleep signal is a proxy derived from night METs and heart rate.",
    "Thresholds are not clinically validated.",
    "Fusion may not beat glucose-only; the ablation shows whether it does.",
]
