"""Diagrams as code. Regenerate with:  python docs/architecture/build_diagrams.py

Writes docs/architecture_diagram.pdf (+ .png) and docs/privacy_boundary.pdf (+ .png).
Plain matplotlib so no Graphviz binary is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
    "#888888",
)


def box(ax, x, y, w, h, title, lines=(), color=BLUE, dashed=False, title_size=10, size=7.4, owner=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            fc=matplotlib.colors.to_rgba(color, 0.12),
            ec=color,
            lw=1.6,
            ls="--" if dashed else "-",
        )
    )
    ax.text(x + 0.008, y + h - 0.012, title, fontsize=title_size, weight="bold", va="top", color="#111")
    if owner:
        ax.text(x + w - 0.008, y + h - 0.012, owner, fontsize=7, va="top", ha="right", color=color, weight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 0.010, y + h - 0.045 - i * 0.030, line, fontsize=size, va="top", color="#222")


def arrow(ax, a, b, color="#333", dashed=False, text=None, rad=0.0, size=7, text_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.4,
            color=color,
            ls="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        tx, ty = text_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.012)
        ax.text(tx, ty, text, fontsize=size, ha="center", color=color)


def canvas(w=16.5, h=9.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def architecture() -> None:
    fig, ax = canvas()
    ax.text(0.01, 0.985, "MadhuTwin: architecture (7 layers)", fontsize=16, weight="bold", va="top")
    ax.text(
        0.01,
        0.955,
        "Research PoC on synthetic/open data. Owners: P1 frontend/integration · P2 model · "
        "P3 data. Dashed = PoC stub or future scope.",
        fontsize=9,
        va="top",
        color="#444",
    )

    ys, h = 0.47, 0.42
    cols = [
        (
            "1 · Data",
            "P3",
            GREEN,
            [
                "CGMacros: CGM, Fitbit HR/METs,",
                "  meal macros  [REAL]",
                "Synthea EHR  [SYNTHETIC]",
                "Genetic risk score  [SYNTHETIC]",
                "Indian food DB + meal engine",
                "  (150–250 items) [SYNTHETIC]",
                "Nearest-neighbour pairing",
                "  EHR ↔ CGMacros participant",
            ],
        ),
        (
            "2 · Feature",
            "P2/P3",
            SKY,
            [
                "5-min grid, missing mask",
                "  (gaps kept, never imputed",
                "  for display)",
                "Sleep proxy (night METs+HR)",
                "Meal GL / fibre / fat",
                "EHR conditioning vector",
                "Feature-lib version",
            ],
        ),
        (
            "3 · Model",
            "P2",
            BLUE,
            [
                "Glucose encoder",
                "FiLM conditioning on EHR",
                "Physics prior (glucose kinetics)",
                "Quantile heads q05…q95",
                "Spike / hypo classifiers",
                "Conformal calibration",
                "Ablation: EHR-only, wearable-",
                "  only, concat, FiLM, FiLM+phys",
            ],
        ),
        (
            "4 · Explain / Sim",
            "P1/P2",
            PURPLE,
            [
                "SHAP / IG attributions",
                "Attention (directional only)",
                "scenario.apply(): levers",
                "  meal swap, portion, walk,",
                "  meal time, sleep (no meds)",
                "Deterministic templates",
                "Attribution ↔ what-if check",
            ],
        ),
        (
            "5 · Serving",
            "P1",
            ORANGE,
            [
                "contracts.py (Pydantic v2)",
                "backend.py: mock | real",
                "  (data / model split)",
                "Alert engine: tiers, gating,",
                "  dedupe, cooldown, budget",
                "FHIR R4B export",
                "Logbook: feedback, audit",
            ],
        ),
        (
            "6 · Dashboard",
            "P1",
            VERM,
            [
                "Streamlit + Plotly",
                "Alert queue (triage)",
                "Patient card",
                "Forecast + bands",
                "Meal timeline",
                "What-if lab",
                "Why panel",
                "Model trust",
            ],
        ),
    ]
    w, gap, x0 = 0.148, 0.012, 0.01
    centers = []
    for i, (title, owner, color, lines) in enumerate(cols):
        x = x0 + i * (w + gap)
        box(ax, x, ys, w, h, title, lines, color=color, owner=owner)
        centers.append((x, x + w))
    labels = ["features", "PatientState", "Forecast", "Forecast + why", "pages"]
    for i in range(5):
        arrow(ax, (centers[i][1] + 0.002, ys + h / 2), (centers[i + 1][0] - 0.002, ys + h / 2), text=labels[i])

    # privacy boundary around data/edge
    ax.add_patch(
        FancyBboxPatch(
            (0.003, ys - 0.045), w + 0.014, h + 0.055, boxstyle="round,pad=0.004", fc="none", ec=VERM, lw=2.2, ls=":"
        )
    )
    ax.text(0.008, ys - 0.018, "privacy boundary: raw ring buffer", fontsize=7.2, color=VERM, weight="bold")
    ax.text(0.008, ys - 0.038, "stays on device/edge; features cross", fontsize=7.2, color=VERM, weight="bold")

    # Pull path
    arrow(
        ax,
        (centers[5][0] + 0.07, ys - 0.005),
        (centers[0][0] + 0.09, ys - 0.05),
        color=VERM,
        dashed=True,
        rad=-0.15,
        text="Pull: 'Request raw window' (consent flag + audit log; PoC stub logs only)",
        size=8.5,
        text_xy=(0.5, 0.315),
    )

    box(
        ax,
        0.01,
        0.03,
        0.98,
        0.17,
        "7 · Governance (cross-cutting)",
        [
            "Provenance label on every number: REAL / SYNTHETIC / MOCK / MODEL ESTIMATE  ·  synthetic-only data rule  ·  "
            "pseudonymous IDs (P-###), no names or addresses",
            "Model + feature-lib version on every alert and FHIR resource  ·  clinician feedback stored for a future "
            "federated loop  ·  consent flag + audit log for Pull",
            "Model-trust page: calibration, subgroups, Clarke grid, ablation, honest limits  ·  what-if labelled "
            "'model estimate, not causal, not medical advice'",
            "FUTURE SCOPE (not built): federated learning, ABDM integration, MQTT edge transport in production",
        ],
        color=GREY,
        size=8.2,
    )
    fig.savefig(DOCS / "architecture_diagram.pdf", bbox_inches="tight")
    fig.savefig(DOCS / "architecture_diagram.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def privacy() -> None:
    fig, ax = canvas(14, 7)
    ax.text(0.01, 0.98, "MadhuTwin: privacy boundary", fontsize=16, weight="bold", va="top")
    ax.text(
        0.01,
        0.935,
        "Fits the spirit of data-minimisation (e.g. DPDP Act); not a compliance claim.",
        fontsize=10,
        va="top",
        color="#444",
        style="italic",
    )
    box(
        ax,
        0.02,
        0.42,
        0.22,
        0.42,
        "Device / edge",
        ["CGM + wearable", "Raw ring buffer", "  STAYS LOCAL", "Aggregator → 5-min", "  features only"],
        color=GREEN,
        size=9.5,
    )
    box(
        ax,
        0.30,
        0.42,
        0.22,
        0.42,
        "Backend (local inference)",
        ["Pseudonymous IDs (P-###)", "Synthetic/open data only", "Model runs locally", "Versions logged"],
        color=BLUE,
        size=9.5,
    )
    box(
        ax,
        0.58,
        0.42,
        0.20,
        0.42,
        "Clinician dashboard",
        ["Forecasts + bands", "Alerts + feedback", "What-if (no meds)", "FHIR export (tagged", "  synthetic)"],
        color=VERM,
        size=9.5,
    )
    box(
        ax,
        0.82,
        0.42,
        0.16,
        0.42,
        "Audit log",
        ["logs/audit.jsonl", "every Pull request", "consent flag", "logs/feedback.jsonl"],
        color=GREY,
        size=9.5,
    )
    arrow(ax, (0.245, 0.66), (0.295, 0.66), text="features only")
    arrow(ax, (0.525, 0.66), (0.575, 0.66), text="forecasts")
    arrow(ax, (0.785, 0.60), (0.815, 0.60))
    arrow(
        ax,
        (0.66, 0.415),
        (0.13, 0.395),
        color=VERM,
        dashed=True,
        rad=-0.25,
        text="Pull raw window: needs consent flag; audited; PoC stub (no data moves)",
        size=9,
        text_xy=(0.40, 0.21),
    )
    ax.add_patch(
        FancyBboxPatch((0.01, 0.40), 0.245, 0.47, boxstyle="round,pad=0.004", fc="none", ec=VERM, lw=2.2, ls=":")
    )
    box(
        ax,
        0.30,
        0.04,
        0.30,
        0.14,
        "FUTURE SCOPE: federated learning",
        ["Train on-device; share model updates,", "not data. Feedback log is the seed."],
        color=PURPLE,
        dashed=True,
        size=9,
    )
    box(
        ax,
        0.64,
        0.04,
        0.30,
        0.14,
        "FUTURE SCOPE: ABDM integration",
        ["Consent-managed exchange via the", "Ayushman Bharat Digital Mission."],
        color=PURPLE,
        dashed=True,
        size=9,
    )
    fig.savefig(DOCS / "privacy_boundary.pdf", bbox_inches="tight")
    fig.savefig(DOCS / "privacy_boundary.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    privacy()
    print("wrote docs/architecture_diagram.{pdf,png} and docs/privacy_boundary.{pdf,png}")
