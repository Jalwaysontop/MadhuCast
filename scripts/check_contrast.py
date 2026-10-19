"""WCAG 2.1 AA contrast audit for the Jamun tokens (light and dark).

    python scripts/check_contrast.py        # exit 1 on any failure

Text pairs need 4.5:1 (3:1 for large text: ≥ 24px, or ≥ 18.66px bold).
Chart marks, focus rings and UI boundaries need 3:1 (WCAG 1.4.11).
Tinted badges are measured on the tint blended over the surface they sit on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.theme.tokens import ALPHA, CHART, DARK, LIGHT  # noqa: E402

TEXT, LARGE, MARK = 4.5, 3.0, 3.0


def _rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h: str) -> float:
    r, g, b = (_lin(c) for c in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(fg: str, bg: str, alpha: float) -> str:
    f, b = _rgb(fg), _rgb(bg)
    return "#" + "".join(f"{round((alpha * x + (1 - alpha) * y) * 255):02X}" for x, y in zip(f, b))


def pairs(p: dict[str, str]) -> list[tuple[str, str, str, float]]:
    out = []
    for bg in ("porcelain", "surface", "sunken"):
        out.append((f"ink on {bg}", p["ink"], p[bg], TEXT))
        out.append((f"ink-muted on {bg}", p["ink_muted"], p[bg], TEXT))
    out.append(("on-jamun on jamun (primary button)", p["on_jamun"], p["jamun"], TEXT))
    out.append(("jamun text on surface (links, active nav)", p["jamun"], p["surface"], TEXT))
    out.append(("jamun text on jamun-tint (active nav)", p["jamun"], p["jamun_tint"], TEXT))
    out.append(("ink on jamun-tint (disclaimer)", p["ink"], p["jamun_tint"], TEXT))
    for k in ("critical", "warning", "watch", "unknown", "target"):
        tint = blend(p[k], p["surface"], ALPHA["tint"])
        out.append((f"{k} text on its tint (tier badge)", p[k], tint, TEXT))
        out.append((f"{k} text on surface (readouts, labels)", p[k], p["surface"], TEXT))
    for k in ("prov_real", "prov_synthetic", "prov_mock", "jamun"):
        out.append((f"{k} badge text on surface", p[k], p["surface"], TEXT))
    for k in ("ink", "jamun", "scenario", "target", "critical", "warning"):
        out.append((f"{k} chart mark on surface", p[k], p["surface"], MARK))
    out.append(("hairline boundary vs porcelain (decorative only)", p["hairline"], p["porcelain"], 1.0))
    return out


def main() -> int:
    failures = 0
    for name, p in (("light", LIGHT), ("dark", DARK)):
        print(f"\n== {name}")
        for label, fg, bg, need in pairs(p):
            r = ratio(fg, bg)
            ok = r >= need
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {r:5.2f}:1  (need {need}:1)  {label}  {fg} on {bg}")
    # Chart marks are theme-proof: each must hold 3:1 on BOTH themes' panel surfaces.
    print("\n== chart marks (theme-proof, both surfaces)")
    for k, v in CHART.items():
        for bg in (LIGHT["surface"], DARK["surface"]):
            r = ratio(v, bg)
            failures += r < MARK
            print(f"  {'PASS' if r >= MARK else 'FAIL'}  {r:5.2f}:1  (need {MARK}:1)  chart {k} {v} on {bg}")
    # Wordmark in the logo image cannot switch theme: large bold text, 3:1 on both sidebars.
    for bg in (LIGHT["surface"], DARK["surface"]):
        r = ratio("#8B5CA3", bg)
        failures += r < LARGE
        print(f"  {'PASS' if r >= LARGE else 'FAIL'}  {r:5.2f}:1  (need {LARGE}:1)  logo wordmark #8B5CA3 on {bg}")
    print(f"\n{'All pairs pass.' if not failures else f'{failures} pair(s) fail.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
