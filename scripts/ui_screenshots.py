"""Visual QA: screenshots of every page (1440×900 and 1280×720, light and dark) plus a live
theme-switch check. Drives the locally installed Google Chrome via Playwright (no browser download).

    MADHUTWIN_DEMO=1 streamlit run dashboard/app.py --server.port 8517   # in another shell
    python scripts/ui_screenshots.py [--base http://localhost:8517] [--check-only]

Writes docs/figures/ui/<page>_<theme>_<width>.png and exits 1 if the theme check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "ui"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = [("alert_queue", ""), ("patient", "patient_card"), ("forecast", "forecast_view"), ("meals", "meal_timeline"),
         ("whatif", "whatif_lab"), ("why", "explain_panel"), ("trust", "trust_page")]
SIZES = [(1440, 900), (1280, 720)]


def ready(page: Page, needs_chart: bool = False) -> None:
    page.wait_for_selector(".mt-header", timeout=60_000)
    if needs_chart:
        page.wait_for_selector(".js-plotly-plot .xtick text", timeout=30_000)
    page.wait_for_timeout(2500)


def capture(base: str, browser) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    shots = []
    for scheme in ("light", "dark"):
        for w, h in SIZES:
            ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme, device_scale_factor=1)
            page = ctx.new_page()
            for name, path in PAGES:
                page.goto(f"{base}/{path}")
                ready(page, needs_chart=name in ("forecast", "meals"))
                if name == "whatif":
                    page.get_by_role("button", name="Combined").click()
                    page.wait_for_timeout(3500)
                f = OUT / f"{name}_{scheme}_{w}.png"
                page.screenshot(path=str(f))
                shots.append(f)
            ctx.close()
    return shots


def theme_check(base: str, browser) -> list[str]:
    """Flip prefers-color-scheme mid-session; page CSS, Plotly chrome and the morph iframe must follow."""
    problems = []
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="light")
    page = ctx.new_page()

    def probe() -> dict:
        return page.evaluate("""() => {
            const q = s => document.querySelector(s);
            const fill = el => el ? getComputedStyle(el).fill : null;
            return {
              attr: document.documentElement.dataset.mtTheme,
              app: getComputedStyle(q('[data-testid="stApp"]')).backgroundColor,
              panel: q('[class*="st-key-panel"]') ? getComputedStyle(q('[class*="st-key-panel"]')).backgroundColor : null,
              tick: fill(q('.js-plotly-plot .xtick text')),
              grid: q('.js-plotly-plot .gridlayer path') ? getComputedStyle(q('.js-plotly-plot .gridlayer path')).stroke : null,
            };
        }""")

    for path, label in (("forecast_view", "forecast"), ("whatif_lab", "what-if")):
        page.emulate_media(color_scheme="light")
        page.goto(f"{base}/{path}")
        ready(page, needs_chart=path == "forecast_view")
        if path == "whatif_lab":
            page.get_by_role("button", name="Combined").click()
            page.wait_for_timeout(3000)
        before = probe()
        frame_before = _iframe_tick(page)
        page.emulate_media(color_scheme="dark")
        page.wait_for_timeout(2000)
        after = probe()
        frame_after = _iframe_tick(page)
        print(f"[{label}] light → dark")
        for k in before:
            print(f"   {k:6} {before[k]!s:28} → {after[k]}")
        if path == "whatif_lab":
            print(f"   iframe {frame_before!s:28} → {frame_after}")
        if before["attr"] != "light" or after["attr"] != "dark":
            problems.append(f"{label}: data-mt-theme did not flip ({before['attr']} → {after['attr']})")
        for k in ("app", "panel") + (("tick", "grid") if before["tick"] else ()):
            if before[k] == after[k]:
                problems.append(f"{label}: {k} colour did not change on theme switch")
        if path == "whatif_lab" and (frame_before is None or frame_before == frame_after):
            problems.append(f"{label}: morph iframe chrome did not follow the theme ({frame_before} → {frame_after})")
    ctx.close()
    return problems


def _iframe_tick(page: Page) -> str | None:
    for fr in page.frames:
        try:
            if fr.locator("#c .xtick text").count():
                return fr.evaluate("() => getComputedStyle(document.querySelector('#c .xtick text')).fill")
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8517")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=True)
        problems = theme_check(args.base, browser)
        if not args.check_only:
            shots = capture(args.base, browser)
            print(f"wrote {len(shots)} screenshots to {OUT.relative_to(ROOT)}")
        browser.close()
    print("theme check:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
