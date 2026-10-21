"""Design-system guards: contrast, Streamlit theme config, both palettes in the stylesheet."""

import sys
import tomllib
from pathlib import Path

from dashboard.theme import css, plotly_template, tokens

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_all_token_pairs_pass_wcag_aa():
    import check_contrast

    assert check_contrast.main() == 0


def test_config_fonts_name_one_family_per_url():
    # Streamlit rejects a font URL naming several families (session start fails silently in the UI).
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    for key in ("font", "headingFont"):
        url = cfg["theme"][key].split(":", 1)[1]
        assert url.count("family=") == 1, f"{key} must load exactly one family"


def test_config_matches_tokens():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    for mode, pal in (("light", tokens.LIGHT), ("dark", tokens.DARK)):
        t = cfg["theme"][mode]
        assert t["primaryColor"] == pal["jamun"] and t["backgroundColor"] == pal["porcelain"]
        assert t["textColor"] == pal["ink"] and t["secondaryBackgroundColor"] == pal["surface"]


def test_stylesheet_has_both_palettes_and_live_theme_hook():
    s = css.stylesheet("dark")
    assert ':root[data-mt-theme="light"]' in s and ':root[data-mt-theme="dark"]' in s
    assert tokens.LIGHT["porcelain"] in s and tokens.DARK["porcelain"] in s
    assert "prefers-reduced-motion" in s and "focus-visible" in s
    assert "dataset.mtTheme" in css.THEME_SYNC_JS


def test_plotly_templates_register():
    plotly_template.register()
    import plotly.io as pio

    assert {"madhutwin_light", "madhutwin_dark"} <= set(pio.templates)
    assert pio.templates["madhutwin_light"].layout.font.family.startswith(tokens.FONT_UI)
