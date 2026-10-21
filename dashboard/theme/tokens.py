"""Jamun design tokens: the single source of truth for the dashboard's look.

Mirrors the Figma file "MadhuTwin — Jamun UI" (variables "Jamun / light" and
"Jamun / dark", text styles, size tokens). Accessibility rules win over any visual
value; scripts/check_contrast.py enforces them.
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["light", "dark"]

LIGHT: dict[str, str] = {
    "porcelain": "#F4F6F7",  # app background (cool, not cream)
    "surface": "#FFFFFF",  # panels
    "sunken": "#ECEFF2",  # table header, tracks
    "ink": "#1B2130",  # primary text, history line, big numerals
    "ink_muted": "#5A6275",  # secondary text, axes
    "hairline": "#DDE2E8",  # dividers, panel borders
    "jamun": "#5B2A6E",  # brand, the twin, forecast median, primary actions
    "jamun_tint": "#F1EAF4",
    "on_jamun": "#FFFFFF",
    "target": "#1F7268",  # 70–180 mg/dL target range
    "critical": "#B3261E",
    "warning": "#8F5100",
    "watch": "#3A6EA5",
    "unknown": "#5F6673",  # always hatched/dashed
    "scenario": "#C2185B",  # what-if scenario curve (also dashed)
    "prov_synthetic": "#7A4A8C",
    "prov_mock": "#7A5F00",
    "prov_real": "#2F6F4F",
}

DARK: dict[str, str] = {
    "porcelain": "#0F121A",
    "surface": "#151925",
    "sunken": "#1C2130",
    "ink": "#E8EAF0",
    "ink_muted": "#A3AAB8",
    "hairline": "#2A3040",
    "jamun": "#B48ACB",
    "jamun_tint": "#2A2033",
    "on_jamun": "#151925",
    "target": "#4FB3A6",
    "critical": "#FF8A80",
    "warning": "#F0B45A",
    "watch": "#8AB4E8",
    "unknown": "#9CA3AF",
    "scenario": "#F48FB1",
    "prov_synthetic": "#C9A6D9",
    "prov_mock": "#E6CB5C",
    "prov_real": "#7BC8A0",
}

# Theme-proof chart palette: data marks keep ≥ 3:1 on BOTH light and dark surfaces, so a
# chart rendered before a theme switch still reads after it (st.context.theme is unreliable
# on first load and on menu changes). Chart text/grid are styled by CSS from live tokens.
CHART: dict[str, str] = {
    "history": "#6B7488",
    "jamun": "#8B5CA3",
    "scenario": "#D6427F",
    "target": "#2E9A8C",
    "warning": "#C27A1A",
    "critical": "#E04E44",
    "unknown": "#8A919E",
    "watch": "#4F86C6",
}

# Band / fill opacities (applied to the base colour)
ALPHA = {"band80": 0.18, "band90": 0.10, "target_fill": 0.08, "tint": 0.12, "scenario_band": 0.12}

FONT_UI = "Hanken Grotesk"
FONT_DEVANAGARI = "Noto Sans Devanagari"
GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700"
    "&family=Noto+Sans+Devanagari:wght@500&display=swap"
)
# 1.25 type scale (px): 12 / 14 / 16 / 20 / 25 / 31 / 39
TYPE = {"caption": 12, "label": 14, "body": 16, "h3": 20, "h2": 25, "h1": 31, "display": 39}
SPACE = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 8: 32}
RADIUS = {"chip": 2, "panel": 8, "pill": 999}
MOTION_MS = 450  # the one orchestrated motion: what-if scenario morph

TIER_COLOR = {"CRITICAL": "critical", "WARNING": "warning", "WATCH": "watch", "UNKNOWN": "unknown", "CLEAR": "target"}
TIER_ICON = {
    "CRITICAL": "octagon-alert",
    "WARNING": "triangle-alert",
    "WATCH": "eye",
    "UNKNOWN": "circle-dashed",
    "CLEAR": "circle-check",
}
PROV_COLOR = {"REAL": "prov_real", "SYNTHETIC": "prov_synthetic", "MOCK": "prov_mock", "MODEL_ESTIMATE": "jamun"}

# Lucide icons (ISC licence), inner SVG bodies on a 24-unit grid, stroke 1.5.
ICONS: dict[str, str] = {
    "octagon-alert": '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "eye": '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
    "circle-dashed": '<path d="M10.1 2.182a10 10 0 0 1 3.8 0"/><path d="M13.9 21.818a10 10 0 0 1-3.8 0"/><path d="M17.609 3.721a10 10 0 0 1 2.69 2.7"/><path d="M2.182 13.9a10 10 0 0 1 0-3.8"/><path d="M20.279 17.609a10 10 0 0 1-2.7 2.69"/><path d="M21.818 10.1a10 10 0 0 1 0 3.8"/><path d="M3.721 6.391a10 10 0 0 1 2.7-2.69"/><path d="M6.391 20.279a10 10 0 0 1-2.69-2.7"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "circle-x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "flask-conical": '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    "approximately-equal": '<path d="M5 15a6.5 6.5 0 0 1 7 0 6.5 6.5 0 0 0 7 0"/><path d="M5 9a6.5 6.5 0 0 1 7 0 6.5 6.5 0 0 0 7 0"/>',
    "trending-up": '<path d="M16 7h6v6"/><path d="m22 7-8.5 8.5-5-5L2 17"/>',
    "trending-down": '<path d="M16 17h6v-6"/><path d="m22 17-8.5-8.5-5 5L2 7"/>',
    "move-right": '<path d="M18 8L22 12L18 16"/><path d="M2 12H22"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
}

# Devanagari display names for the seed food table (display only; data keys stay English).
FOOD_HINDI: dict[str, str] = {
    "white rice": "चावल",
    "brown rice": "ब्राउन राइस",
    "wheat roti": "रोटी",
    "millet roti (bajra)": "बाजरा रोटी",
    "jowar roti": "ज्वार रोटी",
    "ragi mudde": "रागी मुद्दे",
    "idli": "इडली",
    "dosa (plain)": "डोसा",
    "poha": "पोहा",
    "upma": "उपमा",
    "oats porridge": "ओट्स",
    "khichdi": "खिचड़ी",
    "veg biryani": "बिरयानी",
    "sambar": "सांभर",
    "dal (toor)": "दाल",
    "rajma": "राजमा",
    "chana masala": "छोले",
    "mixed veg sabzi": "सब्ज़ी",
    "palak": "पालक",
    "paneer curry": "पनीर",
    "chicken curry": "चिकन करी",
    "boiled egg": "अंडा",
    "curd": "दही",
    "buttermilk": "छाछ",
    "sprouts salad": "अंकुरित सलाद",
    "banana": "केला",
    "apple": "सेब",
    "masala chai (sugar)": "चाय",
    "masala chai (no sugar)": "चाय",
    "samosa": "समोसा",
    "gulab jamun": "गुलाब जामुन",
    "papad": "पापड़",
}


def palette(mode: Mode = "light") -> dict[str, str]:
    return DARK if mode == "dark" else LIGHT


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
