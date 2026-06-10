#!/usr/bin/env python3
"""Generate benchmark comparison card (SVG + PNG) from paper tables."""

from __future__ import annotations

import math
from pathlib import Path

# InternVideo3, Qwen3-VL-8B, InternVL3.5-8B, InternVideo2.5-7B
MODELS = [
    ("InternVideo3", True),
    ("Qwen3-VL\n8B", False),
    ("InternVL3.5\n8B", False),
    ("InternVideo\n2.5-7B", False),
]


def avg(*values: float | None) -> float:
    nums = [v for v in values if v is not None]
    return int(sum(nums) / len(nums) * 10 + 0.5) / 10


SECTIONS = [
    (
        "Long Video Understanding",
        [
            ("Video-MME", [73.8, 71.4, 66.0, 65.1]),
            ("VideoMME-v2", [27.6, 27.9, 26.0, 26.4]),
            ("LongVideoBench", [66.8, 62.4, 62.1, 60.6]),
            ("MLVU", [77.3, 57.6, 53.2, 72.8]),
            ("LVBench", [55.7, 58.0, 43.4, 46.4]),
            ("VRBench", [69.4, 59.4, 64.1, 51.9]),
            ("EgoSchema", [76.6, 69.8, 58.6, 63.9]),
            (
                "Long Avg.",
                [
                    avg(73.8, 27.6, 66.8, 77.3, 55.7, 69.4, 76.6),
                    avg(71.4, 27.9, 62.4, 57.6, 58.0, 59.4, 69.8),
                    avg(66.0, 26.0, 62.1, 53.2, 43.4, 64.1, 58.6),
                    avg(65.1, 26.4, 60.6, 72.8, 46.4, 51.9, 63.9),
                ],
            ),
        ],
    ),
    (
        "Short Video Understanding",
        [
            ("NextQA", [85.5, 83.4, 81.7, 84.9]),
            ("PerceptionTest", [81.4, 72.7, 72.7, 74.9]),
            ("MVBench", [75.0, 68.7, 72.1, 75.7]),
            ("TOMATO", [37.4, 35.7, 24.6, 32.9]),
            ("MotionBench", [60.6, 56.9, 56.6, 60.8]),
            ("TempCompass", [74.0, 74.3, 70.3, 70.1]),
            (
                "Short Avg.",
                [
                    avg(85.5, 81.4, 75.0, 37.4, 60.6, 74.0),
                    avg(83.4, 72.7, 68.7, 35.7, 56.9, 74.3),
                    avg(81.7, 72.7, 72.1, 24.6, 56.6, 70.3),
                    avg(84.9, 74.9, 75.7, 32.9, 60.8, 70.1),
                ],
            ),
        ],
    ),
    (
        "Spatiotemporal Intelligence",
        [
            ("QVHighlights", [59.9, 59.4, 31.3, 32.7]),
            ("Charades-STA", [50.4, 48.3, 27.8, 39.7]),
            ("ActivityNet (ANet)", [47.9, 46.8, 31.3, 24.8]),
            ("VSI-Bench", [68.1, 59.1, 56.0, 33.4]),
            ("MMSI-Bench", [27.6, 27.0, 30.5, 26.9]),
            ("MMSI-Bench-Video", [30.7, 27.6, 29.2, 27.3]),
            (
                "ST Avg.",
                [
                    avg(59.9, 50.4, 47.9, 68.1, 27.6, 30.7),
                    avg(59.4, 48.3, 46.8, 59.1, 27.0, 27.6),
                    avg(31.3, 27.8, 31.3, 56.0, 30.5, 29.2),
                    avg(32.7, 39.7, 24.8, 33.4, 26.9, 27.3),
                ],
            ),
        ],
    ),
    (
        "Overall",
        [
            (
                "Overall Avg.",
                [
                    avg(
                        73.8,
                        27.6,
                        66.8,
                        77.3,
                        55.7,
                        69.4,
                        76.6,
                        85.5,
                        81.4,
                        75.0,
                        37.4,
                        60.6,
                        74.0,
                        59.9,
                        50.4,
                        47.9,
                        68.1,
                        27.6,
                        30.7,
                    ),
                    avg(
                        71.4,
                        27.9,
                        62.4,
                        57.6,
                        58.0,
                        59.4,
                        69.8,
                        83.4,
                        72.7,
                        68.7,
                        35.7,
                        56.9,
                        74.3,
                        59.4,
                        48.3,
                        46.8,
                        59.1,
                        27.0,
                        27.6,
                    ),
                    avg(
                        66.0,
                        26.0,
                        62.1,
                        53.2,
                        43.4,
                        64.1,
                        58.6,
                        81.7,
                        72.7,
                        72.1,
                        24.6,
                        56.6,
                        70.3,
                        31.3,
                        27.8,
                        31.3,
                        56.0,
                        30.5,
                        29.2,
                    ),
                    avg(
                        65.1,
                        26.4,
                        60.6,
                        72.8,
                        46.4,
                        51.9,
                        63.9,
                        84.9,
                        74.9,
                        75.7,
                        32.9,
                        60.8,
                        70.1,
                        32.7,
                        39.7,
                        24.8,
                        33.4,
                        26.9,
                        27.3,
                    ),
                ],
            ),
        ],
    ),
]

FONT = "Inter, Helvetica, Arial, sans-serif"
BOLD_FONT = "Arial Black, Inter Black, Inter, Helvetica, Arial, sans-serif"
COLORS = {
    "bg": "#f5f8f7",
    "card": "#ffffff",
    "border": "#dfe7e3",
    "section_line": "#9cadA8",
    "row_line": "#eef3f1",
    "row_fill": "#ffffff",
    "row_alt": "#fbfdfc",
    "header_band": "#f3faf7",
    "category_fill": "#f4faf8",
    "category_border": "#d5ebe4",
    "footer_line": "#b5bfbc",
    "label": "#34413e",
    "header": "#1e2926",
    "section_title": "#1d2825",
    "highlight_header": "#123d34",
    "text": "#222827",
    "sota": "#d23d3d",
    "footer": "#7b8582",
    "highlight_fill_top": "#f6fffb",
    "highlight_fill_bot": "#dff9f0",
    "highlight_stroke": "#152b28",
    "accent": "#73d9bf",
    "accent_light": "#aaeada",
}

W = 1780
MARGIN_X, MARGIN_Y = 48, 48
CARD_PAD = 26
CAT_COL_CENTER = 160.0
CAT_COL_LEFT = 82.0
CAT_COL_RIGHT = 270.0
BENCH_X = 320.0
COL_X = [740.0, 975.0, 1210.0, 1445.0]
HIGHLIGHT_COL_IDX = 0
HIGHLIGHT_X = COL_X[HIGHLIGHT_COL_IDX] - 111
HIGHLIGHT_W = 222
ROW_LINE_X1 = CAT_COL_RIGHT + 8
ROW_LINE_X2_OFFSET = 72


def fmt_value(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v:.1f}"


def rank_styles(values: list[float | None]) -> list[tuple[str, bool, bool]]:
    """Return (display, is_sota, is_second) per cell."""
    numeric = [(i, v) for i, v in enumerate(values) if v is not None]
    styles: list[tuple[str, bool, bool]] = []
    sota_idx: int | None = None
    second_idx: int | None = None
    if numeric:
        sorted_vals = sorted(numeric, key=lambda x: x[1], reverse=True)
        sota_idx = sorted_vals[0][0]
        if len(sorted_vals) > 1 and sorted_vals[1][1] != sorted_vals[0][1]:
            second_idx = sorted_vals[1][0]
    for i, v in enumerate(values):
        is_sota = i == sota_idx
        is_second = i == second_idx
        styles.append((fmt_value(v), is_sota, is_second))
    return styles


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def split_category(title: str) -> list[str]:
    words = title.split()
    if len(words) == 1:
        return [title]
    if len(words) == 2:
        return words
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def category_label_svg(title: str, y_center: float) -> str:
    lines = split_category(title)
    line_h = 26
    total_h = line_h * len(lines)
    y0 = y_center - total_h / 2 + line_h / 2
    if len(lines) == 1:
        return (
            f'<text x="{CAT_COL_CENTER:.1f}" y="{y_center:.1f}" font-family="{FONT}" '
            f'font-size="21" font-weight="800" fill="{COLORS["section_title"]}" '
            f'text-anchor="middle" dominant-baseline="middle">{esc(lines[0])}</text>'
        )
    parts = []
    for i, line in enumerate(lines):
        yy = y0 + i * line_h
        parts.append(
            f'<text x="{CAT_COL_CENTER:.1f}" y="{yy:.1f}" font-family="{FONT}" '
            f'font-size="21" font-weight="800" fill="{COLORS["section_title"]}" '
            f'text-anchor="middle" dominant-baseline="middle">{esc(line)}</text>'
        )
    return "\n".join(parts)


def multiline_header(name: str, x: float, y: float, *, highlight: bool) -> str:
    lines = name.split("\n")
    if len(lines) == 1:
        fill = COLORS["highlight_header"] if highlight else COLORS["header"]
        weight = "900" if highlight else "700"
        font = BOLD_FONT if highlight else FONT
        stroke = (
            f' stroke="{fill}" stroke-width="0.45" paint-order="stroke fill"'
            if highlight
            else ""
        )
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="23" '
            f'font-weight="{weight}" fill="{fill}"{stroke} text-anchor="middle" '
            f'dominant-baseline="central" alignment-baseline="central">{esc(lines[0])}</text>'
        )
    y0 = y - 12
    y1 = y + 12
    fill = COLORS["highlight_header"] if highlight else COLORS["header"]
    parts = []
    for i, line in enumerate(lines):
        yy = y0 if i == 0 else y1
        parts.append(
            f'<text x="{x:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="23" '
            f'font-weight="700" fill="{fill}" text-anchor="middle" '
            f'dominant-baseline="central" alignment-baseline="central">{esc(line)}</text>'
        )
    return "\n".join(parts)


def cell_svg(
    x: float,
    y: float,
    text: str,
    *,
    col_idx: int,
    is_sota: bool,
    is_second: bool,
) -> str:
    highlight_col = col_idx == HIGHLIGHT_COL_IDX
    weight = "900" if highlight_col else "500"
    font = BOLD_FONT if highlight_col else FONT
    fill = COLORS["sota"] if is_sota else COLORS["text"]
    stroke = (
        f' stroke="{fill}" stroke-width="0.35" paint-order="stroke fill"'
        if highlight_col
        else ""
    )
    parts = [
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="25" '
        f'font-weight="{weight}" fill="{fill}"{stroke} text-anchor="middle" '
        f'dominant-baseline="central" alignment-baseline="central">{esc(text)}</text>'
    ]
    if is_second:
        w = max(28, len(text) * 11)
        x1, x2 = x - w / 2, x + w / 2
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y + 19:.1f}" x2="{x2:.1f}" y2="{y + 19:.1f}" '
            f'stroke="{COLORS["text"]}" stroke-width="2"/>'
        )
    return "\n".join(parts)


def build_svg() -> str:
    row_h = 52
    section_pad = 14
    header_y = MARGIN_Y + CARD_PAD + 88
    content_top = header_y + 52
    row_line_x2 = W - MARGIN_X - ROW_LINE_X2_OFFSET

    n_rows = sum(len(rows) for _, rows in SECTIONS)
    content_h = n_rows * row_h + len(SECTIONS) * section_pad * 2 + 40
    card_h = content_h + 200
    total_h = card_h + MARGIN_Y * 2

    y = content_top
    body: list[str] = []

    card_x, card_y = MARGIN_X, MARGIN_Y
    card_w = W - 2 * MARGIN_X
    highlight_top = card_y + CARD_PAD - 8
    highlight_h = card_h - 2 * CARD_PAD + 16
    highlight_cx = COL_X[HIGHLIGHT_COL_IDX]

    body.append(
        f'<rect width="{W}" height="{total_h}" fill="{COLORS["bg"]}"/>'
    )
    body.append(
        f'<rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="20" '
        f'fill="{COLORS["card"]}" stroke="{COLORS["border"]}" filter="url(#cardShadow)"/>'
    )
    body.append(
        f'<rect x="{CAT_COL_LEFT:.1f}" y="{header_y - 46:.1f}" '
        f'width="{row_line_x2 - CAT_COL_LEFT:.1f}" height="{content_top - header_y + 38:.1f}" '
        f'rx="16" fill="{COLORS["header_band"]}" opacity="0.78"/>'
    )
    body.append(
        f'<rect x="{HIGHLIGHT_X:.1f}" y="{highlight_top:.1f}" width="{HIGHLIGHT_W}" '
        f'height="{highlight_h:.1f}" rx="20" fill="url(#highlightFill)" '
        f'stroke="{COLORS["highlight_stroke"]}" stroke-width="2.2" filter="url(#columnGlow)"/>'
    )
    body.append(
        f'<polygon points="{highlight_cx:.1f},{highlight_top + 45:.1f} '
        f'{highlight_cx - 10.6:.1f},{highlight_top + 51.7:.1f} '
        f'{highlight_cx - 14.5:.1f},{highlight_top + 70:.1f} '
        f'{highlight_cx - 18.4:.1f},{highlight_top + 51.7:.1f} '
        f'{highlight_cx - 29:.1f},{highlight_top + 45:.1f} '
        f'{highlight_cx - 18.4:.1f},{highlight_top + 38.3:.1f} '
        f'{highlight_cx - 14.5:.1f},{highlight_top + 20:.1f} '
        f'{highlight_cx - 10.6:.1f},{highlight_top + 38.3:.1f}" '
        f'fill="{COLORS["accent"]}" stroke="#ffffff" stroke-width="2"/>'
    )
    star_y = highlight_top + highlight_h - 52
    body.append(
        f'<polygon points="{highlight_cx + 27:.1f},{star_y:.1f} '
        f'{highlight_cx + 19:.1f},{star_y + 5.1:.1f} '
        f'{highlight_cx + 16:.1f},{star_y + 19:.1f} '
        f'{highlight_cx + 13:.1f},{star_y + 5.1:.1f} '
        f'{highlight_cx + 5:.1f},{star_y:.1f} '
        f'{highlight_cx + 13:.1f},{star_y - 5.1:.1f} '
        f'{highlight_cx + 16:.1f},{star_y - 19:.1f} '
        f'{highlight_cx + 19:.1f},{star_y - 5.1:.1f}" '
        f'fill="{COLORS["accent_light"]}" stroke="#ffffff" stroke-width="1.8"/>'
    )

    # vertical divider: category | benchmark
    body.append(
        f'<line x1="{CAT_COL_RIGHT:.1f}" y1="{content_top - 8:.1f}" '
        f'x2="{CAT_COL_RIGHT:.1f}" y2="{content_top + content_h:.1f}" '
        f'stroke="{COLORS["section_line"]}" stroke-width="1.2" opacity="0.45"/>'
    )

    # header separator
    body.append(
        f'<line x1="{CAT_COL_LEFT:.1f}" y1="{content_top - 8:.1f}" '
        f'x2="{row_line_x2:.1f}" y2="{content_top - 8:.1f}" '
        f'stroke="{COLORS["section_line"]}" stroke-width="1.8"/>'
    )
    body.append(
        f'<text x="{BENCH_X:.1f}" y="{header_y:.1f}" font-family="{FONT}" '
        f'font-size="24" font-weight="800" fill="{COLORS["header"]}" '
        f'text-anchor="start" dominant-baseline="middle">Benchmark</text>'
    )

    for i, (name, is_primary) in enumerate(MODELS):
        body.append(multiline_header(name, COL_X[i], header_y, highlight=is_primary))

    for sec_idx, (sec_title, rows) in enumerate(SECTIONS):
        section_y0 = y
        section_y1 = y + section_pad * 2 + len(rows) * row_h
        body.append(
            f'<rect x="{CAT_COL_LEFT:.1f}" y="{section_y0 + 6:.1f}" '
            f'width="{CAT_COL_RIGHT - CAT_COL_LEFT - 14:.1f}" '
            f'height="{section_y1 - section_y0 - 12:.1f}" rx="18" '
            f'fill="{COLORS["category_fill"]}" stroke="{COLORS["category_border"]}" '
            f'stroke-width="1.2"/>'
        )
        if sec_idx > 0:
            body.append(
                f'<line x1="{CAT_COL_LEFT:.1f}" y1="{y:.1f}" x2="{row_line_x2:.1f}" '
                f'y2="{y:.1f}" stroke="{COLORS["section_line"]}" stroke-width="1.8"/>'
            )

        y += section_pad
        for row_idx, (bench, values) in enumerate(rows):
            row_y = y + row_h / 2
            body.append(
                f'<text x="{BENCH_X:.1f}" y="{row_y:.1f}" font-family="{FONT}" '
                f'font-size="23" font-weight="600" fill="{COLORS["label"]}" '
                f'text-anchor="start" dominant-baseline="central" '
                f'alignment-baseline="central">{esc(bench)}</text>'
            )
            styles = rank_styles(values)
            for col_idx, (text, is_sota, is_second) in enumerate(styles):
                body.append(
                    cell_svg(
                        COL_X[col_idx],
                        row_y,
                        text,
                        col_idx=col_idx,
                        is_sota=is_sota,
                        is_second=is_second,
                    )
                )
            y += row_h
            body.append(
                f'<line x1="{ROW_LINE_X1:.1f}" y1="{y:.1f}" x2="{row_line_x2:.1f}" '
                f'y2="{y:.1f}" stroke="{COLORS["row_line"]}" stroke-width="1"/>'
            )

        y += section_pad
        body.append(category_label_svg(sec_title, (section_y0 + section_y1) / 2))

    for glow_width, glow_opacity in ((18, 0.16), (11, 0.22), (6, 0.28)):
        body.append(
            f'<rect x="{HIGHLIGHT_X:.1f}" y="{highlight_top:.1f}" width="{HIGHLIGHT_W}" '
            f'height="{highlight_h:.1f}" rx="20" fill="none" '
            f'stroke="{COLORS["accent"]}" stroke-width="{glow_width}" '
            f'opacity="{glow_opacity}"/>'
        )
    body.append(
        f'<rect x="{HIGHLIGHT_X:.1f}" y="{highlight_top:.1f}" width="{HIGHLIGHT_W}" '
        f'height="{highlight_h:.1f}" rx="20" fill="none" '
        f'stroke="{COLORS["highlight_stroke"]}" stroke-width="2.4"/>'
    )

    footer_y = card_y + card_h - 28
    body.append(
        f'<line x1="{ROW_LINE_X1:.1f}" y1="{footer_y - 18:.1f}" x2="{row_line_x2:.1f}" '
        f'y2="{footer_y - 18:.1f}" stroke="{COLORS["footer_line"]}" stroke-width="1.5"/>'
    )
    note = (
        "Note: Bold + red = SOTA among selected models. Underline = 2nd best. "
        "InternVideo3 column highlighted."
    )
    body.append(
        f'<text x="{row_line_x2:.1f}" y="{footer_y:.1f}" font-family="{FONT}" '
        f'font-size="18" font-weight="400" fill="{COLORS["footer"]}" text-anchor="end" '
        f'dominant-baseline="middle">{esc(note)}</text>'
    )

    defs = f"""<defs>
<filter id="cardShadow" x="-8%" y="-8%" width="116%" height="116%">
  <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#21352f" flood-opacity="0.18"/>
</filter>
<filter id="columnGlow" x="-45%" y="-18%" width="190%" height="136%">
  <feGaussianBlur stdDeviation="18" result="blur"/>
  <feColorMatrix in="blur" type="matrix" values="0 0 0 0 0.23 0 0 0 0 0.78 0 0 0 0 0.62 0 0 0 0.72 0" result="glow"/>
  <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<linearGradient id="highlightFill" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0%" stop-color="{COLORS['highlight_fill_top']}"/>
  <stop offset="100%" stop-color="{COLORS['highlight_fill_bot']}"/>
</linearGradient>
</defs>"""

    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{int(total_h)}" '
        f'viewBox="0 0 {W} {int(total_h)}">\n{defs}\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def svg_to_png(svg_path: Path, png_path: Path) -> None:
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=W * 2,
        )
        return
    except (ImportError, OSError):
        pass

    try:
        from playwright.sync_api import sync_playwright

        svg_uri = svg_path.resolve().as_uri()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": 2000})
            page.goto(svg_uri)
            page.wait_for_timeout(300)
            height = page.evaluate(
                "() => document.documentElement.getBoundingClientRect().height"
            )
            page.set_viewport_size({"width": W, "height": math.ceil(height)})
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        return
    except Exception:
        pass

    try:
        import subprocess

        subprocess.run(
            [
                "rsvg-convert",
                "-w",
                str(W * 2),
                str(svg_path),
                "-o",
                str(png_path),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        import subprocess

        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            check=True,
            capture_output=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # macOS Quick Look thumbnail export
    try:
        import subprocess

        out_dir = png_path.parent
        subprocess.run(
            ["qlmanage", "-t", "-s", "3200", "-o", str(out_dir), str(svg_path)],
            check=True,
            capture_output=True,
        )
        thumb = out_dir / f"{svg_path.name}.png"
        if thumb.exists():
            thumb.replace(png_path)
            return
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"PNG export skipped ({exc}). SVG is available at {svg_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fig_dir = root / "assets"
    fig_dir.mkdir(parents=True, exist_ok=True)

    svg_content = build_svg()
    svg_path = fig_dir / "benchmark_comparison_card.svg"
    png_path = fig_dir / "benchmark_comparison_card.png"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Wrote {svg_path}")
    svg_to_png(svg_path, png_path)
    if png_path.exists():
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
