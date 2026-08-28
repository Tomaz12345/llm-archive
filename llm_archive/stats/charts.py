"""Inline SVG chart rendering — no JavaScript, no chart library.

Colors come from the validated categorical palette; the slots used here passed the
six checks against this app's own surfaces (light #ffffff, dark #141b1d) rather than
the reference defaults. Light mode raised one WARN — aqua, yellow and magenta sit
below 3:1 on white — which obligates relief: every chart here ships a value legend,
and the dashboard carries a full data table. That is not decoration, it is the
condition under which those hues are allowed.

Mark rules followed: thin marks, 2px surface gaps between stacked segments and
adjacent bars, rounded data-ends, recessive grid and axes, text in ink tokens
rather than series colors, `<title>` on every mark so hover works without script.

Colors are emitted as CSS custom properties (--series-N) so light and dark each get
their own validated step, chosen in app.css.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

SERIES_SLOTS = 8          # fixed order, never cycled
GAP = 2                   # surface gap between adjacent/stacked marks
RADIUS = 4                # rounded data-end


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def series_var(index: int) -> str:
    """Fixed-order slot. A 9th series folds into 'Other' rather than inventing a hue."""
    return f"var(--series-{min(index, SERIES_SLOTS - 1) + 1})"


@dataclass
class Legend:
    label: str
    value: str
    index: int


def legend_html(entries: list[Legend]) -> str:
    """Always present for >=2 series — identity is never carried by color alone."""
    items = "".join(
        f'<li><span class="swatch" style="background:{series_var(e.index)}"></span>'
        f'<span class="lbl">{_esc(e.label)}</span>'
        f'<b>{_esc(e.value)}</b></li>'
        for e in entries)
    return f'<ul class="legend">{items}</ul>'


# --------------------------------------------------------------------------

def stacked_bars(categories: list[str], series: dict[str, list[int]],
                 height: int = 190, label_every: int = 3) -> str:
    """Monthly volume, stacked by source."""
    if not categories:
        return '<p class="nodata">No dated sessions yet.</p>'

    names = list(series)
    totals = [sum(series[n][i] for n in names) for i in range(len(categories))]
    peak = max(totals) or 1

    width = max(560, len(categories) * 34)
    pad_l, pad_b, pad_t = 36, 22, 8
    plot_h = height - pad_b - pad_t
    band = (width - pad_l) / len(categories)
    bar_w = max(6, band - 8)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'preserveAspectRatio="xMinYMid meet" role="img">']

    # recessive gridlines, labelled
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" '
                     f'x2="{width}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" '
                     f'text-anchor="end">{int(peak * frac)}</text>')

    for i, cat in enumerate(categories):
        x = pad_l + i * band + (band - bar_w) / 2
        cursor = pad_t + plot_h
        for s_i, name in enumerate(names):
            value = series[name][i]
            if not value:
                continue
            seg_h = plot_h * value / peak
            seg_h = max(seg_h - GAP, 1)          # 2px surface gap between segments
            cursor -= seg_h
            top = s_i == len(names) - 1 or all(
                series[n][i] == 0 for n in names[s_i + 1:])
            parts.append(
                f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" '
                f'height="{seg_h:.1f}" fill="{series_var(s_i)}" '
                f'rx="{RADIUS if top else 0}"><title>{_esc(cat)} · '
                f'{_esc(name)}: {value}</title></rect>')
            cursor -= GAP
        if i % label_every == 0:
            parts.append(f'<text class="tick" x="{x + bar_w / 2:.1f}" '
                         f'y="{height - 6}" text-anchor="middle">'
                         f'{_esc(cat[2:])}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _nice_max(peak: float) -> int:
    """Axis ceiling: a clean number a little above the peak.

    Two jobs at once — ticks land on numbers a reader recognises (12,000, not
    11,348), and the headroom keeps the final point off the top edge so its
    direct label has somewhere to sit.
    """
    if peak <= 0:
        return 1
    mag = 10 ** math.floor(math.log10(peak))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if peak * 1.04 <= mag * m:
            return int(mag * m)
    return int(mag * 10)


def line_chart(categories: list[str], values: list[int], height: int = 164) -> str:
    """Cumulative growth. One series, so no legend box — the title names it."""
    if not values:
        return '<p class="nodata">Nothing to plot yet.</p>'
    width = max(560, len(categories) * 34)
    # pad_r reserves room for the end dot and its 2px surface ring; pad_t for the
    # end label. Nothing is drawn past the viewBox where it would be clipped.
    pad_l, pad_r, pad_b, pad_t = 48, 14, 24, 18
    plot_h = height - pad_b - pad_t
    base_y = pad_t + plot_h
    top = _nice_max(max(values))
    step = (width - pad_l - pad_r) / max(len(values) - 1, 1)

    points = [(pad_l + i * step, base_y - plot_h * v / top)
              for i, v in enumerate(values)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(points))
    area = (f"M{points[0][0]:.1f},{base_y:.1f} "
            + " ".join(f"L{x:.1f},{y:.1f}" for x, y in points)
            + f" L{points[-1][0]:.1f},{base_y:.1f} Z")

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'preserveAspectRatio="xMinYMid meet" role="img">',
             '<defs><linearGradient id="areaFade" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0" stop-color="var(--series-1)" stop-opacity=".22"/>'
             '<stop offset="1" stop-color="var(--series-1)" stop-opacity="0"/>'
             '</linearGradient></defs>']

    for frac in (0.0, 0.5, 1.0):
        y = base_y - plot_h * frac
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" '
                     f'x2="{width - pad_r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end">{int(top * frac):,}</text>')

    # Hover column per month: still no script, but every point has its number.
    for i, (x, _) in enumerate(points):
        left = max(pad_l, x - step / 2)
        right = min(width - pad_r, x + step / 2)
        label = _esc(categories[i]) if i < len(categories) else ""
        parts.append(f'<rect class="hit" x="{left:.1f}" y="{pad_t}" '
                     f'width="{max(right - left, 1):.1f}" height="{plot_h}">'
                     f'<title>{label}: {values[i]:,} messages</title></rect>')

    parts.append(f'<path d="{area}" class="area"/>')
    parts.append(f'<path d="{path}" class="spark"/>')

    # emphasised endpoint + one direct label, rather than a number on every point
    ex, ey = points[-1]
    parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" class="endpoint"/>')
    parts.append(f'<text class="endlabel" x="{ex - 9:.1f}" '
                 f'y="{max(ey - 10, pad_t - 4):.1f}" '
                 f'text-anchor="end">{values[-1]:,}</text>')

    last = len(categories) - 1
    for i, cat in enumerate(categories):
        # every third month, but never one crowding the right-hand label
        if i % 3 or last - i < 3:
            continue
        parts.append(f'<text class="tick" x="{points[i][0]:.1f}" y="{height - 7}" '
                     f'text-anchor="middle">{_esc(cat[2:])}</text>')
    if categories:
        parts.append(f'<text class="tick" x="{width}" y="{height - 7}" '
                     f'text-anchor="end">{_esc(categories[last][2:])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(grid: list[list[int]], labels: list[str], peak: int) -> str:
    """Weekday x hour. Sequential single hue, light->dark; zero recedes to surface."""
    if not peak:
        return '<p class="nodata">No timestamped messages.</p>'
    cell, gap = 21, 2
    pad_l, pad_t = 34, 16
    width = pad_l + 24 * cell
    height = pad_t + 7 * cell + 14

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart heat" '
             f'preserveAspectRatio="xMinYMid meet" role="img">']
    for hour in range(0, 24, 3):
        parts.append(f'<text class="tick" x="{pad_l + hour * cell + cell / 2:.1f}" '
                     f'y="{pad_t - 5}" text-anchor="middle">{hour:02d}</text>')
    for row, label in enumerate(labels):
        y = pad_t + row * cell
        parts.append(f'<text class="tick" x="{pad_l - 7}" y="{y + cell / 2 + 3:.1f}" '
                     f'text-anchor="end">{_esc(label)}</text>')
        for hour in range(24):
            value = grid[row][hour]
            # perceptual-ish easing so mid values stay distinguishable
            level = (value / peak) ** 0.55 if value else 0
            fill = (f"color-mix(in oklab, var(--heat-max) {level * 100:.0f}%, "
                    f"var(--heat-min))") if value else "var(--heat-empty)"
            parts.append(
                f'<rect x="{pad_l + hour * cell:.1f}" y="{y:.1f}" '
                f'width="{cell - gap}" height="{cell - gap}" rx="2" fill="{fill}">'
                f'<title>{_esc(label)} {hour:02d}:00 — {value} messages</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def stacked_hbars(categories: list[str], series: dict[str, list[int]],
                  bar_h: int = 16) -> str:
    """Horizontal bars split into per-category segments — e.g. calls by model."""
    if not categories:
        return '<p class="nodata">Nothing recorded.</p>'
    names = list(series)
    totals = [sum(series[n][i] for n in names) for i in range(len(categories))]
    peak = max(totals) or 1
    label_w, value_w = 168, 74
    width = 560
    track = width - label_w - value_w
    height = len(categories) * (bar_h + GAP + 4)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'preserveAspectRatio="xMinYMid meet" role="img">']
    for i, cat in enumerate(categories):
        y = i * (bar_h + GAP + 4)
        parts.append(f'<text class="blabel" x="0" y="{y + bar_h - 3}">'
                     f'{_esc(cat[:26])}</text>')
        last = max((s_i for s_i, n in enumerate(names) if series[n][i]), default=None)
        x = label_w
        for s_i, name in enumerate(names):
            value = series[name][i]
            if not value:
                continue
            seg_w = track * value / peak
            seg_w = max(seg_w - GAP, 1)
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_h}" '
                f'fill="{series_var(s_i)}" rx="{RADIUS if s_i == last else 0}">'
                f'<title>{_esc(cat)} · {_esc(name)}: {value}</title></rect>')
            x += seg_w + GAP
        parts.append(f'<text class="bvalue" x="{width}" y="{y + bar_h - 3}" '
                     f'text-anchor="end">{totals[i]:,}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hbars(items: list[tuple[str, float, str]], accent_index: int = 0,
          bar_h: int = 16) -> str:
    """Horizontal magnitude bars: (label, value, display) triples."""
    if not items:
        return '<p class="nodata">Nothing recorded.</p>'
    peak = max(v for _, v, _ in items) or 1
    label_w, value_w = 168, 74
    width = 560
    track = width - label_w - value_w
    height = len(items) * (bar_h + GAP + 4)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'preserveAspectRatio="xMinYMid meet" role="img">']
    for i, (label, value, display) in enumerate(items):
        y = i * (bar_h + GAP + 4)
        w = max(track * value / peak, 2)
        parts.append(f'<text class="blabel" x="0" y="{y + bar_h - 3}">'
                     f'{_esc(label[:26])}</text>')
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'rx="{RADIUS}" fill="{series_var(accent_index)}">'
            f'<title>{_esc(label)}: {_esc(display)}</title></rect>')
        parts.append(f'<text class="bvalue" x="{width}" y="{y + bar_h - 3}" '
                     f'text-anchor="end">{_esc(display)}</text>')
    parts.append("</svg>")
    return "".join(parts)
