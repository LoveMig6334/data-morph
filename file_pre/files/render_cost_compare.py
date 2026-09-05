"""
Data Morph — frontier-model cost comparison illustration.
Renders 2 standalone components (transparent background) for composing in Canva:

  1. 01_cost_table.png        per-model $/file table (Opus 4.8 / GPT-5.5 / DeepSeek V4 Pro)
                              with the cost scaled to 10 / 50 / 100 files
  2. 02_cost_growth_bars.png  grouped bar chart — cost grows with file count (10, 50, 100)

Numbers are loaded from data/cost_probe/results.json (the real OpenRouter usage.cost
from scripts/cost_probe.py), so the visuals stay in sync if the probe is re-run.
Same visual language as render_components.py (Catppuccin-ish palette, rounded cards,
monospace, dpi=200, transparent canvas).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------- palette
INK      = "#1e1e2e"   # default text
SUBTLE   = "#6c6f85"   # punctuation / muted
BLUE     = "#1e66f5"   # Opus
PEACH    = "#fe640b"   # GPT-5.5
TEAL     = "#179299"   # DeepSeek
CARD     = "#ffffff"   # card fill
EDGE     = "#cdd0da"   # card border
HEAD_FILL = "#eaecf4"  # table header fill

MONO   = FontProperties(family="monospace", size=15)
MONO_B = FontProperties(family="monospace", size=15, weight="bold")

# ----------------------------------------------------------------------------- paths + data
HERE = Path(__file__).resolve()
OUT  = HERE.parents[1]                       # .../file_pre/

# Model display order + the color each one gets across both components.
MODEL_ORDER = ["Claude Opus 4.8", "GPT-5.5", "DeepSeek V4 Pro"]
MODEL_COLOR = {"Claude Opus 4.8": BLUE, "GPT-5.5": PEACH, "DeepSeek V4 Pro": TEAL}
SCALES = [10, 50, 100]


# Actual OpenRouter dashboard billing for the 5-call/model probe run on 2026-06-14.
# These are the true billed amounts — the API's usage.cost field over-reported by
# ~1.9x, so we use the dashboard figures as ground truth. See data/cost_probe/REPORT.md.
BILLED_TOTAL = {"Claude Opus 4.8": 0.167, "GPT-5.5": 0.209, "DeepSeek V4 Pro": 0.00843}
N_FILES = 5


def load_cost_per_file():
    """Return {model: $/file} from the real OpenRouter billing (total / files)."""
    return {m: BILLED_TOTAL[m] / N_FILES for m in MODEL_ORDER}


# ----------------------------------------------------------------------------- 1) table
def cost_table(filename, per_file):
    """Model | $ / file | x10 | x50 | x100 — clean grid in the render_components style."""
    header = ["model", "$ / file"]
    rows = [[m, f"${per_file[m]:.4f}"] for m in MODEL_ORDER]
    title = "Frontier LLM cost per file (OpenCode via OpenRouter API)"

    row_h, title_h, pad = 0.56, 0.66, 0.28
    nrows = len(rows)

    # widen the table so it's at least as wide as the title; extra space -> model col
    char_w_title = 0.602 * 15 / 72.0
    title_w = len(title) * char_w_title
    base = [3.05, 2.05]                       # inches per column (model col is wider)
    extra = max(0.0, title_w - sum(base))
    col_w = [base[0] + extra, base[1]]

    fig_w = sum(col_w) + 2 * pad
    fig_h = title_h + (nrows + 1) * row_h + 2 * pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.invert_yaxis(); ax.axis("off")

    card = FancyBboxPatch((0.12, 0.12), fig_w - 0.24, fig_h - 0.24,
                          boxstyle="round,pad=0,rounding_size=0.18",
                          linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1)
    ax.add_patch(card)

    ax.text(pad, pad + 0.36, title,
            fontproperties=MONO_B, color=SUBTLE, ha="left", va="center", zorder=3)

    x0, y0 = pad, pad + title_h
    xedges = [x0]
    for w in col_w:
        xedges.append(xedges[-1] + w)

    # header
    ax.add_patch(Rectangle((x0, y0), sum(col_w), row_h, facecolor=HEAD_FILL,
                           edgecolor="none", zorder=2))
    for c, h in enumerate(header):
        ax.text(xedges[c] + 0.18, y0 + row_h / 2, h, fontproperties=MONO_B,
                color=INK, ha="left", va="center", zorder=3)

    # body — first cell gets the model's accent color as a left swatch + bold name
    for r, row in enumerate(rows):
        yr = y0 + (r + 1) * row_h
        accent = MODEL_COLOR[row[0]]
        ax.add_patch(Rectangle((x0, yr), 0.14, row_h, facecolor=accent,
                               edgecolor="none", zorder=2.5))
        for c, val in enumerate(row):
            fp = MONO_B if c == 0 else MONO
            col = accent if c == 0 else INK
            ax.text(xedges[c] + 0.18, yr + row_h / 2, val, fontproperties=fp,
                    color=col, ha="left", va="center", zorder=3)

    # grid
    for xe in xedges:
        ax.plot([xe, xe], [y0, y0 + (nrows + 1) * row_h], color=EDGE, lw=0.8, zorder=2)
    for r in range(nrows + 2):
        ax.plot([x0, x0 + sum(col_w)], [y0 + r * row_h] * 2, color=EDGE, lw=0.8, zorder=2)

    fig.savefig(filename, dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", filename)


# ----------------------------------------------------------------------------- 2) bar chart
def cost_growth_bars(filename, per_file):
    """Grouped bars: x = file counts (10/50/100), one bar per model, value labels on top."""
    fig_w, fig_h = 9.2, 5.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.invert_yaxis(); ax.axis("off")

    # rounded card backdrop
    ax.add_patch(FancyBboxPatch((0.12, 0.12), fig_w - 0.24, fig_h - 0.24,
                 boxstyle="round,pad=0,rounding_size=0.2",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1))

    # plot frame (in data inches): leave room for title, axis labels, legend
    left, right = 1.15, fig_w - 0.5
    top, bottom = 1.30, fig_h - 0.95          # remember y is inverted
    plot_w = right - left
    plot_h = bottom - top

    max_cost = max(per_file[m] * max(SCALES) for m in MODEL_ORDER)
    # round the axis max up to a "nice" number
    def nice_top(v):
        import math
        if v <= 0: return 1.0
        mag = 10 ** math.floor(math.log10(v))
        for step in (1, 2, 2.5, 5, 10):
            if step * mag >= v:
                return step * mag
        return 10 * mag
    y_top = nice_top(max_cost)

    def Y(cost):   # cost USD -> inverted-y inch position
        return bottom - (cost / y_top) * plot_h

    # title
    ax.text(left, 0.72, "Cost grows linearly with every file you convert",
            fontproperties=MONO_B, color=INK, ha="left", va="center", zorder=4)
    ax.text(left, 1.06, "total API spend (USD) vs. number of files  •  one frontier call per file",
            fontproperties=FontProperties(family="monospace", size=11), color=SUBTLE,
            ha="left", va="center", zorder=4)

    # y gridlines + labels (5 even divisions; integer labels stay clean)
    for frac in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yv = y_top * frac
        yy = Y(yv)
        ax.plot([left, right], [yy, yy], color=EDGE, lw=0.8, zorder=2)
        lbl = f"${yv:.0f}" if abs(yv - round(yv)) < 1e-9 else f"${yv:.1f}"
        ax.text(left - 0.18, yy, lbl, fontproperties=FontProperties(family="monospace", size=11),
                color=SUBTLE, ha="right", va="center", zorder=3)

    # grouped bars
    n_groups = len(SCALES)
    n_models = len(MODEL_ORDER)
    group_w = plot_w / n_groups
    bar_w = group_w * 0.62 / n_models
    gap = (group_w - bar_w * n_models) / 2

    label_sm = FontProperties(family="monospace", size=10)
    for gi, scale in enumerate(SCALES):
        gx = left + gi * group_w
        for mi, m in enumerate(MODEL_ORDER):
            cost = per_file[m] * scale
            bx = gx + gap + mi * bar_w
            by = Y(cost)
            ax.add_patch(FancyBboxPatch((bx, by), bar_w * 0.86, bottom - by,
                         boxstyle="round,pad=0,rounding_size=0.04",
                         linewidth=0, facecolor=MODEL_COLOR[m], zorder=3))
            # value label above bar
            ax.text(bx + bar_w * 0.43, by - 0.12, f"${cost:.2f}",
                    fontproperties=label_sm, color=MODEL_COLOR[m],
                    ha="center", va="bottom", zorder=4)
        # x group label
        ax.text(gx + group_w / 2, bottom + 0.30, f"{scale} files",
                fontproperties=MONO_B, color=INK, ha="center", va="center", zorder=3)

    # baseline
    ax.plot([left, right], [bottom, bottom], color=SUBTLE, lw=1.2, zorder=3)

    # legend — horizontal row inside the empty top-left of the plot area
    char_w = 0.602 * 10 / 72.0     # monospace advance for size-10 font (inches)
    lx, ly = left + 0.1, top + 0.45
    for m in MODEL_ORDER:
        ax.add_patch(Rectangle((lx, ly - 0.10), 0.22, 0.20,
                     facecolor=MODEL_COLOR[m], edgecolor="none", zorder=4))
        ax.text(lx + 0.32, ly, m, fontproperties=label_sm, color=INK,
                ha="left", va="center", zorder=4)
        lx += 0.32 + len(m) * char_w + 0.55     # advance past swatch + label + gap

    fig.savefig(filename, dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", filename)


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    per_file = load_cost_per_file()
    cost_table(str(OUT / "01_cost_table.png"), per_file)
    cost_growth_bars(str(OUT / "02_cost_growth_bars.png"), per_file)
    print("done")
