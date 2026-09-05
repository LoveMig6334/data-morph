"""
Data Morph — per-stage mini diagrams (input → process → output), one PNG per stage.
Same visual language as render_pipeline.py / render_components.py (palette, rounded
cards, monospace, dpi=200, transparent). Each diagram is a horizontal 3-box flow:
  [ input pill ] → [ stage card (what it does) ] → [ output pill ]

Outputs (file_pre/): stage1_extractor.png ... stage5_validator.png
Used in the article's "Pipeline Design & Solution" section, one per numbered stage.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

INK, SUBTLE, CARD, EDGE = "#1e1e2e", "#6c6f85", "#ffffff", "#cdd0da"
ARROW, PILL = "#9ca0b0", "#eef0f7"
S1, S2, S3, S4, S5 = "#209fb5", "#9ca0b0", "#fe640b", "#8839ef", "#40a02b"

OUT = Path(__file__).resolve().parents[1]
HEAD = FontProperties(family="monospace", size=15, weight="bold")
TITLE = FontProperties(family="monospace", size=15, weight="bold")
DESC = FontProperties(family="monospace", size=11.5)
MOD = FontProperties(family="monospace", size=10)
BADGE = FontProperties(family="monospace", size=14, weight="bold")
PILLF = FontProperties(family="monospace", size=11.5, weight="bold")


def cw(s):
    return 0.602 * s / 72.0


STAGES = [
    dict(n="1", name="EXTRACTOR", accent=S1,
         inp=["source file", "CSV / JSON / TXT"], out=["metadata", "envelope"],
         desc=["Deterministic — no model.", "Sniff schema, types, samples,", "warnings from the raw file."],
         module="datamorph/extractor/"),
    dict(n="2", name="SUMMARIZER", accent=S2,
         inp=["metadata", "envelope"], out=["NL brief", "of the job"],
         desc=["Gemma 2B turns the envelope", "into a short natural-language", "brief.  (deferred)"],
         module="datamorph/models/"),
    dict(n="3", name="SCRIPT GENERATOR", accent=S3,
         inp=["envelope", "(+ brief)"], out=["convert.py", "Python script"],
         desc=["The model writes a short", "Python script. Opus teaches,", "Gemma writes at inference."],
         module="data/teacher_script.py"),
    dict(n="4", name="SANDBOX EXEC", accent=S4,
         inp=["script +", "full file"], out=["converted", "output / error"],
         desc=["Runs the script on the REAL", "file in an isolated subprocess", "(15s timeout + CPU limit)."],
         module="data/sandbox.py"),
    dict(n="5", name="VALIDATOR", accent=S5,
         inp=["converted", "output"], out=["accept", "or retry ≤ 3"],
         desc=["Scores 4 metrics; accept at", "FV/SC/LD=1.0 & CA≥0.95,", "else feed the error back."],
         module="evaluation/metrics.py"),
]

CARD_W, CARD_H = 3.55, 2.55
PILL_W, PILL_H = 2.05, 1.30
GAP = 0.66
M = 0.30
HEADER_H = 0.62


def pill(ax, x, ymid, lines, fill, txt):
    ax.add_patch(FancyBboxPatch((x, ymid - PILL_H / 2), PILL_W, PILL_H,
                 boxstyle="round,pad=0,rounding_size=0.18", linewidth=0, facecolor=fill, zorder=3))
    cx = x + PILL_W / 2
    for i, ln in enumerate(lines):
        ax.text(cx, ymid + (i - (len(lines) - 1) / 2) * 0.34, ln, fontproperties=PILLF,
                color=txt, ha="center", va="center", zorder=4)


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=18, linewidth=2.2, color=ARROW, zorder=2))


def stage_card(ax, x, ytop, s):
    acc = s["accent"]
    ax.add_patch(FancyBboxPatch((x, ytop), CARD_W, CARD_H, boxstyle="round,pad=0,rounding_size=0.16",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=3))
    ax.add_patch(FancyBboxPatch((x + 0.10, ytop + 0.10), CARD_W - 0.20, 0.12,
                 boxstyle="round,pad=0,rounding_size=0.05", linewidth=0, facecolor=acc, zorder=4))
    tx = x + 0.24
    by = ytop + 0.55
    ax.add_patch(Circle((tx + 0.22, by), 0.23, facecolor=acc, edgecolor="none", zorder=4))
    ax.text(tx + 0.22, by, s["n"], fontproperties=BADGE, color=CARD, ha="center", va="center", zorder=5)
    ax.text(tx + 0.60, by, s["name"], fontproperties=TITLE, color=INK, ha="left", va="center", zorder=5)
    dy = ytop + 1.02
    for d in s["desc"]:
        ax.text(tx, dy, d, fontproperties=DESC, color=SUBTLE, ha="left", va="center", zorder=5)
        dy += 0.32
    mw = len(s["module"]) * cw(10) + 0.40
    ax.add_patch(FancyBboxPatch((tx, ytop + CARD_H - 0.50), mw, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.10", linewidth=0, facecolor=PILL, zorder=4))
    ax.text(tx + 0.20, ytop + CARD_H - 0.33, s["module"], fontproperties=MOD, color=SUBTLE,
            ha="left", va="center", zorder=5)


def render(s, fname):
    W = M + PILL_W + GAP + CARD_W + GAP + PILL_W + M
    H = M + HEADER_H + CARD_H + M
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis(); ax.axis("off")

    ax.text(M, M + HEADER_H / 2, f"Stage {s['n']} — {s['name'].title()}",
            fontproperties=HEAD, color=s["accent"], ha="left", va="center")

    band_top = M + HEADER_H
    ymid = band_top + CARD_H / 2
    x = M
    pill(ax, x, ymid, s["inp"], PILL, INK)
    x += PILL_W
    arrow(ax, x, x + GAP, ymid)
    x += GAP
    stage_card(ax, x, band_top, s)
    x += CARD_W
    arrow(ax, x, x + GAP, ymid)
    x += GAP
    pill(ax, x, ymid, s["out"], s["accent"], CARD)

    out = OUT / fname
    fig.savefig(str(out), dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    names = ["stage1_extractor.png", "stage2_summarizer.png", "stage3_script_generator.png",
             "stage4_sandbox.png", "stage5_validator.png"]
    for s, fn in zip(STAGES, names):
        render(s, fn)
    print("done")
