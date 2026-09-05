"""
Data Morph — the teacher data-collection loop, annotated.

Renders one wide standalone diagram (transparent background) for the article's
"Data Set" section:

  data_collection_loop.png
      corpus case → [1 extract envelope][2 Opus writes script][3 sandbox][4 validate]
                  → ✓ accept → data/interim   (pass)
                  ↘ ✗ feedback → retry ≤ 3 back to the teacher (fail)

Mirrors datamorph/data/collect.py::collect_case — the teach→run→verify→retry loop,
the accept rule (FV/SC/LD = 1.0 and CA ≥ 0.95), and where accepted pairs land.
Same visual language as render_pipeline.py (palette, rounded cards, monospace,
dpi=200, transparent canvas).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------- palette
INK    = "#1e1e2e"
SUBTLE = "#6c6f85"
CARD   = "#ffffff"
EDGE   = "#cdd0da"
ARROW  = "#9ca0b0"
PILL   = "#eef0f7"
LOOP   = "#d20f39"   # red — the retry feedback path
PASS   = "#40a02b"   # green — accept

# per-stage accents (matching render_pipeline.py)
S1 = "#209fb5"   # sapphire — extractor
S2 = "#fe640b"   # peach    — teacher (Opus writes the script)
S3 = "#8839ef"   # mauve    — sandbox
S4 = "#40a02b"   # green    — validator
C_IN = "#1e1e2e"

# fonts
HEAD_FP  = FontProperties(family="monospace", size=21, weight="bold")
SUB_FP   = FontProperties(family="monospace", size=12)
TITLE_FP = FontProperties(family="monospace", size=13, weight="bold")
TAG_FP   = FontProperties(family="monospace", size=10, weight="bold")
DESC_FP  = FontProperties(family="monospace", size=11)
MOD_FP   = FontProperties(family="monospace", size=10)
BADGE_FP = FontProperties(family="monospace", size=14, weight="bold")
CHIP_FP  = FontProperties(family="monospace", size=12, weight="bold")
LOOP_FP  = FontProperties(family="monospace", size=11, weight="bold")
GATE_FP  = FontProperties(family="monospace", size=10.5, weight="bold")


def cw(size):  # monospace advance (inches) for a given font size
    return 0.602 * size / 72.0


# ----------------------------------------------------------------------------- geometry
M        = 0.40
CARD_W   = 3.05
CARD_H   = 2.70
GAP      = 0.78          # arrow space between cards
CHIP_W   = 1.98
HEAD_H, SUB_H = 0.52, 0.40
TOP_PAD, BOT_PAD = 0.40, 0.40
LOOP_BAND = 1.30        # space under the row for the retry loop
IPAD = 0.22

STAGES = [
    dict(accent=S1, num="1", title="EXTRACT", tag="deterministic",
         desc=["Build the metadata", "\"envelope\": schema,",
               "types, samples, warnings.", "The teacher sees ONLY this."],
         module="data/envelope.py"),
    dict(accent=S2, num="2", title="TEACHER · OPUS", tag="claude -p --model opus",
         desc=["Reads the envelope + task,", "writes  <analysis>  +  a",
               "Python  <script>.", "Retry feeds back the error."],
         module="data/teacher_script.py"),
    dict(accent=S3, num="3", title="SANDBOX", tag="isolated subprocess",
         desc=["Runs the script on the REAL", "file: timeout + CPU limit.",
               "Captures the converted", "output — or the error."],
         module="data/sandbox.py"),
    dict(accent=S4, num="4", title="VALIDATE", tag="4 metrics + accept gate",
         desc=["Score Format · Schema ·", "Loadability · Content.",
               "Accept when FV/SC/LD = 1.0", "and CA ≥ 0.95."],
         module="data/collect.py"),
]


# ----------------------------------------------------------------------------- helpers
def card(ax, x, ytop, s):
    ax.add_patch(FancyBboxPatch((x, ytop), CARD_W, CARD_H,
                 boxstyle="round,pad=0,rounding_size=0.16",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=3))
    # top accent strip
    ax.add_patch(FancyBboxPatch((x + 0.10, ytop + 0.10), CARD_W - 0.20, 0.12,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 linewidth=0, facecolor=s["accent"], zorder=4))
    tx = x + IPAD
    by = ytop + 0.52
    ax.add_patch(Circle((tx + 0.22, by), 0.23, facecolor=s["accent"],
                        edgecolor="none", zorder=4))
    ax.text(tx + 0.22, by, s["num"], fontproperties=BADGE_FP, color=CARD,
            ha="center", va="center", zorder=5)
    ax.text(tx + 0.58, by, s["title"], fontproperties=TITLE_FP, color=INK,
            ha="left", va="center", zorder=5)
    ax.text(tx, ytop + 0.92, s["tag"], fontproperties=TAG_FP, color=s["accent"],
            ha="left", va="center", zorder=5)
    dy = ytop + 1.22
    for d in s["desc"]:
        ax.text(tx, dy, d, fontproperties=DESC_FP, color=SUBTLE,
                ha="left", va="center", zorder=5)
        dy += 0.30
    mw = len(s["module"]) * cw(10) + 0.40
    my = ytop + CARD_H - IPAD - 0.34
    ax.add_patch(FancyBboxPatch((tx, my), mw, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.10",
                 linewidth=0, facecolor=PILL, zorder=4))
    ax.text(tx + 0.20, my + 0.17, s["module"], fontproperties=MOD_FP, color=SUBTLE,
            ha="left", va="center", zorder=5)


def chip(ax, x, ymid, lines, fill, txtcolor=CARD, w=CHIP_W, h=1.55):
    ax.add_patch(FancyBboxPatch((x, ymid - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.22",
                 linewidth=0, facecolor=fill, zorder=3))
    cx = x + w / 2
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(cx, ymid + (i - (n - 1) / 2) * 0.34, ln, fontproperties=CHIP_FP,
                color=txtcolor, ha="center", va="center", zorder=4)


def harrow(ax, x1, x2, y, color=ARROW):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.2, color=color, zorder=2))


# ----------------------------------------------------------------------------- compose
def main():
    n = len(STAGES)
    out_chip_w = 2.55
    # x left-edges
    x_in = M
    x0 = x_in + CHIP_W + GAP                         # first card
    xs = [x0 + i * (CARD_W + GAP) for i in range(n)]
    x_out = xs[-1] + CARD_W + GAP                     # accept chip
    W = x_out + out_chip_w + M

    Y0 = TOP_PAD + HEAD_H + SUB_H + 0.34
    row_top = Y0
    y_mid = row_top + CARD_H / 2
    H = row_top + CARD_H + LOOP_BAND + BOT_PAD

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis("off")

    cxall = W / 2
    ax.text(cxall, TOP_PAD + HEAD_H / 2,
            "data morph — teacher data-collection loop",
            fontproperties=HEAD_FP, color=INK, ha="center", va="center")
    ax.text(cxall, TOP_PAD + HEAD_H + SUB_H / 2,
            "Each corpus case is taught, run, and verified — accepted only when it passes; "
            "otherwise the error is fed back and retried ≤ 3",
            fontproperties=SUB_FP, color=SUBTLE, ha="center", va="center")

    # input chip (the generated corpus case)
    chip(ax, x_in, y_mid, ["CORPUS CASE", "input + expected", "(seeded generator)"], C_IN)

    # stage cards
    for i, s in enumerate(STAGES):
        card(ax, xs[i], row_top, s)

    # arrows along the row
    harrow(ax, x_in + CHIP_W, x0, y_mid)
    for i in range(n - 1):
        harrow(ax, xs[i] + CARD_W, xs[i + 1], y_mid)
    # validate → accept (green, labeled ✓ pass)
    harrow(ax, xs[-1] + CARD_W, x_out, y_mid, color=PASS)
    ax.text((xs[-1] + CARD_W + x_out) / 2, y_mid - 0.22, "✓ pass",
            fontproperties=GATE_FP, color=PASS, ha="center", va="center", zorder=5)

    # accept output chip → data/interim
    chip(ax, x_out, y_mid, ["✓ ACCEPT", "→ data/interim/", "envelope+script+scores"],
         PASS, w=out_chip_w)

    # retry loop: validator → back to teacher, routed below the row
    yL = row_top + CARD_H + 0.62
    c_vl = xs[-1] + CARD_W / 2          # center of VALIDATE
    c_te = xs[1] + CARD_W / 2           # center of TEACHER
    ax.plot([c_vl, c_vl], [row_top + CARD_H, yL], color=LOOP, linewidth=2.2, zorder=6)
    ax.plot([c_vl, c_te], [yL, yL], color=LOOP, linewidth=2.2, zorder=6)
    ax.add_patch(FancyArrowPatch((c_te, yL), (c_te, row_top + CARD_H),
                 arrowstyle="-|>", mutation_scale=16, linewidth=2.2,
                 color=LOOP, zorder=6))
    ax.text((c_te + c_vl) / 2, yL + 0.26,
            "✗ script error / low score → feedback → retry ≤ 3",
            fontproperties=LOOP_FP, color=LOOP, ha="center", va="center", zorder=6)

    out = Path(__file__).resolve().parents[1] / "data_collection_loop.png"
    fig.savefig(str(out), dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
