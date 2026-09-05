"""
Data Morph — the 5-stage conversion pipeline, annotated.
Renders one tall standalone diagram (transparent background) for Canva/slides:

  06_pipeline.png   input → [1 extract][2 summarize][3 script-gen][4 sandbox][5 validate] → output
                    with the retry loop (fail → back to stage 3) drawn on the right.

Each stage card shows WHAT it does, the implementation module, and a short tag,
so a reader can follow how a messy file becomes a clean conversion. Same visual
language as render_components.py (palette, rounded cards, monospace, dpi=200,
transparent canvas). Details mirror datamorph/ (extractor, sandbox.py CPU/timeout
limits, evaluation/metrics.py, collect.py accept rule CA>=0.95 / retry<=3).
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
PILL   = "#eef0f7"   # faint module-pill fill

# per-stage accents (Catppuccin-latte family, matching the existing graphics)
S1 = "#209fb5"   # sapphire — extractor
S2 = "#9ca0b0"   # gray     — summarizer (deferred)
S3 = "#fe640b"   # peach    — script generation (the distilled skill)
S4 = "#8839ef"   # mauve    — sandbox
S5 = "#40a02b"   # green    — validator
C_IN  = "#1e1e2e"  # input chip (dark)
C_OUT = "#40a02b"  # output chip (green)

# fonts
HEAD_FP  = FontProperties(family="monospace", size=21, weight="bold")
SUB_FP   = FontProperties(family="monospace", size=12)
TITLE_FP = FontProperties(family="monospace", size=17, weight="bold")
TAG_FP   = FontProperties(family="monospace", size=11, weight="bold")
DESC_FP  = FontProperties(family="monospace", size=12.5)
MOD_FP   = FontProperties(family="monospace", size=11)
BADGE_FP = FontProperties(family="monospace", size=16, weight="bold")
CHIP_FP  = FontProperties(family="monospace", size=13, weight="bold")
LOOP_FP  = FontProperties(family="monospace", size=11, weight="bold")


def cw(size):  # monospace advance (inches) for a given font size
    return 0.602 * size / 72.0


# ----------------------------------------------------------------------------- geometry
W            = 9.0
LEFT         = 0.40
RIGHT_GUTTER = 1.05            # space on the right for the retry loop
CARD_X       = LEFT
CARD_W       = W - LEFT - RIGHT_GUTTER
CX           = CARD_X + CARD_W / 2     # horizontal center of cards/arrows

TOP_PAD = 0.40
HEAD_H, SUB_H, HEAD_GAP = 0.52, 0.40, 0.40
CHIP_H  = 0.58
ARROW_GAP = 0.50
BOT_PAD = 0.40

CARD_TOP, TITLE_H, DLINE, MOD_H, CARD_BOT = 0.34, 0.46, 0.345, 0.50, 0.30
TEXT_X = CARD_X + 1.18         # left edge of title/desc (past the number badge)


def card_height(stage):
    return CARD_TOP + TITLE_H + DLINE * len(stage["desc"]) + MOD_H + CARD_BOT


STAGES = [
    dict(accent=S1, num="1", title="EXTRACTOR", tag="deterministic · no LLM",
         desc=["Reads the raw file and builds a compact metadata",
               "\"envelope\": format, schema, column types, row",
               "count, delimiters + a few sample rows."],
         module="datamorph/extractor/   ·   CSV · JSON · TXT"),
    dict(accent=S2, num="2", title="SUMMARIZER", tag="Gemma 2B · deferred",
         desc=["Turns the envelope into a short natural-language",
               "brief of the conversion job.",
               "(inference-time nicety — not needed to train)"],
         module="datamorph/models/gemma_mlx.py"),
    dict(accent=S3, num="3", title="SCRIPT GENERATION", tag="Opus → Gemma",
         desc=["Reads ONLY the envelope — never the whole file —",
               "and writes  <analysis>  +  a Python  <script>.",
               "Teacher: Claude Opus (train).  Student: Gemma 2B."],
         module="datamorph/data/teacher_script.py  +  skills/"),
    dict(accent=S4, num="4", title="SANDBOX EXEC", tag="isolated subprocess",
         desc=["Runs the generated script on the REAL file in an",
               "isolated subprocess: 15s timeout + 15s CPU limit.",
               "Captures the converted output — or the error."],
         module="datamorph/data/sandbox.py"),
    dict(accent=S5, num="5", title="VALIDATOR", tag="4 quality metrics",
         desc=["Format Validity · Schema Compliance · Loadability",
               "· Content Accuracy.",
               "Accept when FV/SC/LD = 1.0  and  CA ≥ 0.95."],
         module="datamorph/evaluation/metrics.py"),
]


# ----------------------------------------------------------------------------- helpers
def chip(ax, cy, text, fill, txtcolor=CARD):
    w = len(text) * cw(13) + 0.8
    x = CX - w / 2
    ax.add_patch(FancyBboxPatch((x, cy), w, CHIP_H,
                 boxstyle="round,pad=0,rounding_size=0.29",
                 linewidth=0, facecolor=fill, zorder=3))
    ax.text(CX, cy + CHIP_H / 2, text, fontproperties=CHIP_FP, color=txtcolor,
            ha="center", va="center", zorder=4)
    return cy + CHIP_H


def arrow_down(ax, y1, y2):
    ax.add_patch(FancyArrowPatch((CX, y1), (CX, y2), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.2, color=ARROW, zorder=2))


def draw_card(ax, cy, s):
    ch = card_height(s)
    ax.add_patch(FancyBboxPatch((CARD_X, cy), CARD_W, ch,
                 boxstyle="round,pad=0,rounding_size=0.16",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=3))
    # left accent strip
    ax.add_patch(FancyBboxPatch((CARD_X + 0.10, cy + 0.10), 0.14, ch - 0.20,
                 boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=0, facecolor=s["accent"], zorder=4))
    # number badge
    badge_cy = cy + CARD_TOP + 0.22
    ax.add_patch(Circle((CARD_X + 0.66, badge_cy), 0.29, facecolor=s["accent"],
                        edgecolor="none", zorder=4))
    ax.text(CARD_X + 0.66, badge_cy, s["num"], fontproperties=BADGE_FP, color=CARD,
            ha="center", va="center", zorder=5)
    # title + right-aligned tag
    ax.text(TEXT_X, badge_cy, s["title"], fontproperties=TITLE_FP, color=INK,
            ha="left", va="center", zorder=5)
    ax.text(CARD_X + CARD_W - 0.30, badge_cy, s["tag"], fontproperties=TAG_FP,
            color=s["accent"], ha="right", va="center", zorder=5)
    # description
    dy = cy + CARD_TOP + TITLE_H + 0.04
    for d in s["desc"]:
        ax.text(TEXT_X, dy + DLINE / 2, d, fontproperties=DESC_FP, color=SUBTLE,
                ha="left", va="center", zorder=5)
        dy += DLINE
    # module pill
    mtext = s["module"]
    mw = len(mtext) * cw(11) + 0.42
    py = dy + 0.08
    ax.add_patch(FancyBboxPatch((TEXT_X, py), mw, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.10",
                 linewidth=0, facecolor=PILL, zorder=4))
    ax.text(TEXT_X + 0.21, py + 0.17, mtext, fontproperties=MOD_FP, color=SUBTLE,
            ha="left", va="center", zorder=5)
    return cy + ch


# ----------------------------------------------------------------------------- compose
def main():
    # total height
    H = (TOP_PAD + HEAD_H + SUB_H + HEAD_GAP
         + CHIP_H + ARROW_GAP
         + sum(card_height(s) + ARROW_GAP for s in STAGES)
         + CHIP_H + BOT_PAD)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis("off")

    # header
    cy = TOP_PAD
    ax.text(CX, cy + HEAD_H / 2, "data morph — 5-stage conversion pipeline",
            fontproperties=HEAD_FP, color=INK, ha="center", va="center")
    cy += HEAD_H
    ax.text(CX, cy + SUB_H / 2,
            "Reads a metadata envelope, not the whole file → writes a script that converts it",
            fontproperties=SUB_FP, color=SUBTLE, ha="center", va="center")
    cy += SUB_H + HEAD_GAP

    # input chip
    cy = chip(ax, cy, "INPUT   ·   messy file:  CSV / JSON / TXT", C_IN)

    # stages
    stage_mid = {}
    for i, s in enumerate(STAGES):
        arrow_down(ax, cy, cy + ARROW_GAP)
        cy += ARROW_GAP
        top = cy
        cy = draw_card(ax, cy, s)
        stage_mid[i] = (top + cy) / 2

    # output chip
    arrow_down(ax, cy, cy + ARROW_GAP)
    cy += ARROW_GAP
    chip(ax, cy, "OUTPUT   ·   clean converted file", C_OUT)

    # retry loop: stage 5 -> back up to stage 3, routed through the right gutter
    gx = CARD_X + CARD_W            # right edge of the cards
    xg = gx + 0.50                  # vertical run sits in the gutter
    y5, y3 = stage_mid[4], stage_mid[2]
    LOOP = "#d20f39"
    ax.plot([gx, xg], [y5, y5], color=LOOP, linewidth=2.2, zorder=6)
    ax.plot([xg, xg], [y5, y3], color=LOOP, linewidth=2.2, zorder=6)
    ax.add_patch(FancyArrowPatch((xg, y3), (gx, y3), arrowstyle="-|>",
                 mutation_scale=16, linewidth=2.2, color=LOOP, zorder=6))
    ax.text(xg + 0.22, (y3 + y5) / 2, "✗ fail → retry ≤ 3  ·  error fed back to stage 3",
            fontproperties=LOOP_FP, color=LOOP, ha="center", va="center",
            rotation=90, zorder=6)

    out = Path(__file__).resolve().parents[1] / "06_pipeline.png"
    fig.savefig(str(out), dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", out)


# =========================================================================== HORIZONTAL
# Same content, laid out left→right as a wide banner. Descriptions are re-wrapped
# into narrower lines to fit the columns; the retry loop is routed underneath.
HSTAGES = [
    dict(accent=S1, num="1", title="EXTRACTOR", tag="deterministic · no LLM",
         desc=["Reads the raw file and", "builds a compact metadata",
               "\"envelope\": format, schema,", "types, row count, samples."],
         module="datamorph/extractor/"),
    dict(accent=S2, num="2", title="SUMMARIZER", tag="Gemma 2B · deferred",
         desc=["Turns the envelope into a", "short NL brief of the",
               "conversion job.", "(deferred — not needed yet)"],
         module="datamorph/models/"),
    dict(accent=S3, num="3", title="SCRIPT GEN", tag="Opus → Gemma",
         desc=["Reads ONLY the envelope —", "never the whole file.",
               "Teacher Opus → student Gemma."],
         module="data/teacher_script.py"),
    dict(accent=S4, num="4", title="SANDBOX EXEC", tag="isolated subprocess",
         desc=["Runs the script on the REAL", "file in an isolated",
               "subprocess: 15s timeout +", "15s CPU limit. Captures",
               "output — or the error."],
         module="data/sandbox.py"),
    dict(accent=S5, num="5", title="VALIDATOR", tag="4 quality metrics",
         desc=["Format Validity · Schema", "Compliance · Loadability ·",
               "Content Accuracy. Accept", "when FV/SC/LD = 1.0 and",
               "CA ≥ 0.95."],
         module="evaluation/metrics.py"),
]

# horizontal geometry (inches)
H_LEFT, H_TOP, H_BOT = 0.40, 0.40, 0.40
H_CARD_W, H_CARD_H = 3.05, 3.20
H_GAP = 0.58                      # arrow space between elements
H_CHIP_W = 1.98
HEADER_H_H, HEADER_SUB_H = 0.52, 0.40
LOOP_BAND = 1.15                  # space under the cards for the retry loop

HT_TITLE = FontProperties(family="monospace", size=13, weight="bold")
HT_TAG   = FontProperties(family="monospace", size=10, weight="bold")
HT_DESC  = FontProperties(family="monospace", size=11)
HT_MOD   = FontProperties(family="monospace", size=10)
HT_BADGE = FontProperties(family="monospace", size=14, weight="bold")
HT_CHIP  = FontProperties(family="monospace", size=12, weight="bold")

IPAD = 0.22                       # inner padding of a card


def _h_card(ax, x, ytop, s):
    ax.add_patch(FancyBboxPatch((x, ytop), H_CARD_W, H_CARD_H,
                 boxstyle="round,pad=0,rounding_size=0.16",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=3))
    # top accent strip
    ax.add_patch(FancyBboxPatch((x + 0.10, ytop + 0.10), H_CARD_W - 0.20, 0.12,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 linewidth=0, facecolor=s["accent"], zorder=4))
    tx = x + IPAD
    # badge + title — sits clearly below the accent strip (strip bottom = ytop+0.22)
    by = ytop + 0.52
    ax.add_patch(Circle((tx + 0.22, by), 0.23, facecolor=s["accent"],
                        edgecolor="none", zorder=4))
    ax.text(tx + 0.22, by, s["num"], fontproperties=HT_BADGE, color=CARD,
            ha="center", va="center", zorder=5)
    ax.text(tx + 0.58, by, s["title"], fontproperties=HT_TITLE, color=INK,
            ha="left", va="center", zorder=5)
    # tag
    ax.text(tx, ytop + 0.92, s["tag"], fontproperties=HT_TAG, color=s["accent"],
            ha="left", va="center", zorder=5)
    # desc
    dy = ytop + 1.22
    for d in s["desc"]:
        ax.text(tx, dy, d, fontproperties=HT_DESC, color=SUBTLE,
                ha="left", va="center", zorder=5)
        dy += 0.30
    # module pill pinned near the bottom
    mw = len(s["module"]) * cw(10) + 0.40
    my = ytop + H_CARD_H - IPAD - 0.34
    ax.add_patch(FancyBboxPatch((tx, my), mw, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.10",
                 linewidth=0, facecolor=PILL, zorder=4))
    ax.text(tx + 0.20, my + 0.17, s["module"], fontproperties=HT_MOD, color=SUBTLE,
            ha="left", va="center", zorder=5)


def _h_chip(ax, x, ymid, lines, fill, txtcolor=CARD):
    h = 1.55
    ax.add_patch(FancyBboxPatch((x, ymid - h / 2), H_CHIP_W, h,
                 boxstyle="round,pad=0,rounding_size=0.22",
                 linewidth=0, facecolor=fill, zorder=3))
    cx = x + H_CHIP_W / 2
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(cx, ymid + (i - (n - 1) / 2) * 0.34, ln, fontproperties=HT_CHIP,
                color=txtcolor, ha="center", va="center", zorder=4)


def _h_arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.2, color=ARROW, zorder=2))


SMALL_FP = FontProperties(family="monospace", size=10)
GROUP_FP = FontProperties(family="monospace", size=11, weight="bold")
MODEL_FP = FontProperties(family="monospace", size=10.5, weight="bold")


def _diag_arrow(ax, p1, p2, rad=0.0, color=ARROW, label=None, loff=(0.0, -0.16)):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=color, zorder=5))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + loff[0], (p1[1] + p2[1]) / 2 + loff[1]
        ax.text(mx, my, label, fontproperties=SMALL_FP, color=color,
                ha="center", va="center", zorder=6)


def _model_card(ax, x, ytop, s, w, h):
    """Script generator, emphasized: filled accent header + 'the model' ribbon."""
    acc = s["accent"]
    ax.add_patch(FancyBboxPatch((x, ytop), w, h,
                 boxstyle="round,pad=0,rounding_size=0.16",
                 linewidth=2.6, edgecolor=acc, facecolor="#fff6f0", zorder=3))
    # filled header band
    hb = 0.80
    ax.add_patch(FancyBboxPatch((x + 0.10, ytop + 0.10), w - 0.20, hb,
                 boxstyle="round,pad=0,rounding_size=0.12",
                 linewidth=0, facecolor=acc, zorder=4))
    ax.add_patch(Circle((x + 0.44, ytop + 0.38), 0.22, facecolor=CARD, edgecolor="none", zorder=5))
    ax.text(x + 0.44, ytop + 0.38, s["num"], fontproperties=HT_BADGE, color=acc,
            ha="center", va="center", zorder=6)
    ax.text(x + 0.80, ytop + 0.38, "SCRIPT GENERATOR", fontproperties=HT_TITLE, color=CARD,
            ha="left", va="center", zorder=6)
    ax.text(x + 0.22, ytop + 0.68, "▶  the model writes the script",
            fontproperties=MODEL_FP, color=CARD, ha="left", va="center", zorder=6)
    # body
    tx = x + IPAD
    dy = ytop + hb + 0.28
    for d in s["desc"]:
        ax.text(tx, dy, d, fontproperties=HT_DESC, color=SUBTLE, ha="left", va="center", zorder=5)
        dy += 0.30
    mw = len(s["module"]) * cw(10) + 0.40
    my = ytop + h - IPAD - 0.34
    ax.add_patch(FancyBboxPatch((tx, my), mw, 0.34, boxstyle="round,pad=0,rounding_size=0.10",
                 linewidth=0, facecolor=PILL, zorder=4))
    ax.text(tx + 0.20, my + 0.17, s["module"], fontproperties=HT_MOD, color=SUBTLE,
            ha="left", va="center", zorder=5)


def _script_doc(ax, cx, cy, acc=S3):
    """A small document glyph representing the generated script artifact."""
    w, hh = 1.62, 0.92
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - hh / 2), w, hh,
                 boxstyle="round,pad=0,rounding_size=0.08",
                 linewidth=1.8, edgecolor=acc, facecolor=CARD, zorder=6))
    ax.text(cx, cy - 0.13, "{ } convert.py", fontproperties=MODEL_FP, color=acc,
            ha="center", va="center", zorder=7)
    ax.text(cx, cy + 0.14, "generated script", fontproperties=SMALL_FP, color=SUBTLE,
            ha="center", va="center", zorder=7)


def render_horizontal():
    ext, summ, sg, sb, vl = HSTAGES
    M, chip_w, cardw, sgw, ch = 0.40, H_CHIP_W, 3.00, 3.30, 2.70
    g_in, g_merge, g_script, g, glane = 0.95, 1.15, 2.05, 0.85, 0.80

    # x left-edges
    x_in = M
    x_br = x_in + chip_w + g_in
    x_sg = x_br + cardw + g_merge
    x_sb = x_sg + sgw + g_script
    x_vl = x_sb + cardw + g
    x_out = x_vl + cardw + g
    W_h = x_out + chip_w + M

    # vertical: top lane (extractor) / mid line / bottom lane (summarizer)
    header_block = HEADER_H_H + HEADER_SUB_H + 0.34
    Y0 = H_TOP + header_block
    y_top = Y0
    y_bot = Y0 + ch + glane
    band_h = 2 * ch + glane
    y_mid = Y0 + band_h / 2          # the main flow line (center)
    mid_top = y_mid - ch / 2
    H_h = Y0 + band_h + H_BOT + 0.15

    fig = plt.figure(figsize=(W_h, H_h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_h); ax.set_ylim(0, H_h)
    ax.invert_yaxis(); ax.axis("off")

    cxall = W_h / 2
    ax.text(cxall, H_TOP + HEADER_H_H / 2, "data morph — conversion pipeline",
            fontproperties=HEAD_FP, color=INK, ha="center", va="center")
    ax.text(cxall, H_TOP + HEADER_H_H + HEADER_SUB_H / 2,
            "Two readings of the file feed the model — it writes a script that is run in a sandbox and validated",
            fontproperties=SUB_FP, color=SUBTLE, ha="center", va="center")

    ext_mid, summ_mid = y_top + ch / 2, y_bot + ch / 2

    # chips
    _h_chip(ax, x_in, y_mid, ["INPUT", "messy file", "CSV / JSON / TXT"], C_IN)
    _h_chip(ax, x_out, y_mid, ["OUTPUT", "clean", "converted file"], C_OUT)

    # branch cards (parallel)
    _h_card(ax, x_br, y_top, ext)
    _h_card(ax, x_br, y_bot, summ)

    # dashed group around sandbox + validator
    gpad = 0.22
    gx0, gx1 = x_sb - gpad, x_vl + cardw + gpad
    gy0, gy1 = mid_top - 0.46, mid_top + ch + gpad
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                 boxstyle="round,pad=0,rounding_size=0.14", linewidth=1.6,
                 edgecolor=S4, facecolor="none", linestyle=(0, (5, 3)), zorder=2.4))
    ax.text((gx0 + gx1) / 2, gy0 + 0.21, "RUN THE SCRIPT  &  VALIDATE",
            fontproperties=GROUP_FP, color=S4, ha="center", va="center", zorder=3)

    # mid-line cards
    _model_card(ax, x_sg, mid_top, sg, sgw, ch)
    _h_card(ax, x_sb, mid_top, sb)
    _h_card(ax, x_vl, mid_top, vl)

    # --- arrows ---
    # input fans out to the two readers
    _diag_arrow(ax, (x_in + chip_w, y_mid), (x_br, ext_mid), rad=0.16)
    _diag_arrow(ax, (x_in + chip_w, y_mid), (x_br, summ_mid), rad=-0.16)
    # readers merge into the model, labeled with what they hand over
    _diag_arrow(ax, (x_br + cardw, ext_mid), (x_sg, y_mid - 0.52), rad=-0.16,
                label="envelope", loff=(0.06, -0.18), color=S1)
    _diag_arrow(ax, (x_br + cardw, summ_mid), (x_sg, y_mid + 0.52), rad=0.16,
                label="brief", loff=(0.06, 0.18), color=S2)
    # model → script doc → sandbox
    art_cx = (x_sg + sgw + x_sb) / 2
    _h_arrow(ax, x_sg + sgw, art_cx - 0.88, y_mid)
    _script_doc(ax, art_cx, y_mid)
    _h_arrow(ax, art_cx + 0.88, x_sb, y_mid)
    # sandbox → validator → output
    _h_arrow(ax, x_sb + cardw, x_vl, y_mid)
    _h_arrow(ax, x_vl + cardw, x_out, y_mid)

    # retry loop: validator → back to the model, routed below the mid line (right side)
    LOOP = "#d20f39"
    yL = mid_top + ch + 0.42
    c_vl, c_sg = x_vl + cardw / 2, x_sg + sgw / 2
    ax.plot([c_vl, c_vl], [gy1, yL], color=LOOP, linewidth=2.2, zorder=6)
    ax.plot([c_vl, c_sg], [yL, yL], color=LOOP, linewidth=2.2, zorder=6)
    ax.add_patch(FancyArrowPatch((c_sg, yL), (c_sg, mid_top + ch), arrowstyle="-|>",
                 mutation_scale=16, linewidth=2.2, color=LOOP, zorder=6))
    ax.text((c_sg + c_vl) / 2, yL + 0.24, "✗ fail → retry ≤ 3  ·  error fed back to the model",
            fontproperties=LOOP_FP, color=LOOP, ha="center", va="center", zorder=6)

    out = Path(__file__).resolve().parents[1] / "07_pipeline_horizontal.png"
    fig.savefig(str(out), dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", out)


def render_separate_boxes():
    """Each pipeline component as its own transparent PNG (no arrows/lines),
    so they can be arranged and connected by hand in Canva."""
    ext, summ, sg, sb, vl = HSTAGES
    comp = Path(__file__).resolve().parents[1] / "pipeline_components"
    comp.mkdir(exist_ok=True)
    m = 0.16   # margin so rounded borders aren't clipped

    def solo(draw, w, h, name):
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.invert_yaxis(); ax.axis("off")
        draw(ax)
        out = comp / name
        fig.savefig(str(out), dpi=200, transparent=True)
        plt.close(fig)
        print("wrote", out)

    chh = 1.55
    solo(lambda ax: _h_chip(ax, m, m + chh / 2, ["INPUT", "messy file", "CSV / JSON / TXT"], C_IN),
         H_CHIP_W + 2 * m, chh + 2 * m, "box_0_input.png")
    solo(lambda ax: _h_card(ax, m, m, ext),
         H_CARD_W + 2 * m, H_CARD_H + 2 * m, "box_1_extractor.png")
    solo(lambda ax: _h_card(ax, m, m, summ),
         H_CARD_W + 2 * m, H_CARD_H + 2 * m, "box_2_summarizer.png")
    sgw = 3.30
    solo(lambda ax: _model_card(ax, m, m, sg, sgw, H_CARD_H),
         sgw + 2 * m, H_CARD_H + 2 * m, "box_3_script_generator.png")
    dw, dh = 1.62, 0.92
    solo(lambda ax: _script_doc(ax, (dw + 2 * m) / 2, (dh + 2 * m) / 2),
         dw + 2 * m, dh + 2 * m, "box_artifact_convert_py.png")
    solo(lambda ax: _h_card(ax, m, m, sb),
         H_CARD_W + 2 * m, H_CARD_H + 2 * m, "box_4_sandbox.png")
    solo(lambda ax: _h_card(ax, m, m, vl),
         H_CARD_W + 2 * m, H_CARD_H + 2 * m, "box_5_validator.png")
    solo(lambda ax: _h_chip(ax, m, m + chh / 2, ["OUTPUT", "clean", "converted file"], C_OUT),
         H_CHIP_W + 2 * m, chh + 2 * m, "box_6_output.png")


if __name__ == "__main__":
    main()
    render_horizontal()
    render_separate_boxes()
