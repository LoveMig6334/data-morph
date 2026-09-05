"""
Data Morph — "where the tokens go" explainer.
Renders 3 standalone cards (transparent background) for composing in Canva,
explaining the three token types you pay for on every frontier-model conversion,
mapped to THIS task (write a script that converts a file):

  03_input_tokens.png      INPUT  tokens = the source file the model reads  (scales w/ file size)
  04_reasoning_tokens.png  REASONING tokens = the hidden thinking about how to write the script
  05_output_tokens.png     OUTPUT tokens = the Python script the model writes

Same visual language as render_components.py (palette, rounded cards, monospace,
dpi=200, transparent canvas). Token counts are the real per-file averages from the
cost probe (scripts/cost_probe.py → data/cost_probe/results.json).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------- palette
INK    = "#1e1e2e"
SUBTLE = "#6c6f85"
BLUE   = "#1e66f5"   # input
PEACH  = "#fe640b"   # reasoning
TEAL   = "#179299"   # output
CARD   = "#ffffff"
EDGE   = "#cdd0da"
EX_FILL = "#f4f5fb"  # example sub-box fill

OUT = Path(__file__).resolve().parents[1]    # .../file_pre/

KICKER_FP = FontProperties(family="monospace", size=13, weight="bold")
TITLE_FP  = FontProperties(family="monospace", size=19, weight="bold")
DESC_FP   = FontProperties(family="monospace", size=13)
EXT_FP    = FontProperties(family="monospace", size=12, weight="bold")
EX_FP     = FontProperties(family="monospace", size=13)
FOOT_FP   = FontProperties(family="monospace", size=12, weight="bold")

# vertical layout increments (inches)
W       = 6.7
PAD     = 0.34
TX      = PAD + 0.40          # left text margin (past the accent bar)
H_KICK  = 0.34
H_TITLE = 0.60
H_DGAP  = 0.08
H_DLINE = 0.34
H_BGAP  = 0.22
EX_TOP  = 0.30
EX_BOT  = 0.24
EX_TITLE_H = 0.42
EX_LINE_H  = 0.36
H_FGAP  = 0.26
H_FOOT  = 0.34


def concept_card(filename, accent, kicker, title, desc, example_title, example, footnote):
    ex_h = EX_TOP + EX_TITLE_H + EX_LINE_H * len(example) + EX_BOT
    H = (PAD + H_KICK + H_TITLE + H_DGAP + H_DLINE * len(desc) + H_BGAP
         + ex_h + H_FGAP + H_FOOT + PAD)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis("off")

    # card + accent bar
    ax.add_patch(FancyBboxPatch((0.12, 0.12), W - 0.24, H - 0.24,
                 boxstyle="round,pad=0,rounding_size=0.18",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1))
    ax.add_patch(FancyBboxPatch((PAD, PAD), 0.16, H - 2 * PAD,
                 boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=0, facecolor=accent, zorder=2))

    cy = PAD
    ax.text(TX, cy + H_KICK / 2, kicker, fontproperties=KICKER_FP, color=accent,
            ha="left", va="center", zorder=3)
    cy += H_KICK
    ax.text(TX, cy + H_TITLE / 2, title, fontproperties=TITLE_FP, color=INK,
            ha="left", va="center", zorder=3)
    cy += H_TITLE + H_DGAP
    for d in desc:
        ax.text(TX, cy + H_DLINE / 2, d, fontproperties=DESC_FP, color=SUBTLE,
                ha="left", va="center", zorder=3)
        cy += H_DLINE
    cy += H_BGAP

    # example sub-box
    ax.add_patch(FancyBboxPatch((TX, cy), W - TX - PAD, ex_h,
                 boxstyle="round,pad=0,rounding_size=0.10",
                 linewidth=1.2, edgecolor=EDGE, facecolor=EX_FILL, zorder=2))
    ey = cy + EX_TOP
    ax.text(TX + 0.22, ey + EX_TITLE_H / 2 - 0.04, example_title, fontproperties=EXT_FP,
            color=accent, ha="left", va="center", zorder=3)
    ey += EX_TITLE_H
    for line in example:
        ax.text(TX + 0.22, ey + EX_LINE_H / 2, line, fontproperties=EX_FP,
                color=INK, ha="left", va="center", zorder=3)
        ey += EX_LINE_H
    cy += ex_h + H_FGAP

    ax.text(TX, cy + H_FOOT / 2, footnote, fontproperties=FOOT_FP, color=accent,
            ha="left", va="center", zorder=3)

    fig.savefig(filename, dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", filename)


# ----------------------------------------------------------------------------- cards
concept_card(
    str(OUT / "03_input_tokens.png"), BLUE,
    kicker="[1]  INPUT TOKENS",
    title="What the model reads",
    desc=["The whole source file is fed in as text.",
          "Bigger file  →  more input tokens.",
          "Cost scales with FILE SIZE."],
    example_title="customers.csv  (the file)",
    example=["user_name,user_email,order_id,...",
             "Alice,alice@example.com,1001,...",
             "Bob,bob@example.com,1002,..."],
    footnote="≈ 500–700 tokens for this tiny sample",
)

concept_card(
    str(OUT / "04_reasoning_tokens.png"), PEACH,
    kicker="[2]  REASONING TOKENS",
    title="Thinking how to convert",
    desc=["Hidden chain-of-thought: plan the parse,",
          "the edge cases, and the output shape.",
          "Billed like output — you never see them."],
    example_title="model's private scratchpad",
    example=["› group user_* / order_* by prefix",
             "› cast price \"9.99\" → number 9.99",
             "› one orders[] array per customer"],
    footnote="≈ 1,000 invisible tokens/file (GPT-5.5)",
)

concept_card(
    str(OUT / "05_output_tokens.png"), TEAL,
    kicker="[3]  OUTPUT TOKENS",
    title="The generated script",
    desc=["The actual Python the model writes.",
          "Grows with script length & complexity —",
          "a NEW script for every file you convert."],
    example_title="convert.py  (the output)",
    example=["import csv, json", "",
             "def convert(src, dst):",
             "    rows = csv.DictReader(open(src))",
             "    ..."],
    footnote="≈ 1,200–2,500 tokens of code/file",
)

print("done")
