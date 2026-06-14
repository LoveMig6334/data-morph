"""
Data Morph — CSV -> JSON (nested) problem illustration.
Renders 3 standalone components (transparent background) for composing in Canva:

  1. csv_flat.png            flat CSV source (one customer spread over many rows)
  2. json_naive_wrong.png    naive row-by-row output  -> wrapper lost + numbers as strings
  3. json_nested_correct.png correct grouped output    -> orders[] array + numbers as numbers

Each panel is a rounded card on a transparent canvas so arrows / captions can be
added later. A 4th "combined_overview.png" is a quick reference layout (white bg).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------- palette
INK      = "#1e1e2e"   # default text
SUBTLE   = "#6c6f85"   # punctuation / muted
BLUE     = "#1e66f5"   # json keys
TEAL     = "#179299"   # string values
PEACH    = "#fe640b"   # number values
CARD     = "#ffffff"   # card fill
EDGE     = "#cdd0da"   # card border

BAD_FILL, BAD_EDGE   = "#fde0e4", "#d20f39"   # problem highlight
GOOD_FILL, GOOD_EDGE = "#dbf0d6", "#40a02b"   # good highlight
HEAD_FILL            = "#eaecf4"               # table header fill
DUP_FILL             = "#fde0e4"               # duplicated-key cell fill

MONO = FontProperties(family="monospace", size=15)
MONO_B = FontProperties(family="monospace", size=15, weight="bold")

# monospace metrics (DejaVu Sans Mono): advance ~= 0.602 * fontsize
FS        = 15
CHAR_W_IN = 0.602 * FS / 72.0     # inches per character column
LINE_H_IN = 1.70  * FS / 72.0     # inches per text line
PAD_COLS  = 2.0                   # horizontal padding (in char units), each side
PAD_LINES = 1.2                   # vertical padding (in line units), top & bottom


# ----------------------------------------------------------------------------- helpers
def code_panel(filename, lines, title=None, title_color=INK):
    """lines: list of rows; each row is a list of (text, kind) segments.
    kind in {plain, key, str, num, punct, bad, good}."""
    ncols = max(sum(len(t) for t, _ in row) for row in lines)
    ncols = max(ncols, len(title) if title else 0) + 2 * PAD_COLS
    title_lines = 1.4 if title else 0
    nlines = len(lines) + 2 * PAD_LINES + title_lines

    fig_w = ncols * CHAR_W_IN
    fig_h = nlines * LINE_H_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nlines)
    ax.invert_yaxis()
    ax.axis("off")

    # rounded card
    inset = 0.12
    card = FancyBboxPatch(
        (inset, inset), ncols - 2 * inset, nlines - 2 * inset,
        boxstyle="round,pad=0,rounding_size=0.6",
        linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1,
    )
    ax.add_patch(card)

    y = PAD_LINES + 0.6
    if title:
        ax.text(PAD_COLS, y, title, fontproperties=MONO_B, color=title_color,
                ha="left", va="center", zorder=3)
        y += title_lines

    color_map = {"plain": INK, "key": BLUE, "str": TEAL, "num": PEACH, "punct": SUBTLE}
    for row in lines:
        x = PAD_COLS
        for text, kind in row:
            w = len(text)
            if kind in ("bad", "good"):
                fill, edge = (BAD_FILL, BAD_EDGE) if kind == "bad" else (GOOD_FILL, GOOD_EDGE)
                ax.add_patch(Rectangle((x - 0.15, y - 0.55), w + 0.30, 1.05,
                                       facecolor=fill, edgecolor=edge, linewidth=1.2,
                                       zorder=2, joinstyle="round"))
                ax.text(x, y, text, fontproperties=MONO, color=INK,
                        ha="left", va="center", zorder=3)
            else:
                ax.text(x, y, text, fontproperties=MONO,
                        color=color_map.get(kind, INK),
                        ha="left", va="center", zorder=3)
            x += w
        y += 1.0

    fig.savefig(filename, dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", filename)


def csv_table(filename, header, rows, dup_col=0, dup_groups=None, title="customers.csv"):
    """Render a flat CSV as a clean grid; optionally shade a duplicated key column."""
    ncols = len(header)
    nrows = len(rows)
    col_w = 2.55           # inches per column
    row_h = 0.52           # inches per row
    title_h = 0.62
    pad = 0.28

    fig_w = ncols * col_w + 2 * pad
    fig_h = title_h + (nrows + 1) * row_h + 2 * pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    card = FancyBboxPatch(
        (0.12, 0.12), fig_w - 0.24, fig_h - 0.24,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1,
    )
    ax.add_patch(card)

    ax.text(pad + 0.1, pad + 0.34, title, fontproperties=MONO_B, color=SUBTLE,
            ha="left", va="center", zorder=3)

    x0, y0 = pad, pad + title_h

    # header
    ax.add_patch(Rectangle((x0, y0), ncols * col_w, row_h,
                           facecolor=HEAD_FILL, edgecolor="none", zorder=2))
    for c, h in enumerate(header):
        ax.text(x0 + c * col_w + 0.18, y0 + row_h / 2, h,
                fontproperties=MONO_B, color=INK, ha="left", va="center", zorder=3)

    # body
    for r, row in enumerate(rows):
        yr = y0 + (r + 1) * row_h
        # shade duplicated key cells
        if dup_groups:
            for grp in dup_groups:
                if r in grp and len(grp) > 1:
                    ax.add_patch(Rectangle((x0 + dup_col * col_w, yr),
                                           col_w, row_h, facecolor=DUP_FILL,
                                           edgecolor="none", zorder=1.5))
        for c, val in enumerate(row):
            ax.text(x0 + c * col_w + 0.18, yr + row_h / 2, val,
                    fontproperties=MONO, color=INK, ha="left", va="center", zorder=3)

    # grid lines
    for c in range(ncols + 1):
        ax.plot([x0 + c * col_w] * 2, [y0, y0 + (nrows + 1) * row_h],
                color=EDGE, linewidth=0.8, zorder=2)
    for r in range(nrows + 2):
        ax.plot([x0, x0 + ncols * col_w], [y0 + r * row_h] * 2,
                color=EDGE, linewidth=0.8, zorder=2)

    fig.savefig(filename, dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", filename)


# ----------------------------------------------------------------------------- data
OUT = "/mnt/user-data/outputs/"

# 1) flat CSV source — Alice appears twice (two orders), Bob once
csv_table(
    OUT + "01_csv_flat.png",
    header=["customer_id", "name", "order_id", "amount"],
    rows=[
        ["C001", "Alice", "O100", "250"],
        ["C001", "Alice", "O101", "80"],
        ["C002", "Bob",   "O102", "1200"],
    ],
    dup_col=0,
    dup_groups=[[0, 1]],   # rows 0 & 1 share customer_id C001
    title="customers.csv  (flat — 1 customer spread over many rows)",
)

# 2) NAIVE / WRONG output: one object per row -> Alice duplicated, numbers quoted
code_panel(
    OUT + "02_json_naive_wrong.png",
    title="output.json  — naive row-by-row",
    title_color=BAD_EDGE,
    lines=[
        [("[", "punct")],
        [("  {", "punct"), ('"customer_id"', "key"), (": ", "punct"),
         ('"C001"', "str"), (", ", "punct"), ('"name"', "key"), (": ", "punct"),
         ('"Alice"', "str"), (", ", "punct"), ('"amount"', "key"), (": ", "punct"),
         ('"250"', "bad"), ("},", "punct")],
        [("  {", "punct"), ('"customer_id"', "key"), (": ", "punct"),
         ('"C001"', "bad"), (", ", "punct"), ('"name"', "key"), (": ", "punct"),
         ('"Alice"', "bad"), (", ", "punct"), ('"amount"', "key"), (": ", "punct"),
         ('"80"', "bad"), ("},", "punct")],
        [("  {", "punct"), ('"customer_id"', "key"), (": ", "punct"),
         ('"C002"', "str"), (", ", "punct"), ('"name"', "key"), (": ", "punct"),
         ('"Bob"', "str"), (", ", "punct"), ('"amount"', "key"), (": ", "punct"),
         ('"1200"', "bad"), ("}", "punct")],
        [("]", "punct")],
        [("", "plain")],
        [("✗  wrapper lost: Alice split across 2 objects", "plain")],
        [("✗  numbers kept as strings (\"250\", \"80\", \"1200\")", "plain")],
    ],
)

# 3) CORRECT nested output: grouped by customer, orders[] array, numbers as numbers
code_panel(
    OUT + "03_json_nested_correct.png",
    title="output.json  — correct nested",
    title_color=GOOD_EDGE,
    lines=[
        [("[", "punct")],
        [("  {", "punct")],
        [("    ", "plain"), ('"customer_id"', "key"), (": ", "punct"), ('"C001"', "str"), (",", "punct")],
        [("    ", "plain"), ('"name"', "key"), (": ", "punct"), ('"Alice"', "str"), (",", "punct")],
        [("    ", "plain"), ('"orders"', "key"), (": ", "punct"), ("[", "good")],
        [("      {", "punct"), ('"order_id"', "key"), (": ", "punct"), ('"O100"', "str"),
         (", ", "punct"), ('"amount"', "key"), (": ", "punct"), ("250", "good"), ("},", "punct")],
        [("      {", "punct"), ('"order_id"', "key"), (": ", "punct"), ('"O101"', "str"),
         (", ", "punct"), ('"amount"', "key"), (": ", "punct"), ("80", "good"), ("}", "punct")],
        [("    ", "plain"), ("]", "good")],
        [("  },", "punct")],
        [("  {", "punct")],
        [("    ", "plain"), ('"customer_id"', "key"), (": ", "punct"), ('"C002"', "str"), (",", "punct")],
        [("    ", "plain"), ('"name"', "key"), (": ", "punct"), ('"Bob"', "str"), (",", "punct")],
        [("    ", "plain"), ('"orders"', "key"), (": ", "punct"), ("[", "good"),
         ("{", "punct"), ('"order_id"', "key"), (": ", "punct"), ('"O102"', "str"),
         (", ", "punct"), ('"amount"', "key"), (": ", "punct"), ("1200", "good"), ("}", "punct"), ("]", "good")],
        [("  }", "punct")],
        [("]", "punct")],
        [("", "plain")],
        [("✓  Alice grouped: 2 orders in one orders[] array", "plain")],
        [("✓  amount cast to number (250, 80, 1200)", "plain")],
    ],
)

print("done")
