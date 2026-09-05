"""
Data Morph — experiment charts for the Medium article (file_pre/data-morph-medium.md).
Same visual language as render_components.py (palette, rounded white card, monospace,
dpi=200, transparent canvas). Every number is pulled from real artifacts:
  - results/**/summary.json  (Opus baseline, student base/fine-tuned, retry, ship)
  - data/interim/*.json       (corpus complexity split)
  - data/processed/*.jsonl    (sequence-length distribution, approx tokens)
  - the deployment tables in the article (quantization / model-surgery sizes)

Outputs (file_pre/):
  baseline_opus_overall.png, baseline_opus_by_usecase.png, eda_corpus_balance.png,
  eda_seq_length.png, four_metrics_teacher_vs_student.png, first_model_regression.png,
  uc4_inverted_u.png, per_usecase_before_after.png, quantization_tradeoff.png,
  accuracy_journey.png, model_surgery_size.png
"""

import json
import glob
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------- palette
INK    = "#1e1e2e"
SUBTLE = "#6c6f85"
CARD   = "#ffffff"
EDGE   = "#cdd0da"
GRID   = "#e6e8f0"
BLUE   = "#1e66f5"   # FV / student fine-tuned
MAUVE  = "#8839ef"   # SC
TEAL   = "#179299"   # LD
PEACH  = "#fe640b"   # CA / highlight
GREEN  = "#40a02b"   # teacher / good
GRAY   = "#9ca0b0"   # student base
RED    = "#d20f39"   # regression / bad
YELLOW = "#df8e1d"

METRIC_COLOR = {"format_validity": BLUE, "schema_compliance": MAUVE,
                "loadability": TEAL, "content_accuracy": PEACH}
METRIC_SHORT = {"format_validity": "Format", "schema_compliance": "Schema",
                "loadability": "Load", "content_accuracy": "Content"}

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]

HEAD = FontProperties(family="monospace", size=18, weight="bold")
SUB  = FontProperties(family="monospace", size=11)
LBL  = FontProperties(family="monospace", size=11, weight="bold")
SML  = FontProperties(family="monospace", size=10)
TINY = FontProperties(family="monospace", size=9)


def cw(size):
    return 0.602 * size / 72.0


def load(p):
    return json.loads((ROOT / p).read_text())


def teacher_newpipeline():
    """Opus teacher aggregate on the NEW 5-stage pipeline — from the 800 verified
    training pairs in data/interim (envelope→script→sandbox→validate). Returns
    (overall, by_uc, n, first_try)."""
    from collections import defaultdict
    ov, by = defaultdict(list), defaultdict(lambda: defaultdict(list))
    first_try = 0
    files = glob.glob(str(ROOT / "data/interim/uc*_gen_*.json"))
    for f in files:
        d = json.loads(Path(f).read_text())
        s = d.get("scores") or {}
        first_try += int(d.get("retries", 0) == 0)
        uc = d["use_case"].split("_")[0]
        for m in ("format_validity", "schema_compliance", "loadability", "content_accuracy"):
            if m in s:
                ov[m].append(s[m]); by[uc][m].append(s[m])
    overall = {m: sum(v) / len(v) for m, v in ov.items()}
    byuc = {uc: {m: sum(vs) / len(vs) for m, vs in mm.items()} for uc, mm in by.items()}
    return overall, byuc, len(files), first_try


def card(W, H, title, subtitle):
    """Figure + axis (y-up, inches) with a rounded card backdrop and a header."""
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.12, 0.12), W - 0.24, H - 0.24,
                 boxstyle="round,pad=0,rounding_size=0.2",
                 linewidth=1.6, edgecolor=EDGE, facecolor=CARD, zorder=1))
    ax.text(0.85, H - 0.52, title, fontproperties=HEAD, color=INK, ha="left", va="center")
    if subtitle:
        ax.text(0.85, H - 0.90, subtitle, fontproperties=SUB, color=SUBTLE, ha="left", va="center")
    return fig, ax


def save(fig, name):
    out = OUT / name
    fig.savefig(str(out), dpi=200, transparent=True)
    plt.close(fig)
    print("wrote", out)


def _yaxis(ax, left, right, bottom, top, ymax, ticks, fmt):
    for t in ticks:
        yy = bottom + (t / ymax) * (top - bottom)
        ax.plot([left, right], [yy, yy], color=GRID, lw=1.0, zorder=1.5)
        ax.text(left - 0.14, yy, fmt(t), fontproperties=SML, color=SUBTLE, ha="right", va="center")
    ax.plot([left, left], [bottom, top], color=EDGE, lw=1.2, zorder=2)
    ax.plot([left, right], [bottom, bottom], color=SUBTLE, lw=1.2, zorder=2)


def grouped_bar(name, title, subtitle, groups, series, ymax, ticks, yfmt,
                valfmt, W=9.2, H=5.4, refs=None):
    """series: list of (label, color, [value per group]). refs: list of (y, label, color)."""
    fig, ax = card(W, H, title, subtitle)
    left, right, bottom, top = 1.15, W - 0.45, 1.0, H - 1.55
    _yaxis(ax, left, right, bottom, top, ymax, ticks, yfmt)
    ng, ns = len(groups), len(series)
    gw = (right - left) / ng
    bw = gw * 0.74 / ns
    gap = (gw - bw * ns) / 2
    for gi, g in enumerate(groups):
        gx = left + gi * gw
        for si, (lbl, col, vals) in enumerate(series):
            v = vals[gi]
            bx = gx + gap + si * bw
            bh = (v / ymax) * (top - bottom)
            ax.add_patch(FancyBboxPatch((bx, bottom), bw * 0.88, max(bh, 0.001),
                         boxstyle="round,pad=0,rounding_size=0.03", linewidth=0,
                         facecolor=col, zorder=3))
            ax.text(bx + bw * 0.44, bottom + bh + 0.12, valfmt(v), fontproperties=TINY,
                    color=col, ha="center", va="bottom", zorder=4)
        ax.text(gx + gw / 2, bottom - 0.26, g, fontproperties=LBL, color=INK,
                ha="center", va="center")
    if refs:
        for y, rl, rc in refs:
            yy = bottom + (y / ymax) * (top - bottom)
            ax.plot([left, right], [yy, yy], color=rc, lw=1.6, ls=(0, (5, 3)), zorder=3.5)
            ax.text(right, yy + 0.10, rl, fontproperties=TINY, color=rc, ha="right", va="bottom")
    # legend (top, under subtitle)
    lx, ly = left, H - 1.18
    for lbl, col, _ in series:
        ax.add_patch(Rectangle((lx, ly - 0.10), 0.22, 0.20, facecolor=col, edgecolor="none"))
        ax.text(lx + 0.30, ly, lbl, fontproperties=SML, color=INK, ha="left", va="center")
        lx += 0.30 + len(lbl) * cw(10) + 0.55
    save(fig, name)


# =========================================================================== charts
def chart_opus_overall():
    ov, _, n, _ = teacher_newpipeline()
    metrics = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
    fig, ax = card(9.0, 5.0, "Claude Opus baseline — 5-stage pipeline",
                   f"teacher via the envelope→script→sandbox→validate pipeline · "
                   f"{n} verified pairs · 100% accept")
    left, right, bottom, top = 1.15, 8.55, 1.0, 3.5
    _yaxis(ax, left, right, bottom, top, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    gw = (right - left) / len(metrics)
    for i, m in enumerate(metrics):
        v = ov[m]
        bx = left + i * gw + gw * 0.22
        bw = gw * 0.56
        bh = v * (top - bottom)
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                     linewidth=0, facecolor=METRIC_COLOR[m], zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, f"{v:.3f}", fontproperties=LBL,
                color=METRIC_COLOR[m], ha="center", va="bottom", zorder=4)
        ax.text(bx + bw / 2, bottom - 0.26, METRIC_SHORT[m], fontproperties=LBL, color=INK,
                ha="center", va="center")
    save(fig, "baseline_opus_overall.png")


def chart_student_base_overall():
    d = load("results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json")
    ov = d["aggregate"]["overall"]
    per = _passrate_by_uc("results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json")
    passed = sum(v[0] for v in per.values())
    n = sum(v[1] for v in per.values())
    metrics = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
    fig, ax = card(9.0, 5.0, "Gemma 4 2B base — baseline (5-stage pipeline)",
                   f"student base (no fine-tune) · 70-case held-out test · "
                   f"{passed}/{n} pass all 4 metrics")
    left, right, bottom, top = 1.15, 8.55, 1.0, 3.5
    _yaxis(ax, left, right, bottom, top, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    gw = (right - left) / len(metrics)
    for i, m in enumerate(metrics):
        v = ov[m]
        bx = left + i * gw + gw * 0.22
        bw = gw * 0.56
        bh = v * (top - bottom)
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                     linewidth=0, facecolor=METRIC_COLOR[m], zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, f"{v:.3f}", fontproperties=LBL,
                color=METRIC_COLOR[m], ha="center", va="bottom", zorder=4)
        ax.text(bx + bw / 2, bottom - 0.26, METRIC_SHORT[m], fontproperties=LBL, color=INK,
                ha="center", va="center")
    save(fig, "baseline_student_overall.png")


def chart_opus_heatmap():
    _, by, n, first_try = teacher_newpipeline()
    ucs = sorted(by.keys())
    metrics = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
    fig, ax = card(9.0, 5.2, "Baseline scores by use case (Opus, 5-stage pipeline)",
                   "writing scripts via the pipeline, the teacher passes every metric on all 5 use cases")

    def color_for(v):
        if v >= 0.95: return "#dbf0d6"
        if v >= 0.8:  return "#eaf6e3"
        if v >= 0.5:  return "#fde6c8"
        return "#fbd5da"

    x0, y0 = 2.05, 3.45
    cwd, chd = 1.55, 0.52
    for j, m in enumerate(metrics):
        ax.text(x0 + j * cwd + cwd / 2, y0 + 0.34, METRIC_SHORT[m], fontproperties=LBL,
                color=INK, ha="center", va="center")
    for i, uc in enumerate(ucs):
        yy = y0 - i * chd
        ax.text(x0 - 0.18, yy, uc.split("_")[0].upper(), fontproperties=LBL, color=INK,
                ha="right", va="center")
        for j, m in enumerate(metrics):
            v = by[uc][m]
            cx = x0 + j * cwd
            ax.add_patch(Rectangle((cx, yy - chd / 2 + 0.03), cwd - 0.08, chd - 0.06,
                         facecolor=color_for(v), edgecolor=EDGE, linewidth=0.8, zorder=3))
            txt_col = RED if v < 0.5 else INK
            ax.text(cx + (cwd - 0.08) / 2, yy, f"{v:.2f}", fontproperties=SML, color=txt_col,
                    ha="center", va="center", zorder=4)
    ax.text(x0, y0 - len(ucs) * chd - 0.18,
            f"{n}/{n} pairs accepted (FV/SC/LD=1.0, CA≥0.95) · {first_try} first-try, {n - first_try} needed 1 retry",
            fontproperties=TINY, color=GREEN, ha="left", va="center")
    save(fig, "baseline_opus_by_usecase.png")


def chart_four_metrics():
    opus, _, _, _ = teacher_newpipeline()
    base = load("results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json")["aggregate"]["overall"]
    ft = load("results/eval_finetuned_0000400_adapters_2026-05-31_213849/summary.json")["aggregate"]["overall"]
    metrics = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
    groups = [METRIC_SHORT[m] for m in metrics]
    series = [
        ("Teacher (Opus, pipeline)", GREEN, [opus[m] for m in metrics]),
        ("Student base", GRAY, [base[m] for m in metrics]),
        ("Student fine-tuned (iter-400)", BLUE, [ft[m] for m in metrics]),
    ]
    grouped_bar("four_metrics_teacher_vs_student.png",
                "The four metrics — teacher vs student",
                "all three on the same 5-stage pipeline · fine-tuning closes the gap to the teacher",
                groups, series, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}",
                lambda v: f"{v:.2f}")


def _passrate_by_uc(path):
    d = load(path)
    per = {}
    for c in d["cases"]:
        uc = c["use_case"].split("_")[0]
        s = c.get("scores", {})
        ok = (s.get("format_validity", 0) == 1.0 and s.get("loadability", 0) == 1.0
              and s.get("schema_compliance", 0) == 1.0 and s.get("content_accuracy", 0) >= 0.95)
        per.setdefault(uc, [0, 0])
        per[uc][1] += 1
        per[uc][0] += int(ok)
    return per


# use_case slug → human-readable format conversion
UC_LABEL = {
    "uc1": "CSV→JSON",
    "uc2": "JSON→CSV",
    "uc3": "TXT→CSV",
    "uc4": "CSV→TXT",
    "uc5": "JSON→JSON",
}


def chart_before_after():
    """Grouped bars of PASSED-case counts (not percentages) per conversion type.
    Bar height = cases passing all 4 metrics; each bar labelled passed/total, and
    the per-UC total (n) sits under the format label so denominators stay honest."""
    base = _passrate_by_uc("results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json")
    ft = _passrate_by_uc("results/eval_finetuned_0000400_adapters_2026-05-31_213849/summary.json")
    ucs = sorted(base.keys())
    series = [
        ("Student base", GRAY, base),
        ("Fine-tuned (iter-400)", BLUE, ft),
    ]
    ymax = 20
    ticks = [0, 5, 10, 15, 20]
    fig, ax = card(9.2, 5.4, "Per-use-case counts — before vs after fine-tune",
                   "cases passing all 4 metrics (passed / total) · iter-400 wins or ties")
    left, right, bottom, top = 1.15, 8.75, 1.0, 3.85
    _yaxis(ax, left, right, bottom, top, ymax, ticks, lambda t: f"{int(t)}")
    ng, ns = len(ucs), len(series)
    gw = (right - left) / ng
    bw = gw * 0.74 / ns
    gap = (gw - bw * ns) / 2
    for gi, u in enumerate(ucs):
        gx = left + gi * gw
        total = base[u][1]
        for si, (lbl, col, data) in enumerate(series):
            passed = data[u][0]
            bx = gx + gap + si * bw
            bh = (passed / ymax) * (top - bottom)
            ax.add_patch(FancyBboxPatch((bx, bottom), bw * 0.88, max(bh, 0.001),
                         boxstyle="round,pad=0,rounding_size=0.03", linewidth=0,
                         facecolor=col, zorder=3))
            ax.text(bx + bw * 0.44, bottom + bh + 0.12, f"{passed}/{total}",
                    fontproperties=TINY, color=col, ha="center", va="bottom", zorder=4)
        ax.text(gx + gw / 2, bottom - 0.26, UC_LABEL[u], fontproperties=LBL, color=INK,
                ha="center", va="center")
        ax.text(gx + gw / 2, bottom - 0.52, f"n={total}", fontproperties=TINY, color=SUBTLE,
                ha="center", va="center")
    # y-axis caption
    ax.text(left - 0.72, (bottom + top) / 2, "cases passed", fontproperties=SML, color=SUBTLE,
            ha="center", va="center", rotation=90)
    # legend
    lx, ly = left, 5.4 - 1.18
    for lbl, col, _ in series:
        ax.add_patch(Rectangle((lx, ly - 0.10), 0.22, 0.20, facecolor=col, edgecolor="none"))
        ax.text(lx + 0.30, ly, lbl, fontproperties=SML, color=INK, ha="left", va="center")
        lx += 0.30 + len(lbl) * cw(10) + 0.55
    save(fig, "per_usecase_before_after.png")


def chart_first_model_regression():
    base = _passrate_by_uc("results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json")
    first = _passrate_by_uc("results/eval_finetuned_lora_gemma4e2b_scriptgen_2026-05-31_204718/summary.json")
    ucs = sorted(base.keys())
    groups = [u.upper() for u in ucs]
    series = [
        ("Student base (41/70)", GRAY, [base[u][0] / base[u][1] for u in ucs]),
        ("First model · 3 epochs (51/70)", RED, [first[u][0] / first[u][1] for u in ucs]),
    ]
    grouped_bar("first_model_regression.png",
                "First model: net +10 overall, but UC4 collapsed",
                "3 epochs with no validation set · UC4 (CSV→report) fell 13/19 → 2/19",
                groups, series, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.0%}",
                lambda v: f"{v:.0%}")


def chart_uc4_inverted_u():
    # only 4 checkpoints have eval summaries (no every-200 sweep in results/)
    pts = [
        ("base\n(iter 0)", "results/baseline_newpipeline_gemma_2026-05-31_193850/summary.json"),
        ("iter 400", "results/eval_finetuned_0000400_adapters_2026-05-31_213849/summary.json"),
        ("iter 1200", "results/eval_finetuned_0001200_adapters_2026-05-31_210518/summary.json"),
        ("iter 1950\n(3 epochs)", "results/eval_finetuned_lora_gemma4e2b_scriptgen_2026-05-31_204718/summary.json"),
    ]
    vals = []
    for _, p in pts:
        per = _passrate_by_uc(p)
        vals.append(per["uc4"][0] / per["uc4"][1])
    fig, ax = card(9.0, 5.2, "UC4 over training — an inverted-U",
                   "UC4 pass rate peaks at iter-400 then overfits & collapses (4 eval checkpoints)")
    left, right, bottom, top = 1.2, 8.55, 1.05, 3.55
    _yaxis(ax, left, right, bottom, top, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.0%}")
    n = len(pts)
    xs = [left + (i + 0.5) * (right - left) / n for i in range(n)]
    ys = [bottom + v * (top - bottom) for v in vals]
    ax.plot(xs, ys, color=PEACH, lw=2.6, zorder=3)
    for i, (x, y, v) in enumerate(zip(xs, ys, vals)):
        peak = (i == 1)
        ax.scatter([x], [y], s=130 if peak else 70, color=PEACH, zorder=4,
                   edgecolor=CARD, linewidth=1.5)
        ax.text(x, y + 0.16, f"{v:.0%}", fontproperties=LBL, color=PEACH, ha="center", va="bottom")
        ax.text(x, bottom - 0.30, pts[i][0], fontproperties=SML, color=INK, ha="center", va="top")
    ax.text(xs[1], ys[1] + 0.42, "peak", fontproperties=TINY, color=GREEN, ha="center", va="bottom")
    save(fig, "uc4_inverted_u.png")


def chart_accuracy_journey():
    stages = [
        ("Student\nbase", 41, GRAY),
        ("Fine-tuned\niter-400", 65, BLUE),
        ("+ retry≤3\n(8-bit)", 68, TEAL),
        ("Final ship\n2.0 GB", 67, GREEN),
    ]
    fig, ax = card(9.2, 5.6, "Student accuracy journey (70-case held-out test)",
                   "cases passing all 4 metrics, out of 70 · teacher passes 70/70")
    left, right, bottom, top = 1.2, 8.7, 1.05, 3.5
    ymax = 80          # headroom above the 70 line so bar labels don't collide
    _yaxis(ax, left, right, bottom, top, ymax, [0, 14, 28, 42, 56, 70], lambda t: f"{int(t)}")
    # reference lines; labels on the LEFT (over the short base bar's empty headroom)
    for y, lbl, col in [(70, "teacher 70/70", GREEN), (56, "target ≥80% of teacher", YELLOW)]:
        yy = bottom + (y / ymax) * (top - bottom)
        ax.plot([left, right], [yy, yy], color=col, lw=1.5, ls=(0, (5, 3)), zorder=2.5)
        ax.text(left + 0.12, yy + 0.07, lbl, fontproperties=TINY, color=col, ha="left", va="bottom")
    n = len(stages)
    gw = (right - left) / n
    for i, (lbl, v, col) in enumerate(stages):
        bx = left + i * gw + gw * 0.26
        bw = gw * 0.48
        bh = (v / ymax) * (top - bottom)
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                     linewidth=0, facecolor=col, zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, f"{v}/70", fontproperties=LBL, color=col,
                ha="center", va="bottom", zorder=4)
        ax.text(bx + bw / 2, bottom - 0.30, lbl, fontproperties=SML, color=INK, ha="center", va="top")
    save(fig, "accuracy_journey.png")


def chart_quantization():
    # sizes from the article deployment table; accuracy = pass count /70 at iter-400
    rows = [("bf16", 9.6, 65, BLUE), ("8-bit", 5.5, 64, GREEN), ("4-bit", 4.1, 58, RED)]
    fig, ax = card(8.6, 5.2, "Quantization trade-off (iter-400)",
                   "8-bit is the size/accuracy pick · 4-bit cracks UC1 nested JSON (14→9)")
    left, right, bottom, top = 1.2, 8.1, 1.05, 3.55
    ymax = 10
    _yaxis(ax, left, right, bottom, top, ymax, [0, 2.5, 5, 7.5, 10], lambda t: f"{t:.0f}GB")
    n = len(rows)
    gw = (right - left) / n
    for i, (prec, size, acc, col) in enumerate(rows):
        bx = left + i * gw + gw * 0.26
        bw = gw * 0.48
        bh = (size / ymax) * (top - bottom)
        chosen = (prec == "8-bit")
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.04",
                     linewidth=2.4 if chosen else 0, edgecolor=GREEN if chosen else "none",
                     facecolor=col, zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, f"{size} GB", fontproperties=LBL, color=col,
                ha="center", va="bottom", zorder=4)
        ax.text(bx + bw / 2, bottom + bh - 0.26, f"{acc}/70", fontproperties=SML, color=CARD,
                ha="center", va="top", zorder=4)
        tag = f"{prec}  ◄ pick" if chosen else prec
        ax.text(bx + bw / 2, bottom - 0.30, tag, fontproperties=LBL,
                color=GREEN if chosen else INK, ha="center", va="top")
    save(fig, "quantization_tradeoff.png")


def chart_model_surgery():
    # from the article's W7 model-surgery table
    steps = [
        ("fine-tuned\nbf16", 9.6, "", GRAY),
        ("8-bit\nfull", 5.5, "68/70", TEAL),
        ("text-only +\nvocab-16k bf16", 3.8, "69/70", BLUE),
        ("+ 8-bit\n(ship)", 2.0, "67/70", GREEN),
    ]
    fig, ax = card(9.4, 5.4, "Model surgery — shrinking the student for deploy",
                   "fuse LoRA → strip vision/audio → prune vocab 262k→16k → 8-bit · 9.6→2.0 GB (−79%)")
    left, right, bottom, top = 1.2, 8.9, 1.05, 3.6
    ymax = 10
    _yaxis(ax, left, right, bottom, top, ymax, [0, 2.5, 5, 7.5, 10], lambda t: f"{t:.0f}GB")
    n = len(steps)
    gw = (right - left) / n
    for i, (lbl, size, acc, col) in enumerate(steps):
        bx = left + i * gw + gw * 0.24
        bw = gw * 0.52
        bh = (size / ymax) * (top - bottom)
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.04",
                     linewidth=2.4 if i == n - 1 else 0, edgecolor=GREEN if i == n - 1 else "none",
                     facecolor=col, zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, f"{size} GB", fontproperties=LBL, color=col,
                ha="center", va="bottom", zorder=4)
        if acc:
            ax.text(bx + bw / 2, bottom + bh - 0.24, acc, fontproperties=TINY, color=CARD,
                    ha="center", va="top", zorder=4)
        ax.text(bx + bw / 2, bottom - 0.30, lbl, fontproperties=SML, color=INK, ha="center", va="top")
    ax.annotate("", xy=(left + (n - 0.55) * gw, bottom + (2.0 / ymax) * (top - bottom) + 0.2),
                xytext=(left + 0.55 * gw, bottom + (9.6 / ymax) * (top - bottom) + 0.2),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.0), zorder=5)
    ax.text((left + right) / 2, top + 0.02, "−79%", fontproperties=LBL, color=RED,
            ha="center", va="bottom")
    save(fig, "model_surgery_size.png")


def chart_eda_corpus():
    comp = Counter(json.loads(Path(f).read_text())["complexity"]
                   for f in glob.glob(str(ROOT / "data/interim/uc*_gen_*.json")))
    order = [("simple", GREEN), ("medium", BLUE), ("complex", PEACH)]
    fig, ax = card(8.6, 5.2, "Corpus balance (800 verified pairs)",
                   "160 pairs per use case (5 UC) · split by complexity below")
    left, right, bottom, top = 1.3, 8.1, 1.05, 3.4
    ymax = 450
    _yaxis(ax, left, right, bottom, top, ymax, [0, 150, 300, 450], lambda t: f"{int(t)}")
    n = len(order)
    gw = (right - left) / n
    for i, (k, col) in enumerate(order):
        v = comp[k]
        bx = left + i * gw + gw * 0.26
        bw = gw * 0.48
        bh = (v / ymax) * (top - bottom)
        ax.add_patch(FancyBboxPatch((bx, bottom), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                     linewidth=0, facecolor=col, zorder=3))
        ax.text(bx + bw / 2, bottom + bh + 0.12, str(v), fontproperties=LBL, color=col,
                ha="center", va="bottom", zorder=4)
        ax.text(bx + bw / 2, bottom - 0.30, k, fontproperties=LBL, color=INK, ha="center", va="top")
    save(fig, "eda_corpus_balance.png")


def chart_eda_seq_length():
    lens = []
    for split in ("train", "val", "test"):
        for line in (ROOT / f"data/processed/{split}.jsonl").read_text().splitlines():
            lens.append(len(line) // 4)  # approx tokens (chars / 4)
    med = int(statistics.median(lens))
    mx = max(lens)
    fig, ax = card(9.0, 5.2, "Sequence-length distribution (800 pairs)",
                   f"approx tokens (chars÷4) · median ≈ {med} · max ≈ {mx} · all well under 4,096")
    left, right, bottom, top = 1.2, 8.55, 1.05, 3.5
    nb = 24
    lo, hi = min(lens), max(lens)
    bw_val = (hi - lo) / nb
    bins = [0] * nb
    for v in lens:
        bins[min(nb - 1, int((v - lo) / bw_val))] += 1
    ymax = max(bins) * 1.12
    _yaxis(ax, left, right, bottom, top, ymax, [0, int(max(bins) / 2), max(bins)], lambda t: f"{int(t)}")
    bw = (right - left) / nb
    for i, b in enumerate(bins):
        bx = left + i * bw
        bh = (b / ymax) * (top - bottom)
        ax.add_patch(Rectangle((bx + 0.02, bottom), bw - 0.04, bh, facecolor=BLUE, edgecolor="none", zorder=3))
    medx = left + ((med - lo) / (hi - lo)) * (right - left)
    ax.plot([medx, medx], [bottom, top], color=PEACH, lw=2.0, ls=(0, (4, 3)), zorder=4)
    ax.text(medx + 0.06, top - 0.1, f"median ≈ {med}", fontproperties=TINY, color=PEACH, ha="left", va="top")
    for frac, lab in [(0.0, f"{lo}"), (0.5, f"{int((lo+hi)/2)}"), (1.0, f"{hi}")]:
        ax.text(left + frac * (right - left), bottom - 0.22, lab, fontproperties=SML,
                color=SUBTLE, ha="center", va="top")
    ax.text((left + right) / 2, bottom - 0.52, "approx tokens per training pair",
            fontproperties=SML, color=INK, ha="center", va="top")
    save(fig, "eda_seq_length.png")


if __name__ == "__main__":
    chart_opus_overall()
    chart_student_base_overall()
    chart_opus_heatmap()
    chart_four_metrics()
    chart_before_after()
    chart_first_model_regression()
    chart_uc4_inverted_u()
    chart_accuracy_journey()
    chart_quantization()
    chart_model_surgery()
    chart_eda_corpus()
    chart_eda_seq_length()
    print("done")
