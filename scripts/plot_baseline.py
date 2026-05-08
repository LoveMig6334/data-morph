"""Render plots from a baseline summary.json.

Usage (from project root):
    python scripts/plot_baseline.py                      # latest run
    python scripts/plot_baseline.py results/baseline_... # specific run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
METRIC_LABELS = ["Format\nValidity", "Schema\nCompliance", "Loadability", "Content\nAccuracy"]
METRIC_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def pick_run(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    runs = sorted((PROJECT_ROOT / "results").glob("baseline_*"))
    if not runs:
        raise SystemExit("No baseline runs found under results/")
    return runs[-1]


def plot_overall(summary: dict, out_path: Path) -> None:
    overall = summary["aggregate"]["overall"]
    values = [overall[m] for m in METRICS]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(METRIC_LABELS, values, color=METRIC_COLORS, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title(
        f"Claude Opus baseline — overall scores\n"
        f"n={summary['aggregate']['n_cases']} cases, "
        f"{summary['aggregate']['n_teacher_errors']} teacher errors",
        fontsize=11,
    )
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(3.55, 0.81, "80% target\nfor student", fontsize=8, color="gray",
            va="bottom", ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_use_case(summary: dict, out_path: Path) -> None:
    by_uc = summary["aggregate"]["by_use_case"]
    ucs = sorted(by_uc.keys())
    x = np.arange(len(ucs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label, color) in enumerate(zip(METRICS, METRIC_LABELS, METRIC_COLORS)):
        values = [by_uc[u][m] for u in ucs]
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=label.replace("\n", " "),
               color=color, edgecolor="black", linewidth=0.5)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Baseline scores by use case", fontsize=11)
    ax.set_xticks(x)
    # shorten uc names for display
    short = [u.replace("_", " ").replace("uc", "UC").split(maxsplit=1)[-1][:24] for u in ucs]
    ax.set_xticklabels(short, rotation=15, ha="right", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_case_heatmap(summary: dict, out_path: Path) -> None:
    cases = summary["cases"]
    matrix = np.array([[c["scores"][m] for m in METRICS] for c in cases])
    case_ids = [c["case_id"] for c in cases]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([lbl.replace("\n", " ") for lbl in METRIC_LABELS], fontsize=9)
    ax.set_yticks(range(len(case_ids)))
    ax.set_yticklabels(case_ids, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.35 < v < 0.85 else "white",
                    fontsize=8, fontweight="bold")

    ax.set_title("Per-case scores — green = pass, red = fail", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Score", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    run_dir = pick_run(sys.argv[1] if len(sys.argv) > 1 else None)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_overall(summary, plots_dir / "overall_metrics.png")
    plot_by_use_case(summary, plots_dir / "by_use_case.png")
    plot_per_case_heatmap(summary, plots_dir / "per_case_heatmap.png")

    print(f"Wrote 3 plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
