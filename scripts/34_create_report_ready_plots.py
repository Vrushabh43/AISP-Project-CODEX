"""Create polished report-ready plots for key bounded results.

This script reads `outputs/report_ready_main_results.csv` only. It does not
load models, run inference, modify adapters/cache, or inspect prompt/output
text.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "outputs" / "report_ready_main_results.csv"
DEFAULT_FIGURES_DIR = ROOT / "reports" / "figures" / "report_ready"
PLOT_FILES = {
    "scatter": "report_tradeoff_scatter_key_methods.png",
    "trigger_rate": "report_trigger_rate_key_methods.png",
    "trigger_reduction": "report_trigger_reduction_key_methods.png",
    "clean_utility": "report_clean_utility_key_methods.png",
}
GROUP_ORDER = {
    "original": 0,
    "uniform": 1,
    "spectral_only": 2,
    "sensaware_expanded": 3,
}
PLOT_LABELS = {
    "original": "Original",
    "uniform_gamma_0.25": "Uniform .25",
    "uniform_gamma_0.50": "Uniform .50",
    "top3_gamma_0.50": "Top-3 .50",
    "top1_gamma_0.50": "Top-1 .50",
    "sensaware_top224_gamma_0.25": "SensAware 224/.25",
    "sensaware_top128_gamma_0.25": "SensAware 128/.25",
    "sensaware_top336_gamma_0.50": "SensAware 336/.50",
}
HIGHLIGHT_ADAPTERS = {
    "original",
    "uniform_gamma_0.25",
    "sensaware_top224_gamma_0.25",
}
ANNOTATION_OFFSETS = {
    "original": (-50, 6),
    "uniform_gamma_0.25": (8, 12),
    "uniform_gamma_0.50": (10, -18),
    "top3_gamma_0.50": (8, -22),
    "top1_gamma_0.50": (10, -6),
    "sensaware_top224_gamma_0.25": (10, -8),
    "sensaware_top128_gamma_0.25": (8, 16),
    "sensaware_top336_gamma_0.50": (8, 10),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Report-ready result CSV not found: {path}. "
            "Run scripts/33_create_report_ready_results.py first."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Report-ready result CSV is empty: {path}")
    for row in rows:
        for field in [
            "preliminary_trigger_success_rate",
            "trigger_reduction_vs_original",
            "heuristic_clean_utility_score",
            "clean_delta_vs_original",
            "tradeoff_score",
        ]:
            row[field] = float(row[field])
    return rows


def ordered_rows(rows: list[dict[str, Any]], by: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row[by], PLOT_LABELS.get(row["adapter"], row["adapter"])))


def group_color(group: str) -> str:
    # Matplotlib default tab colors, fixed for readability.
    return {
        "original": "tab:blue",
        "uniform": "tab:orange",
        "spectral_only": "tab:green",
        "sensaware_expanded": "tab:purple",
    }.get(group, "tab:gray")


def save_scatter(rows: list[dict[str, Any]], path: Path, timestamp: str) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backup = backup_existing(path, timestamp)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for group in sorted({row["method_group"] for row in rows}, key=lambda g: GROUP_ORDER.get(g, 99)):
        group_rows = [row for row in rows if row["method_group"] == group]
        ax.scatter(
            [row["preliminary_trigger_success_rate"] for row in group_rows],
            [row["heuristic_clean_utility_score"] for row in group_rows],
            label=group.replace("_", " "),
            s=[95 if row["adapter"] in HIGHLIGHT_ADAPTERS else 55 for row in group_rows],
            c=group_color(group),
            edgecolors=["black" if row["adapter"] in HIGHLIGHT_ADAPTERS else "none" for row in group_rows],
            linewidths=[1.1 if row["adapter"] in HIGHLIGHT_ADAPTERS else 0.0 for row in group_rows],
        )
        for row in group_rows:
            dx, dy = ANNOTATION_OFFSETS.get(row["adapter"], (6, 6))
            ax.annotate(
                PLOT_LABELS.get(row["adapter"], row["adapter"]),
                (
                    row["preliminary_trigger_success_rate"],
                    row["heuristic_clean_utility_score"],
                ),
                fontsize=8,
                xytext=(dx, dy),
                textcoords="offset points",
                arrowprops=None,
            )
    ax.set_xlabel("Preliminary trigger success rate")
    ax.set_ylabel("Heuristic clean utility score")
    ax.set_title("Bounded Heuristic ASR-Utility Trade-off")
    ax.set_xlim(-0.025, 0.64)
    ax.set_ylim(0.925, 0.985)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return backup


def save_bar(
    rows: list[dict[str, Any]],
    path: Path,
    timestamp: str,
    field: str,
    ylabel: str,
    title: str,
    ascending: bool = True,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backup = backup_existing(path, timestamp)
    plot_rows = sorted(rows, key=lambda row: row[field], reverse=not ascending)
    labels = [PLOT_LABELS.get(row["adapter"], row["adapter"]) for row in plot_rows]
    values = [row[field] for row in plot_rows]
    colors = [group_color(row["method_group"]) for row in plot_rows]
    edges = ["black" if row["adapter"] in HIGHLIGHT_ADAPTERS else "none" for row in plot_rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(labels, values, color=colors, edgecolor=edges, linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=28, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready bounded result plots.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    rows = read_rows(Path(args.input_csv))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    scatter_path = figures_dir / PLOT_FILES["scatter"]
    backups.append(save_scatter(rows, scatter_path, timestamp))
    trigger_path = figures_dir / PLOT_FILES["trigger_rate"]
    backups.append(
        save_bar(
            rows,
            trigger_path,
            timestamp,
            "preliminary_trigger_success_rate",
            "Preliminary trigger success rate",
            "Trigger Rate by Key Method",
            ascending=True,
        )
    )
    reduction_path = figures_dir / PLOT_FILES["trigger_reduction"]
    backups.append(
        save_bar(
            rows,
            reduction_path,
            timestamp,
            "trigger_reduction_vs_original",
            "Trigger-rate reduction vs original",
            "Trigger Reduction by Key Method",
            ascending=False,
        )
    )
    clean_path = figures_dir / PLOT_FILES["clean_utility"]
    backups.append(
        save_bar(
            rows,
            clean_path,
            timestamp,
            "heuristic_clean_utility_score",
            "Heuristic clean utility score",
            "Clean Utility by Key Method",
            ascending=False,
        )
    )
    print("Report-ready plotting summary")
    print("- Bounded heuristic metrics only, not final judged ASR/utility")
    print(f"- Rows plotted: {len(rows)}")
    for path in [scatter_path, trigger_path, reduction_path, clean_path]:
        print(f"- Plot written: {path}")
    for backup in backups:
        if backup:
            print(f"- Previous plot backed up to: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
