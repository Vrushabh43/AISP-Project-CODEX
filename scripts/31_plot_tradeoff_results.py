"""Plot bounded heuristic ASR/utility trade-off results.

This script reads `outputs/consolidated_tradeoff_results.csv` and writes
matplotlib figures only. It does not load models, run inference, or inspect
full prompt/output text.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "outputs" / "consolidated_tradeoff_results.csv"
DEFAULT_FIGURES_DIR = ROOT / "reports" / "figures"
DEFAULT_LOGS_DIR = ROOT / "logs"
GROUP_ORDER = {
    "original": 0,
    "uniform": 1,
    "spectral_only": 2,
    "sensaware_first": 3,
    "sensaware_expanded": 4,
    "other": 99,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Consolidated trade-off CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Consolidated trade-off CSV is empty: {path}")
    for row in rows:
        for field in [
            "preliminary_trigger_success_rate",
            "heuristic_clean_utility_score",
            "trigger_reduction_vs_original_abs",
            "clean_utility_diff_vs_original",
        ]:
            row[field] = float(row[field])
    rows.sort(
        key=lambda row: (
            GROUP_ORDER.get(row.get("adapter_group", "other"), 99),
            row["preliminary_trigger_success_rate"],
            row["adapter"],
        )
    )
    return rows


def short_label(adapter: str) -> str:
    return (
        adapter.replace("sensaware_", "sens_")
        .replace("uniform_gamma_", "uni_")
        .replace("gamma_", "g")
    )


def save_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted({row["adapter_group"] for row in rows}, key=lambda g: GROUP_ORDER.get(g, 99))
    for group in groups:
        group_rows = [row for row in rows if row["adapter_group"] == group]
        ax.scatter(
            [row["preliminary_trigger_success_rate"] for row in group_rows],
            [row["heuristic_clean_utility_score"] for row in group_rows],
            label=group,
        )
        for row in group_rows:
            ax.annotate(
                short_label(row["adapter"]),
                (
                    row["preliminary_trigger_success_rate"],
                    row["heuristic_clean_utility_score"],
                ),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("Preliminary trigger success rate")
    ax.set_ylabel("Heuristic clean utility score")
    ax.set_title("Bounded Heuristic ASR-Utility Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_bar(rows: list[dict[str, Any]], path: Path, field: str, ylabel: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [short_label(row["adapter"]) for row in rows]
    values = [row[field] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reduction_delta(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted({row["adapter_group"] for row in rows}, key=lambda g: GROUP_ORDER.get(g, 99))
    for group in groups:
        group_rows = [row for row in rows if row["adapter_group"] == group]
        ax.scatter(
            [row["trigger_reduction_vs_original_abs"] for row in group_rows],
            [row["clean_utility_diff_vs_original"] for row in group_rows],
            label=group,
        )
        for row in group_rows:
            ax.annotate(
                short_label(row["adapter"]),
                (
                    row["trigger_reduction_vs_original_abs"],
                    row["clean_utility_diff_vs_original"],
                ),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Trigger-rate reduction vs original")
    ax.set_ylabel("Clean-utility delta vs original")
    ax.set_title("Trigger Reduction vs Clean Utility Change")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bounded heuristic trade-off results.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    rows = read_rows(Path(args.input_csv))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "asr_utility_tradeoff_scatter": figures_dir / "asr_utility_tradeoff_scatter.png",
        "trigger_rate_bar_by_method": figures_dir / "trigger_rate_bar_by_method.png",
        "clean_utility_bar_by_method": figures_dir / "clean_utility_bar_by_method.png",
        "trigger_reduction_vs_clean_delta": figures_dir / "trigger_reduction_vs_clean_delta.png",
    }
    save_scatter(rows, paths["asr_utility_tradeoff_scatter"])
    save_bar(
        rows,
        paths["trigger_rate_bar_by_method"],
        "preliminary_trigger_success_rate",
        "Preliminary trigger success rate",
        "Trigger Rate by Method",
    )
    save_bar(
        rows,
        paths["clean_utility_bar_by_method"],
        "heuristic_clean_utility_score",
        "Heuristic clean utility score",
        "Clean Utility by Method",
    )
    save_reduction_delta(rows, paths["trigger_reduction_vs_clean_delta"])

    log_path = Path(args.logs_dir) / f"tradeoff_plots_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "input_csv": str(Path(args.input_csv)),
            "row_count": len(rows),
            "figures": {name: str(path) for name, path in paths.items()},
            "caveat": "Bounded heuristic metrics only; not final judged ASR/utility.",
        },
    )
    print("Trade-off plotting summary")
    print("- Bounded heuristic metrics only, not final judged ASR/utility")
    print(f"- Rows plotted: {len(rows)}")
    for path in paths.values():
        print(f"- Plot written: {path}")
    print(f"- JSON log written: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
