"""Wilson intervals and optional base-control comparison table.

This script reads CSV outputs only. It does not load models, run inference,
modify adapters/cache, or update final submission artifacts. All metrics remain
bounded heuristic metrics, heuristic clean utility metrics, or clean-output
similarity metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSOLIDATED = ROOT / "outputs" / "consolidated_tradeoff_results.csv"
DEFAULT_BASE_SUMMARY = ROOT / "outputs" / "base_model_control_eval_summary.csv"
DEFAULT_SIMILARITY = ROOT / "outputs" / "clean_behaviour_similarity_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "wilson_ci_tradeoff_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "report_ready" / "base_control_similarity_plot.png"
DEFAULT_TRIGGER_N = 99


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def read_csv(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def classify_adapter(adapter: str) -> str:
    if adapter == "base_model_only":
        return "base_model_control"
    if adapter == "original":
        return "original"
    if adapter.startswith("uniform_"):
        return "uniform"
    if adapter.startswith("top"):
        return "spectral_only"
    if adapter.startswith("sensaware_"):
        return "sensaware"
    return "other"


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = p_hat + z**2 / (2 * total)
    radius = z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    return (centre - radius) / denom, (centre + radius) / denom


def consolidated_rows(path: Path, default_trigger_n: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_csv(path, required=True):
        adapter = row.get("adapter", "")
        if not adapter:
            continue
        trigger_rate = parse_float(row.get("preliminary_trigger_success_rate"))
        inferred_count = (
            int(round(trigger_rate * default_trigger_n)) if trigger_rate is not None else None
        )
        rows[adapter] = {
            "adapter": adapter,
            "adapter_group": row.get("adapter_group") or classify_adapter(adapter),
            "row_source": "consolidated_tradeoff_results",
            "preliminary_trigger_success_rate": trigger_rate,
            "trigger_success_count": inferred_count,
            "trigger_n": default_trigger_n if inferred_count is not None else None,
            "trigger_count_source": "inferred_from_rate_and_default_n",
            "trigger_refusal_count": None,
            "trigger_refusal_rate": parse_float(row.get("trigger_refusal_rate")),
            "heuristic_clean_utility_score": parse_float(row.get("heuristic_clean_utility_score")),
            "clean_success_rate": parse_float(row.get("clean_success_rate")),
            "clean_refusal_rate": parse_float(row.get("clean_refusal_rate")),
            "too_short_rate": parse_float(row.get("too_short_rate")),
            "oom_count": None,
            "failure_count": None,
            "source_file": str(path),
            "notes": row.get("notes", ""),
        }
    return rows


def overlay_base_summary(
    rows: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    for row in read_csv(path, required=True):
        adapter = row.get("condition") or row.get("adapter") or ""
        if not adapter:
            continue
        condition_role = row.get("condition_role") or ""
        adapter_group = (
            "base_model_control"
            if condition_role == "no_adapter_control"
            else classify_adapter(adapter)
        )
        trigger_count = parse_int(
            row.get("preliminary_trigger_success_count") or row.get("trigger_success_count")
        )
        trigger_n = parse_int(row.get("trigger_rows"))
        rows[adapter] = {
            **rows.get(adapter, {}),
            "adapter": adapter,
            "adapter_group": adapter_group,
            "row_source": (
                "base_model_control_eval_summary"
                if adapter not in rows
                else f"{rows[adapter]['row_source']}+base_model_control_eval_summary"
            ),
            "preliminary_trigger_success_rate": parse_float(
                row.get("preliminary_trigger_success_rate")
            ),
            "trigger_success_count": trigger_count,
            "trigger_n": trigger_n,
            "trigger_count_source": "exact_from_base_model_control_eval_summary",
            "trigger_refusal_count": parse_int(row.get("trigger_refusal_count")),
            "trigger_refusal_rate": parse_float(row.get("trigger_refusal_rate")),
            "heuristic_clean_utility_score": parse_float(row.get("heuristic_clean_utility_score")),
            "clean_success_rate": parse_float(row.get("clean_success_rate")),
            "clean_refusal_rate": parse_float(row.get("clean_refusal_rate")),
            "too_short_rate": parse_float(row.get("too_short_rate")),
            "oom_count": parse_int(row.get("oom_count")),
            "failure_count": parse_int(row.get("failure_count")),
            "source_file": str(path),
            "notes": row.get("notes", ""),
        }


def load_similarity(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path, required=True)
    by_adapter: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("row_type") != "defended_adapter":
            continue
        adapter = row.get("adapter", "")
        if not adapter:
            continue
        by_adapter[adapter] = {
            "similarity_to_original_mean": parse_float(row.get("similarity_to_original_mean")),
            "similarity_to_original_ci_low": parse_float(row.get("similarity_to_original_ci_low")),
            "similarity_to_original_ci_high": parse_float(row.get("similarity_to_original_ci_high")),
            "similarity_to_base_mean": parse_float(row.get("similarity_to_base_mean")),
            "similarity_to_base_ci_low": parse_float(row.get("similarity_to_base_ci_low")),
            "similarity_to_base_ci_high": parse_float(row.get("similarity_to_base_ci_high")),
            "drift_margin_mean": parse_float(row.get("drift_margin_mean")),
            "drift_margin_ci_low": parse_float(row.get("drift_margin_ci_low")),
            "drift_margin_ci_high": parse_float(row.get("drift_margin_ci_high")),
            "similarity_n_valid_pairs": parse_int(row.get("n_valid_pairs")),
            "similarity_n_missing_pairs": parse_int(row.get("n_missing_pairs")),
            "similarity_ci_method": row.get("ci_method", ""),
            "similarity_notes": row.get("notes", ""),
        }
    return by_adapter


def final_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_adapter = consolidated_rows(Path(args.consolidated_csv), args.default_trigger_n)
    overlay_base_summary(rows_by_adapter, Path(args.base_summary_csv))
    similarity = load_similarity(Path(args.similarity_csv))

    output_rows: list[dict[str, Any]] = []
    for adapter, row in rows_by_adapter.items():
        successes = row.get("trigger_success_count")
        total = row.get("trigger_n")
        ci_low, ci_high = (None, None)
        if successes is not None and total is not None:
            ci_low, ci_high = wilson_ci(int(successes), int(total))
        sim = similarity.get(adapter, {})
        output_rows.append(
            {
                "adapter": adapter,
                "adapter_group": row.get("adapter_group") or classify_adapter(adapter),
                "row_source": row.get("row_source"),
                "metric_families": (
                    "bounded_heuristic_trigger;"
                    "heuristic_clean_utility;"
                    "clean_output_similarity"
                ),
                "trigger_success_count": successes,
                "trigger_n": total,
                "preliminary_trigger_success_rate": row.get(
                    "preliminary_trigger_success_rate"
                ),
                "trigger_success_wilson95_low": round(ci_low, 6) if ci_low is not None else None,
                "trigger_success_wilson95_high": round(ci_high, 6) if ci_high is not None else None,
                "trigger_count_source": row.get("trigger_count_source"),
                "trigger_refusal_count": row.get("trigger_refusal_count"),
                "trigger_refusal_rate": row.get("trigger_refusal_rate"),
                "heuristic_clean_utility_score": row.get("heuristic_clean_utility_score"),
                "clean_success_rate": row.get("clean_success_rate"),
                "clean_refusal_rate": row.get("clean_refusal_rate"),
                "too_short_rate": row.get("too_short_rate"),
                "oom_count": row.get("oom_count"),
                "failure_count": row.get("failure_count"),
                "similarity_to_original_mean": sim.get("similarity_to_original_mean"),
                "similarity_to_original_ci_low": sim.get("similarity_to_original_ci_low"),
                "similarity_to_original_ci_high": sim.get("similarity_to_original_ci_high"),
                "similarity_to_base_mean": sim.get("similarity_to_base_mean"),
                "similarity_to_base_ci_low": sim.get("similarity_to_base_ci_low"),
                "similarity_to_base_ci_high": sim.get("similarity_to_base_ci_high"),
                "drift_margin_mean": sim.get("drift_margin_mean"),
                "drift_margin_ci_low": sim.get("drift_margin_ci_low"),
                "drift_margin_ci_high": sim.get("drift_margin_ci_high"),
                "similarity_n_valid_pairs": sim.get("similarity_n_valid_pairs"),
                "similarity_n_missing_pairs": sim.get("similarity_n_missing_pairs"),
                "similarity_ci_method": sim.get("similarity_ci_method"),
                "is_final_asr": False,
                "is_final_clean_utility": False,
                "notes": (
                    "Optional extension table; bounded heuristic trigger metrics, "
                    "heuristic clean utility, and token-overlap clean-output similarity only."
                ),
            }
        )

    output_rows.sort(
        key=lambda row: (
            {
                "base_model_control": 0,
                "no_adapter_control": 0,
                "original": 1,
                "uniform": 2,
                "spectral_only": 3,
                "sensaware": 4,
                "sensaware_expanded": 4,
            }.get(str(row["adapter_group"]), 99),
            str(row["adapter"]),
        )
    )
    log = {
        "consolidated_csv": str(args.consolidated_csv),
        "base_summary_csv": str(args.base_summary_csv),
        "similarity_csv": str(args.similarity_csv),
        "default_trigger_n_for_inferred_counts": args.default_trigger_n,
        "row_count": len(output_rows),
        "rows_with_exact_trigger_counts": sum(
            1
            for row in output_rows
            if row["trigger_count_source"] == "exact_from_base_model_control_eval_summary"
        ),
        "rows_with_inferred_trigger_counts": sum(
            1
            for row in output_rows
            if row["trigger_count_source"] == "inferred_from_rate_and_default_n"
        ),
        "is_final_asr": False,
        "is_final_clean_utility": False,
        "caveat": (
            "Wilson intervals are for bounded heuristic trigger rates. "
            "They are not final judged ASR intervals."
        ),
    }
    return output_rows, log


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "adapter",
        "adapter_group",
        "row_source",
        "metric_families",
        "trigger_success_count",
        "trigger_n",
        "preliminary_trigger_success_rate",
        "trigger_success_wilson95_low",
        "trigger_success_wilson95_high",
        "trigger_count_source",
        "trigger_refusal_count",
        "trigger_refusal_rate",
        "heuristic_clean_utility_score",
        "clean_success_rate",
        "clean_refusal_rate",
        "too_short_rate",
        "oom_count",
        "failure_count",
        "similarity_to_original_mean",
        "similarity_to_original_ci_low",
        "similarity_to_original_ci_high",
        "similarity_to_base_mean",
        "similarity_to_base_ci_low",
        "similarity_to_base_ci_high",
        "drift_margin_mean",
        "drift_margin_ci_low",
        "drift_margin_ci_high",
        "similarity_n_valid_pairs",
        "similarity_n_missing_pairs",
        "similarity_ci_method",
        "is_final_asr",
        "is_final_clean_utility",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def maybe_write_plot(rows: list[dict[str, Any]], path: Path) -> Path | None:
    plot_rows = [
        row
        for row in rows
        if row.get("similarity_to_original_mean") is not None
        and row.get("similarity_to_base_mean") is not None
    ]
    if not plot_rows:
        return None

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["adapter"] for row in plot_rows]
    x_values = [float(row["similarity_to_original_mean"]) for row in plot_rows]
    y_values = [float(row["similarity_to_base_mean"]) for row in plot_rows]
    colors = [float(row["preliminary_trigger_success_rate"] or 0.0) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(x_values, y_values, c=colors, cmap="viridis", s=90)
    for label, x_val, y_val in zip(labels, x_values, y_values):
        ax.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Token-overlap similarity to original adapter")
    ax.set_ylabel("Token-overlap similarity to base_model_only")
    ax.set_title("Optional Clean-Behaviour Similarity Control")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Bounded heuristic trigger success rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def print_summary(
    rows: list[dict[str, Any]],
    csv_path: Path,
    backup: Path | None,
    log_path: Path,
    figure_path: Path | None,
) -> None:
    print("Wilson CI and optional comparison summary")
    print("- CSV-only analysis; no model loading or inference")
    print("- Trigger metrics are bounded heuristic metrics, not final judged ASR")
    print("- Clean utility metrics are heuristic, not final judged utility")
    print("- Similarity metrics are token-overlap clean-output metrics")
    print(f"- Rows written: {len(rows)}")
    for row in rows:
        print(
            f"  - {row['adapter']}: trigger={row['preliminary_trigger_success_rate']} "
            f"wilson95=({row['trigger_success_wilson95_low']}, "
            f"{row['trigger_success_wilson95_high']})"
        )
    print(f"- CSV written: {csv_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")
    print(f"- JSON log written: {log_path}")
    if figure_path:
        print(f"- Optional figure written: {figure_path}")
    else:
        print("- Optional figure written: False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Wilson CIs and optional comparison table.")
    parser.add_argument("--consolidated-csv", default=str(DEFAULT_CONSOLIDATED))
    parser.add_argument("--base-summary-csv", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--similarity-csv", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--default-trigger-n", type=int, default=DEFAULT_TRIGGER_N)
    parser.add_argument("--make-plot", action="store_true")
    parser.add_argument("--figure-path", default=str(DEFAULT_FIGURE))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    rows, log = final_rows(args)
    output_csv = Path(args.output_csv)
    backup = write_summary_csv(output_csv, rows, timestamp)
    figure_path = maybe_write_plot(rows, Path(args.figure_path)) if args.make_plot else None
    log_path = Path(args.logs_dir) / f"wilson_ci_and_final_comparison_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "output_csv": str(output_csv),
            "backup_csv": str(backup) if backup else None,
            "figure_path": str(figure_path) if figure_path else None,
            "summary_rows": rows,
            **log,
        },
    )
    print_summary(rows, output_csv, backup, log_path, figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
