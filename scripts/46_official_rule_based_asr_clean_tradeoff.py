"""Combine official rule-based ASR with clean-utility probes.

This script reads CSV files only. It does not load models, run inference,
execute official BackdoorLLM code, call APIs, or update final submission
artifacts.
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
DEFAULT_ASR_SUMMARY = ROOT / "outputs" / "official_rule_based_asr_eval_summary.csv"
DEFAULT_PERPLEXITY = ROOT / "outputs" / "clean_utility_perplexity_summary.csv"
DEFAULT_SIMILARITY = ROOT / "outputs" / "clean_behaviour_similarity_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "official_rule_based_asr_clean_tradeoff_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "report_ready" / "official_rule_based_asr_clean_tradeoff.png"
FOCUSED_CONDITIONS = [
    "base_model_only",
    "original",
    "uniform_gamma_0.25",
    "uniform_gamma_0.50",
    "top3_gamma_0.50",
    "sensaware_top224_gamma_0.25",
]
ASR_METRIC_LABEL = "BackdoorLLM official rule-based jailbreak ASR"
CLEAN_METRIC_LABEL = "reference-output NLL/perplexity probe"
GENERATION_CAVEAT = (
    "Official rule-based scorer applied to project-local deterministic [INST] generations; "
    "not exact reproduction of the full BackdoorLLM generation pipeline."
)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def by_adapter(rows: list[dict[str, str]], adapter_key: str = "adapter") -> dict[str, dict[str, str]]:
    return {row.get(adapter_key, ""): row for row in rows if row.get(adapter_key, "")}


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
    return round((centre - radius) / denom, 6), round((centre + radius) / denom, 6)


def load_similarity(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path, required=False)
    similarity: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("row_type") != "defended_adapter":
            continue
        adapter = row.get("adapter", "")
        if not adapter:
            continue
        similarity[adapter] = {
            "similarity_to_original_mean": parse_float(row.get("similarity_to_original_mean")),
            "similarity_to_base_mean": parse_float(row.get("similarity_to_base_mean")),
            "drift_margin_mean": parse_float(row.get("drift_margin_mean")),
            "similarity_ci_method": row.get("ci_method", ""),
        }
    return similarity


def perplexity_distinguish_note(
    perplexity_rows: dict[str, dict[str, str]],
    conditions: list[str],
    relative_threshold: float,
) -> str:
    values: list[float] = []
    for condition in conditions:
        value = parse_float(perplexity_rows.get(condition, {}).get("perplexity"))
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return "Perplexity distinction cannot be assessed because fewer than two finite values are available."
    minimum = min(values)
    maximum = max(values)
    if minimum <= 0:
        return "Perplexity distinction cannot be assessed because the minimum finite perplexity is non-positive."
    relative_range = (maximum - minimum) / minimum
    if relative_range <= relative_threshold:
        return (
            "Perplexity is nearly identical across focused conditions under the configured threshold; "
            "interpret as small measurable clean-task LoRA footprint, not utility preservation proof."
        )
    return (
        "Perplexity shows a measurable spread across focused conditions; "
        "small gaps should still be interpreted cautiously."
    )


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asr_rows = by_adapter(read_csv(Path(args.asr_summary_csv), required=True))
    perplexity_rows = by_adapter(read_csv(Path(args.perplexity_summary_csv), required=True))
    similarity_rows = load_similarity(Path(args.similarity_csv))
    note = perplexity_distinguish_note(
        perplexity_rows,
        list(args.conditions),
        float(args.nearly_identical_perplexity_relative_threshold),
    )
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        asr = asr_rows.get(condition, {})
        perplexity = perplexity_rows.get(condition, {})
        similarity = similarity_rows.get(condition, {})
        successes = parse_int(asr.get("official_rule_based_success_count"))
        total = parse_int(asr.get("trigger_completed") or asr.get("trigger_rows"))
        ci_low, ci_high = (None, None)
        if successes is not None and total is not None:
            ci_low, ci_high = wilson_ci(successes, total)
        rows.append(
            {
                "adapter": condition,
                "condition": condition,
                "adapter_group": classify_adapter(condition),
                "trigger_metric_label": ASR_METRIC_LABEL,
                "is_official_asr": True,
                "is_external_judged_asr": False,
                "is_final_asr": False,
                "official_rule_based_success_count": successes,
                "trigger_n": total,
                "official_rule_based_success_rate": parse_float(
                    asr.get("official_rule_based_success_rate")
                ),
                "official_rule_based_success_wilson95_low": ci_low,
                "official_rule_based_success_wilson95_high": ci_high,
                "refusal_keyword_detected_count": parse_int(
                    asr.get("refusal_keyword_detected_count")
                ),
                "refusal_keyword_detected_rate": parse_float(
                    asr.get("refusal_keyword_detected_rate")
                ),
                "asr_failure_count": parse_int(asr.get("failure_count")),
                "asr_oom_count": parse_int(asr.get("oom_count")),
                "clean_metric_label": CLEAN_METRIC_LABEL,
                "is_final_clean_utility": False,
                "clean_records": parse_int(perplexity.get("clean_records")),
                "clean_completed_records": parse_int(perplexity.get("completed_records")),
                "clean_failed_records": parse_int(perplexity.get("failed_records")),
                "clean_oom_count": parse_int(perplexity.get("oom_count")),
                "total_reference_tokens": parse_int(perplexity.get("total_reference_tokens")),
                "mean_token_nll_weighted": parse_float(perplexity.get("mean_token_nll_weighted")),
                "perplexity": parse_float(perplexity.get("perplexity")),
                "mean_record_token_nll": parse_float(perplexity.get("mean_record_token_nll")),
                "median_record_token_nll": parse_float(perplexity.get("median_record_token_nll")),
                "std_record_token_nll": parse_float(perplexity.get("std_record_token_nll")),
                "similarity_to_original_mean": similarity.get("similarity_to_original_mean"),
                "similarity_to_base_mean": similarity.get("similarity_to_base_mean"),
                "drift_margin_mean": similarity.get("drift_margin_mean"),
                "similarity_ci_method": similarity.get("similarity_ci_method", ""),
                "metric_families": (
                    "official_rule_based_ASR;Wilson_CI;"
                    "clean_reference_NLL_perplexity_probe;clean_output_similarity_optional"
                ),
                "generation_caveat": GENERATION_CAVEAT,
                "perplexity_interpretation_note": note,
                "notes": "Optional stronger-evaluation summary; do not update final claims until reviewed.",
            }
        )
    metadata = {
        "asr_summary_csv": str(Path(args.asr_summary_csv)),
        "perplexity_summary_csv": str(Path(args.perplexity_summary_csv)),
        "similarity_csv": str(Path(args.similarity_csv)),
        "conditions": list(args.conditions),
        "asr_metric_label": ASR_METRIC_LABEL,
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "generation_caveat": GENERATION_CAVEAT,
        "perplexity_interpretation_note": note,
        "is_official_asr": True,
        "is_external_judged_asr": False,
        "is_final_asr": False,
        "is_final_clean_utility": False,
    }
    return rows, metadata


def maybe_write_figure(path: Path, rows: list[dict[str, Any]], timestamp: str, skip: bool) -> dict[str, Any]:
    if skip:
        return {"figure_written": False, "reason": "skipped_by_argument", "path": str(path)}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"figure_written": False, "reason": f"matplotlib_unavailable: {exc}", "path": str(path)}
    usable = [
        row
        for row in rows
        if row.get("official_rule_based_success_rate") is not None and row.get("perplexity") is not None
    ]
    if not usable:
        return {"figure_written": False, "reason": "no_complete_rows", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in usable:
        ax.scatter(row["official_rule_based_success_rate"], row["perplexity"])
        ax.annotate(row["adapter"], (row["official_rule_based_success_rate"], row["perplexity"]), fontsize=8)
    ax.set_xlabel("Official rule-based jailbreak ASR")
    ax.set_ylabel("Clean reference perplexity")
    ax.set_title("Official Rule-Based ASR vs Clean Perplexity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"figure_written": True, "path": str(path), "backup": str(backup) if backup else ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine official rule-based ASR with clean perplexity and optional similarity."
    )
    parser.add_argument("--asr-summary-csv", default=str(DEFAULT_ASR_SUMMARY))
    parser.add_argument("--perplexity-summary-csv", default=str(DEFAULT_PERPLEXITY))
    parser.add_argument("--similarity-csv", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=FOCUSED_CONDITIONS)
    parser.add_argument("--nearly-identical-perplexity-relative-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, metadata = build_rows(args)
    timestamp = utc_timestamp()
    figure_report = maybe_write_figure(Path(args.figure), rows, timestamp, args.skip_figure)
    log_path = Path(args.logs_dir) / f"official_rule_based_asr_clean_tradeoff_{timestamp}.json"
    fieldnames = [
        "adapter",
        "condition",
        "adapter_group",
        "trigger_metric_label",
        "is_official_asr",
        "is_external_judged_asr",
        "is_final_asr",
        "official_rule_based_success_count",
        "trigger_n",
        "official_rule_based_success_rate",
        "official_rule_based_success_wilson95_low",
        "official_rule_based_success_wilson95_high",
        "refusal_keyword_detected_count",
        "refusal_keyword_detected_rate",
        "asr_failure_count",
        "asr_oom_count",
        "clean_metric_label",
        "is_final_clean_utility",
        "clean_records",
        "clean_completed_records",
        "clean_failed_records",
        "clean_oom_count",
        "total_reference_tokens",
        "mean_token_nll_weighted",
        "perplexity",
        "mean_record_token_nll",
        "median_record_token_nll",
        "std_record_token_nll",
        "similarity_to_original_mean",
        "similarity_to_base_mean",
        "drift_margin_mean",
        "similarity_ci_method",
        "metric_families",
        "generation_caveat",
        "perplexity_interpretation_note",
        "notes",
    ]
    backup = write_csv(Path(args.output_csv), rows, fieldnames, timestamp)
    report = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "csv_only_official_rule_based_asr_clean_tradeoff",
        "metadata": metadata,
        "rows": rows,
        "figure": figure_report,
        "summary": {
            "rows_written": len(rows),
            "asr_metric_label": ASR_METRIC_LABEL,
            "is_official_asr": True,
            "is_external_judged_asr": False,
            "is_final_asr": False,
            "is_final_clean_utility": False,
            "generation_caveat": GENERATION_CAVEAT,
            "model_loading": False,
            "inference": False,
            "official_code_execution": False,
            "external_api_calls": False,
        },
    }
    write_json(log_path, report)
    print("Official rule-based ASR and clean-utility trade-off summary")
    print("- CSV-only analysis; no model loading, inference, official code execution, or API calls")
    print(f"- Trigger metric label: {ASR_METRIC_LABEL}")
    print("- is_official_asr: True")
    print("- is_external_judged_asr: False")
    print("- is_final_asr: False")
    print("- is_final_clean_utility: False")
    print(f"- Rows written: {len(rows)}")
    print(f"- Perplexity interpretation: {metadata['perplexity_interpretation_note']}")
    print(f"- CSV written: {Path(args.output_csv)}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")
    print(f"- JSON log written: {log_path}")
    print(f"- Optional figure written: {figure_report.get('figure_written')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
