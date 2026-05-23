"""Consolidate bounded heuristic ASR/utility trade-off results.

This script reads existing CSV summaries only. It does not load models, run
inference, modify adapters/cache, or print harmful prompt/output content.
All metrics remain bounded heuristic metrics, not final judged ASR/utility.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "outputs" / "expanded_sensaware_asr_utility_tradeoff_summary.csv",
    ROOT / "outputs" / "sensaware_asr_utility_tradeoff_summary.csv",
    ROOT / "outputs" / "asr_utility_tradeoff_summary.csv",
]
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "consolidated_tradeoff_results.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
GROUP_ORDER = {
    "original": 0,
    "uniform": 1,
    "spectral_only": 2,
    "sensaware_first": 3,
    "sensaware_expanded": 4,
    "other": 99,
}
SOURCE_PRIORITY = {
    "expanded_sensaware_asr_utility_tradeoff_summary.csv": 30,
    "sensaware_asr_utility_tradeoff_summary.csv": 20,
    "asr_utility_tradeoff_summary.csv": 10,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def parse_float(row: dict[str, str], *names: str, default: float | None = None) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def classify_adapter(adapter: str) -> str:
    if adapter == "original":
        return "original"
    if adapter.startswith("uniform_"):
        return "uniform"
    if adapter.startswith("top1_") or adapter.startswith("top3_"):
        return "spectral_only"
    if adapter.startswith("sensaware_top16_") or adapter.startswith("sensaware_top32_"):
        return "sensaware_first"
    if adapter.startswith("sensaware_top128_") or adapter.startswith("sensaware_top224_") or adapter.startswith("sensaware_top336_"):
        return "sensaware_expanded"
    return "other"


def load_source(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    priority = SOURCE_PRIORITY.get(path.name, 0)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            adapter = row.get("adapter", "").strip()
            if not adapter:
                continue
            trigger = parse_float(row, "preliminary_trigger_success_rate")
            clean = parse_float(row, "heuristic_clean_utility_score")
            if trigger is None or clean is None:
                continue
            rows.append(
                {
                    "adapter": adapter,
                    "adapter_group": classify_adapter(adapter),
                    "preliminary_trigger_success_rate": trigger,
                    "trigger_refusal_rate": parse_float(row, "trigger_refusal_rate"),
                    "heuristic_clean_utility_score": clean,
                    "clean_success_rate": parse_float(row, "clean_success_rate"),
                    "clean_refusal_rate": parse_float(row, "clean_refusal_rate"),
                    "too_short_rate": parse_float(row, "too_short_rate", "clean_too_short_rate"),
                    "mean_output_tokens": parse_float(
                        row,
                        "mean_output_tokens",
                        "mean_clean_output_tokens",
                    ),
                    "mean_latency": parse_float(
                        row,
                        "mean_latency",
                        "mean_clean_latency_seconds",
                    ),
                    "is_final_asr": str(row.get("is_final_asr", "False")),
                    "is_final_clean_utility": str(row.get("is_final_clean_utility", "False")),
                    "notes": row.get("notes", "") or row.get("heuristic_tradeoff_note", ""),
                    "source_file": str(path),
                    "source_priority": priority,
                }
            )
    return rows


def dedupe_by_adapter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best.get(row["adapter"])
        if current is None or row["source_priority"] > current["source_priority"]:
            best[row["adapter"]] = row
    return list(best.values())


def add_derived_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original = next((row for row in rows if row["adapter"] == "original"), None)
    if original is None:
        raise ValueError("No original adapter row found; cannot compute deltas.")
    original_trigger = float(original["preliminary_trigger_success_rate"])
    original_clean = float(original["heuristic_clean_utility_score"])
    for row in rows:
        trigger = float(row["preliminary_trigger_success_rate"])
        clean = float(row["heuristic_clean_utility_score"])
        trigger_abs = original_trigger - trigger
        row["trigger_reduction_vs_original_abs"] = round(trigger_abs, 6)
        row["trigger_reduction_vs_original_rel"] = (
            round(trigger_abs / original_trigger, 6) if original_trigger > 0 else None
        )
        row["clean_utility_diff_vs_original"] = round(clean - original_clean, 6)
        row["simple_tradeoff_score"] = round(clean - trigger, 6)
        row["pareto_high_clean_lower_trigger"] = (
            clean >= original_clean - 0.05 and trigger < original_trigger
        )

    for row in rows:
        trigger = float(row["preliminary_trigger_success_rate"])
        clean = float(row["heuristic_clean_utility_score"])
        dominated = False
        for other in rows:
            other_trigger = float(other["preliminary_trigger_success_rate"])
            other_clean = float(other["heuristic_clean_utility_score"])
            if (
                other_trigger <= trigger
                and other_clean >= clean
                and (other_trigger < trigger or other_clean > clean)
            ):
                dominated = True
                break
        row["pareto_non_dominated"] = not dominated
    return rows


def best_by_trigger(rows: list[dict[str, Any]], group: str | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if group is None or row["adapter_group"] == group]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["preliminary_trigger_success_rate"]))


def make_analysis(rows: list[dict[str, Any]], inputs: list[Path]) -> dict[str, Any]:
    best_overall = best_by_trigger(rows)
    best_spectral = best_by_trigger(rows, "spectral_only")
    best_sensaware = min(
        [row for row in rows if row["adapter_group"].startswith("sensaware")],
        key=lambda row: float(row["preliminary_trigger_success_rate"]),
        default=None,
    )
    best_uniform = best_by_trigger(rows, "uniform")
    return {
        "inputs": [str(path) for path in inputs],
        "row_count": len(rows),
        "groups": sorted({row["adapter_group"] for row in rows}),
        "best_overall_by_trigger_rate": best_overall,
        "best_spectral_only_by_trigger_rate": best_spectral,
        "best_sensaware_by_trigger_rate": best_sensaware,
        "best_uniform_by_trigger_rate": best_uniform,
        "sensaware_beats_spectral_only_by_trigger_rate": (
            best_sensaware is not None
            and best_spectral is not None
            and float(best_sensaware["preliminary_trigger_success_rate"])
            < float(best_spectral["preliminary_trigger_success_rate"])
        ),
        "sensaware_beats_uniform_by_trigger_rate": (
            best_sensaware is not None
            and best_uniform is not None
            and float(best_sensaware["preliminary_trigger_success_rate"])
            < float(best_uniform["preliminary_trigger_success_rate"])
        ),
        "caveat": "Bounded heuristic metrics only; not final judged ASR/utility.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "adapter",
        "adapter_group",
        "preliminary_trigger_success_rate",
        "heuristic_clean_utility_score",
        "trigger_reduction_vs_original_abs",
        "trigger_reduction_vs_original_rel",
        "clean_utility_diff_vs_original",
        "simple_tradeoff_score",
        "pareto_high_clean_lower_trigger",
        "pareto_non_dominated",
        "trigger_refusal_rate",
        "clean_success_rate",
        "clean_refusal_rate",
        "too_short_rate",
        "mean_output_tokens",
        "mean_latency",
        "is_final_asr",
        "is_final_clean_utility",
        "source_file",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bounded heuristic trade-off results.")
    parser.add_argument("--inputs", nargs="+", default=[str(path) for path in DEFAULT_INPUTS])
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    input_paths = [Path(path) for path in args.inputs]
    raw_rows: list[dict[str, Any]] = []
    available_inputs: list[Path] = []
    for path in input_paths:
        rows = load_source(path)
        if rows:
            available_inputs.append(path)
            raw_rows.extend(rows)
    if not raw_rows:
        raise RuntimeError("No trade-off rows found in available input files.")
    rows = add_derived_metrics(dedupe_by_adapter(raw_rows))
    rows.sort(
        key=lambda row: (
            GROUP_ORDER.get(row["adapter_group"], 99),
            float(row["preliminary_trigger_success_rate"]),
            row["adapter"],
        )
    )
    analysis = make_analysis(rows, available_inputs)
    output_csv = Path(args.output_csv)
    backup = write_csv(output_csv, rows, timestamp)
    log_path = Path(args.logs_dir) / f"tradeoff_analysis_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "analysis": analysis,
            "output_csv": str(output_csv),
            "backup_csv": str(backup) if backup else None,
            "is_final_asr": False,
            "is_final_clean_utility": False,
        },
    )

    print("Trade-off analysis summary")
    print("- Bounded heuristic metrics only, not final judged ASR/utility")
    print(f"- Consolidated rows: {len(rows)}")
    print(f"- Best overall by trigger rate: {analysis['best_overall_by_trigger_rate']['adapter']}")
    print(f"- Best SensAware by trigger rate: {analysis['best_sensaware_by_trigger_rate']['adapter']}")
    print(f"- Best spectral-only by trigger rate: {analysis['best_spectral_only_by_trigger_rate']['adapter']}")
    print(f"- Best uniform by trigger rate: {analysis['best_uniform_by_trigger_rate']['adapter']}")
    print(
        "- SensAware beats spectral-only by trigger rate: "
        f"{analysis['sensaware_beats_spectral_only_by_trigger_rate']}"
    )
    print(
        "- SensAware beats uniform by trigger rate: "
        f"{analysis['sensaware_beats_uniform_by_trigger_rate']}"
    )
    print(f"- CSV written: {output_csv}")
    print(f"- JSON log written: {log_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
