"""Diagnose why first sensitivity-aware variants underperformed.

This script is adapter/evaluation-output analysis only. It does not load a
model, run inference, modify adapters, or print harmful prompt/output text.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_CSV = ROOT / "outputs" / "clean_sensitivity_component_scores.csv"
DEFAULT_SENS_TRADEOFF = ROOT / "outputs" / "sensaware_asr_utility_tradeoff_summary.csv"
DEFAULT_SPECTRAL_STATS = ROOT / "outputs" / "spectral_stats.csv"
DEFAULT_OLD_TRADEOFF = ROOT / "outputs" / "asr_utility_tradeoff_summary.csv"
DEFAULT_ADAPTER_DIR = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "sensaware_failure_analysis_summary.csv"
SENSAWARE_VARIANTS = [
    "sensaware_top16_gamma_0.50",
    "sensaware_top32_gamma_0.50",
    "sensaware_top32_gamma_0.25",
]
SPECTRAL_VARIANTS = [
    "top1_gamma_0.50",
    "top3_gamma_0.50",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def module_key(row: dict[str, Any]) -> str:
    return str(row.get("module_name", ""))


def component_key(row: dict[str, Any]) -> tuple[str, int]:
    return (module_key(row), int(row.get("component_index", 0)))


def read_variant_report(adapter_dir: Path, variant: str) -> dict[str, Any] | None:
    path = adapter_dir / variant / "sanitisation_report.json"
    if not path.exists():
        return None
    return load_json(path)


def selected_components_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    selected = report.get("selected_components")
    if isinstance(selected, list):
        return selected
    rows: list[dict[str, Any]] = []
    for module in report.get("module_reports", []):
        for index in module.get("selected_indices", []):
            rows.append(
                {
                    "module_name": module.get("module_name"),
                    "layer_id": module.get("layer_id"),
                    "target_module": module.get("target_module"),
                    "component_index": index,
                }
            )
    return rows


def edited_modules_from_report(report: dict[str, Any]) -> set[str]:
    return {str(row.get("module_name")) for row in report.get("module_reports", [])}


def target_layer(row: dict[str, Any]) -> str:
    return f"{row.get('layer_id')}:{row.get('target_module')}"


def summarize_variant(
    variant: str,
    report: dict[str, Any] | None,
    top_spectral_modules: set[str],
    top_spectral_components: set[tuple[str, int]],
) -> dict[str, Any]:
    if report is None:
        return {
            "variant": variant,
            "report_found": False,
            "selected_components": 0,
            "edited_modules": 0,
            "component_index_histogram": "{}",
            "target_module_histogram": "{}",
            "overlap_top50_spectral_modules": 0,
            "overlap_top50_spectral_components": 0,
        }
    selected = selected_components_from_report(report)
    edited = edited_modules_from_report(report)
    component_hist = Counter(int(row.get("component_index", 0)) for row in selected)
    target_hist = Counter(str(row.get("target_module")) for row in selected)
    selected_component_keys = {component_key(row) for row in selected}
    return {
        "variant": variant,
        "report_found": True,
        "selected_components": len(selected),
        "edited_modules": len(edited),
        "component_index_histogram": json.dumps(dict(sorted(component_hist.items()))),
        "target_module_histogram": json.dumps(dict(sorted(target_hist.items()))),
        "selected_layers_targets": ";".join(sorted({target_layer(row) for row in selected})),
        "overlap_top50_spectral_modules": len(edited & top_spectral_modules),
        "overlap_top50_spectral_components": len(selected_component_keys & top_spectral_components),
    }


def tradeoff_by_adapter(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("adapter", ""): row for row in rows}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    sensitivity_rows = read_csv(Path(args.score_csv))
    spectral_rows = read_csv(Path(args.spectral_stats_csv))
    sens_tradeoff = tradeoff_by_adapter(read_csv(Path(args.sensaware_tradeoff_csv)))
    old_tradeoff = tradeoff_by_adapter(read_csv(Path(args.old_tradeoff_csv)))

    top_spectral_by_top1 = sorted(
        spectral_rows,
        key=lambda row: float(row.get("top1_energy_share", 0.0) or 0.0),
        reverse=True,
    )
    top_spectral_modules = {row["module_name"] for row in top_spectral_by_top1[:50]}
    top_spectral_components: set[tuple[str, int]] = set()
    for row in top_spectral_by_top1[:50]:
        top_spectral_components.add((row["module_name"], 0))

    variant_summaries = [
        summarize_variant(
            variant,
            read_variant_report(Path(args.adapters_dir), variant),
            top_spectral_modules,
            top_spectral_components,
        )
        for variant in SENSAWARE_VARIANTS
    ]
    spectral_summaries = [
        summarize_variant(
            variant,
            read_variant_report(Path(args.adapters_dir), variant),
            top_spectral_modules,
            top_spectral_components,
        )
        for variant in SPECTRAL_VARIANTS
    ]

    sensitivity_component_hist = Counter(
        int(row.get("component_index", 0)) for row in sensitivity_rows
    )
    sensitivity_target_hist = Counter(str(row.get("target_module")) for row in sensitivity_rows)
    top_sensitivity = sorted(
        sensitivity_rows,
        key=lambda row: float(row.get("suspiciousness_score_prelim", 0.0) or 0.0),
        reverse=True,
    )
    first_sens_trigger = {
        variant: sens_tradeoff.get(variant, {}).get("preliminary_trigger_success_rate")
        for variant in SENSAWARE_VARIANTS
    }
    baseline_trigger = {
        "original": sens_tradeoff.get("original", old_tradeoff.get("original", {})).get(
            "preliminary_trigger_success_rate"
        ),
        "top1_gamma_0.50": sens_tradeoff.get("top1_gamma_0.50", old_tradeoff.get("top1_gamma_0.50", {})).get(
            "preliminary_trigger_success_rate"
        ),
        "top3_gamma_0.50": sens_tradeoff.get("top3_gamma_0.50", old_tradeoff.get("top3_gamma_0.50", {})).get(
            "preliminary_trigger_success_rate"
        ),
        "uniform_gamma_0.50": sens_tradeoff.get(
            "uniform_gamma_0.50", old_tradeoff.get("uniform_gamma_0.50", {})
        ).get("preliminary_trigger_success_rate"),
        "uniform_gamma_0.25": sens_tradeoff.get(
            "uniform_gamma_0.25", old_tradeoff.get("uniform_gamma_0.25", {})
        ).get("preliminary_trigger_success_rate"),
    }

    diagnosis = {
        "first_sensaware_underperformed": True,
        "primary_reason": (
            "The first sensitivity-aware variants edited only 16 modules and 16-32 "
            "components, while spectral-only top1/top3 edited all 224 modules and "
            "uniform scaling attenuated all adapter updates."
        ),
        "component_distribution_note": (
            "The first sensitivity-aware selection is dominated by component 0 because "
            "the initial candidate set favored highly concentrated modules."
        ),
        "recommended_expanded_variants": [
            "sensaware_top128_gamma_0.50",
            "sensaware_top128_gamma_0.25",
            "sensaware_top224_gamma_0.50",
            "sensaware_top224_gamma_0.25",
            "sensaware_top336_gamma_0.50",
        ],
    }
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "inputs": {
            "score_csv": str(args.score_csv),
            "sensaware_tradeoff_csv": str(args.sensaware_tradeoff_csv),
            "spectral_stats_csv": str(args.spectral_stats_csv),
            "old_tradeoff_csv": str(args.old_tradeoff_csv),
            "adapters_dir": str(args.adapters_dir),
        },
        "sensitivity_score_rows": len(sensitivity_rows),
        "sensitivity_component_index_histogram": dict(sorted(sensitivity_component_hist.items())),
        "sensitivity_target_module_histogram": dict(sorted(sensitivity_target_hist.items())),
        "top_sensitivity_rows": top_sensitivity[:20],
        "first_sensaware_variant_summaries": variant_summaries,
        "spectral_baseline_summaries": spectral_summaries,
        "baseline_trigger_rates": baseline_trigger,
        "first_sensaware_trigger_rates": first_sens_trigger,
        "diagnosis": diagnosis,
        "caveat": "Heuristic bounded diagnosis only; not a final research claim.",
    }


def write_summary_csv(path: Path, report: dict[str, Any]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, report["timestamp_utc"])
    fieldnames = [
        "row_type",
        "name",
        "report_found",
        "selected_components",
        "edited_modules",
        "component_index_histogram",
        "target_module_histogram",
        "overlap_top50_spectral_modules",
        "overlap_top50_spectral_components",
        "trigger_rate",
        "note",
    ]
    rows: list[dict[str, Any]] = []
    for item in report["first_sensaware_variant_summaries"]:
        rows.append(
            {
                "row_type": "first_sensaware",
                "name": item["variant"],
                "report_found": item["report_found"],
                "selected_components": item["selected_components"],
                "edited_modules": item["edited_modules"],
                "component_index_histogram": item["component_index_histogram"],
                "target_module_histogram": item["target_module_histogram"],
                "overlap_top50_spectral_modules": item["overlap_top50_spectral_modules"],
                "overlap_top50_spectral_components": item["overlap_top50_spectral_components"],
                "trigger_rate": report["first_sensaware_trigger_rates"].get(item["variant"]),
                "note": "First sensitivity-aware variant.",
            }
        )
    for item in report["spectral_baseline_summaries"]:
        rows.append(
            {
                "row_type": "spectral_baseline",
                "name": item["variant"],
                "report_found": item["report_found"],
                "selected_components": item["selected_components"],
                "edited_modules": item["edited_modules"],
                "component_index_histogram": item["component_index_histogram"],
                "target_module_histogram": item["target_module_histogram"],
                "overlap_top50_spectral_modules": item["overlap_top50_spectral_modules"],
                "overlap_top50_spectral_components": item["overlap_top50_spectral_components"],
                "trigger_rate": report["baseline_trigger_rates"].get(item["variant"]),
                "note": "Baseline edited many more modules/components.",
            }
        )
    for name in report["diagnosis"]["recommended_expanded_variants"]:
        rows.append(
            {
                "row_type": "recommendation",
                "name": name,
                "report_found": "",
                "selected_components": "",
                "edited_modules": "",
                "component_index_histogram": "",
                "target_module_histogram": "",
                "overlap_top50_spectral_modules": "",
                "overlap_top50_spectral_components": "",
                "trigger_rate": "",
                "note": "Generate from expanded clean-sensitivity scores.",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, backup: Path | None) -> None:
    print("Sensitivity-aware failure analysis summary")
    print(f"- Sensitivity score rows: {report['sensitivity_score_rows']}")
    print("- First sensitivity-aware variants:")
    for row in report["first_sensaware_variant_summaries"]:
        print(
            f"  - {row['variant']}: components={row['selected_components']} "
            f"modules={row['edited_modules']} trigger_rate="
            f"{report['first_sensaware_trigger_rates'].get(row['variant'])}"
        )
    print("- Spectral baselines:")
    for row in report["spectral_baseline_summaries"]:
        print(
            f"  - {row['variant']}: components={row['selected_components']} "
            f"modules={row['edited_modules']} trigger_rate="
            f"{report['baseline_trigger_rates'].get(row['variant'])}"
        )
    print(f"- Diagnosis: {report['diagnosis']['primary_reason']}")
    print("- Recommended expanded variants:")
    for name in report["diagnosis"]["recommended_expanded_variants"]:
        print(f"  - {name}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV summary written: {csv_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze first sensitivity-aware underperformance.")
    parser.add_argument("--score-csv", default=str(DEFAULT_SCORE_CSV))
    parser.add_argument("--sensaware-tradeoff-csv", default=str(DEFAULT_SENS_TRADEOFF))
    parser.add_argument("--spectral-stats-csv", default=str(DEFAULT_SPECTRAL_STATS))
    parser.add_argument("--old-tradeoff-csv", default=str(DEFAULT_OLD_TRADEOFF))
    parser.add_argument("--adapters-dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    json_path = Path(args.logs_dir) / f"sensaware_failure_analysis_{report['timestamp_utc']}.json"
    write_json(json_path, report)
    csv_path = Path(args.summary_csv)
    backup = write_summary_csv(csv_path, report)
    print_summary(report, json_path, csv_path, backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
