"""Adapter-file smoke check for expanded sensitivity-aware LoRA adapters.

This script validates adapter files only. It does not load the base model, run
inference, use GPU, modify cached adapters, or print harmful content.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SMOKE_PATH = ROOT / "scripts" / "23_smoke_check_sensitivity_aware_adapters.py"
DEFAULT_VARIANTS = [
    "sensaware_top128_gamma_0.50",
    "sensaware_top128_gamma_0.25",
    "sensaware_top224_gamma_0.50",
    "sensaware_top224_gamma_0.25",
    "sensaware_top336_gamma_0.50",
]
DEFAULT_VARIANTS_DIR = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "expanded_sensaware_adapter_smoke_check_summary.csv"
EXPECTED_VARIANTS = {
    "sensaware_top128_gamma_0.50": {"selected_components": 128, "gamma": 0.50},
    "sensaware_top128_gamma_0.25": {"selected_components": 128, "gamma": 0.25},
    "sensaware_top224_gamma_0.50": {"selected_components": 224, "gamma": 0.50},
    "sensaware_top224_gamma_0.25": {"selected_components": 224, "gamma": 0.25},
    "sensaware_top336_gamma_0.50": {"selected_components": 336, "gamma": 0.50},
}


def load_base_smoke_module() -> Any:
    spec = importlib.util.spec_from_file_location("sensaware_smoke_helpers", BASE_SMOKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {BASE_SMOKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_VARIANTS = EXPECTED_VARIANTS
    return module


base_smoke = load_base_smoke_module()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, backup: Path | None) -> None:
    summary = report["summary"]
    print("Expanded sensitivity-aware adapter smoke check summary")
    print(f"- Variants checked: {summary['variants_checked']}")
    print(f"- Passed: {summary['passed_count']}")
    print(f"- Failed: {summary['failed_count']}")
    for row in report["results"]:
        status = "PASS" if row["passed"] else "FAIL"
        tensor = row["tensor_check"]
        report_check = row["report_check"]
        print(
            f"  - {row['variant_name']}: {status} "
            f"tensors={tensor.get('tensor_count')} "
            f"pairs={tensor.get('complete_pair_count')} "
            f"ranks={tensor.get('unique_ranks')} "
            f"finite={tensor.get('all_finite')} "
            f"modules_edited={report_check.get('modules_edited')}"
        )
    print(f"- Safe to proceed to bounded eval: {summary['safe_to_proceed_to_bounded_eval']}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV summary written: {csv_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check expanded sensitivity-aware adapter files."
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.variants).difference(EXPECTED_VARIANTS))
    if unknown:
        raise ValueError(
            "Unexpected expanded sensitivity-aware variants for this smoke check: "
            + ", ".join(unknown)
        )

    report = base_smoke.build_report(args)
    report["script"] = Path(__file__).name
    report["timestamp_utc"] = utc_timestamp()
    report["expected_variants"] = EXPECTED_VARIANTS
    json_path = Path(args.logs_dir) / (
        f"expanded_sensaware_adapter_smoke_check_{report['timestamp_utc']}.json"
    )
    write_json(json_path, report)
    csv_path = Path(args.summary_csv)
    csv_backup = base_smoke.write_summary_csv(csv_path, report["results"], report["timestamp_utc"])
    print_summary(report, json_path, csv_path, csv_backup)
    return 0 if report["summary"]["safe_to_proceed_to_bounded_eval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
