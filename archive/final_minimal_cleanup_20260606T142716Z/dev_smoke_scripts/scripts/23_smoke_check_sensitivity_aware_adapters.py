"""Adapter-file smoke check for corrected sensitivity-aware LoRA adapters.

This script validates adapter files only. It does not load the base model, run
inference, use GPU, modify cached adapters, or print harmful content.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import (  # noqa: E402
    adapter_config_path,
    adapter_model_path,
    load_adapter_config,
    locate_adapter_snapshot,
)


HELPER_PATH = ROOT / "scripts" / "08_smoke_check_sanitised_adapters.py"
DEFAULT_VARIANTS = [
    "sensaware_top16_gamma_0.50",
    "sensaware_top32_gamma_0.50",
    "sensaware_top32_gamma_0.25",
]
DEFAULT_VARIANTS_DIR = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "sensaware_adapter_smoke_check_summary.csv"
EXPECTED_VARIANTS = {
    "sensaware_top16_gamma_0.50": {"selected_components": 16, "gamma": 0.50},
    "sensaware_top32_gamma_0.50": {"selected_components": 32, "gamma": 0.50},
    "sensaware_top32_gamma_0.25": {"selected_components": 32, "gamma": 0.25},
}


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("sanitised_smoke_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sensaware_report(variant_name: str, report_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(report_path),
        "exists": report_path.exists(),
        "ok": False,
        "error": None,
        "variant_name": None,
        "selected_component_count": None,
        "selected_module_count": None,
        "modules_edited": None,
        "gamma": None,
        "warnings": [],
        "errors": [],
        "checks": {},
    }
    if not report_path.exists():
        result["error"] = "sanitisation_report.json missing"
        return result
    try:
        report = load_json(report_path)
        summary = report.get("summary", {})
        variant = report.get("variant", {})
        expected = EXPECTED_VARIANTS[variant_name]
        result.update(
            {
                "variant_name": report.get("variant_name")
                or summary.get("variant_name")
                or variant.get("name"),
                "selected_component_count": report.get("selected_component_count")
                or summary.get("selected_component_count"),
                "selected_module_count": report.get("selected_module_count")
                or summary.get("selected_module_count"),
                "modules_edited": report.get("modules_edited") or summary.get("modules_edited"),
                "gamma": report.get("gamma") or summary.get("gamma") or variant.get("gamma"),
                "warnings": report.get("warnings", []) or summary.get("warnings", []),
                "errors": report.get("errors", []) or summary.get("errors", []),
            }
        )
        checks = {
            "variant_name_matches_folder": result["variant_name"] == variant_name,
            "selected_components_match_expected": int(result["selected_component_count"])
            == int(expected["selected_components"]),
            "gamma_matches_expected": abs(float(result["gamma"]) - float(expected["gamma"]))
            < 1e-12,
            "modules_edited_positive": int(result["modules_edited"]) > 0,
            "selected_module_count_positive": int(result["selected_module_count"]) > 0,
            "warnings_empty": not result["warnings"],
            "errors_empty": not result["errors"],
        }
        result["checks"] = checks
        result["ok"] = all(checks.values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_variant(
    variant_name: str,
    variants_dir: Path,
    original_config: dict[str, Any],
    original_tensors: dict[str, Any],
) -> dict[str, Any]:
    path = variants_dir / variant_name
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    report_path = path / "sanitisation_report.json"
    config_check = {"ok": False, "mismatches": [{"error": "adapter_config.json missing"}]}
    if config_path.exists():
        try:
            config_check = helper.compare_config(original_config, load_json(config_path))
        except Exception as exc:
            config_check = {"ok": False, "mismatches": [{"error": f"{type(exc).__name__}: {exc}"}]}

    tensor_check = helper.inspect_safetensors(weights_path, load_values=True)
    shape_check = (
        helper.compare_shapes(original_tensors, tensor_check)
        if tensor_check.get("ok")
        else {"ok": False}
    )
    report_check = validate_sensaware_report(variant_name, report_path)
    required_files = {
        "adapter_config.json": config_path.exists(),
        "adapter_model.safetensors": weights_path.exists(),
        "sanitisation_report.json": report_path.exists(),
    }
    tensor_expectations = {
        "tensor_count_is_448": tensor_check.get("tensor_count") == 448,
        "lora_A_count_is_224": tensor_check.get("lora_A_count") == 224,
        "lora_B_count_is_224": tensor_check.get("lora_B_count") == 224,
        "complete_pair_count_is_224": tensor_check.get("complete_pair_count") == 224,
        "incomplete_pair_count_is_0": tensor_check.get("incomplete_pair_count") == 0,
        "all_ranks_are_8": tensor_check.get("unique_ranks") == [8],
        "all_required_target_modules_exist": bool(
            tensor_check.get("all_required_target_modules_exist")
        ),
        "all_finite": tensor_check.get("all_finite") is True,
        "shapes_match_original": shape_check.get("ok") is True,
    }
    passed = (
        path.exists()
        and all(required_files.values())
        and config_check.get("ok") is True
        and tensor_check.get("ok") is True
        and all(tensor_expectations.values())
        and report_check.get("ok") is True
    )
    return {
        "variant_name": variant_name,
        "path": str(path),
        "exists": path.exists(),
        "passed": passed,
        "required_files": required_files,
        "config_check": config_check,
        "tensor_check": tensor_check,
        "shape_check": shape_check,
        "tensor_expectations": tensor_expectations,
        "report_check": report_check,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "variant_name",
        "passed",
        "adapter_config_exists",
        "adapter_model_exists",
        "sanitisation_report_exists",
        "config_ok",
        "tensor_count",
        "lora_A_count",
        "lora_B_count",
        "complete_pair_count",
        "incomplete_pair_count",
        "unique_ranks",
        "target_module_hints",
        "all_required_target_modules_exist",
        "all_finite",
        "shapes_match_original",
        "report_ok",
        "selected_component_count",
        "selected_module_count",
        "modules_edited",
        "gamma",
        "warnings",
        "errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            tensor = row["tensor_check"]
            report = row["report_check"]
            writer.writerow(
                {
                    "variant_name": row["variant_name"],
                    "passed": row["passed"],
                    "adapter_config_exists": row["required_files"]["adapter_config.json"],
                    "adapter_model_exists": row["required_files"]["adapter_model.safetensors"],
                    "sanitisation_report_exists": row["required_files"][
                        "sanitisation_report.json"
                    ],
                    "config_ok": row["config_check"].get("ok"),
                    "tensor_count": tensor.get("tensor_count"),
                    "lora_A_count": tensor.get("lora_A_count"),
                    "lora_B_count": tensor.get("lora_B_count"),
                    "complete_pair_count": tensor.get("complete_pair_count"),
                    "incomplete_pair_count": tensor.get("incomplete_pair_count"),
                    "unique_ranks": json.dumps(tensor.get("unique_ranks")),
                    "target_module_hints": json.dumps(tensor.get("target_module_hints")),
                    "all_required_target_modules_exist": tensor.get(
                        "all_required_target_modules_exist"
                    ),
                    "all_finite": tensor.get("all_finite"),
                    "shapes_match_original": row["shape_check"].get("ok"),
                    "report_ok": report.get("ok"),
                    "selected_component_count": report.get("selected_component_count"),
                    "selected_module_count": report.get("selected_module_count"),
                    "modules_edited": report.get("modules_edited"),
                    "gamma": report.get("gamma"),
                    "warnings": json.dumps(report.get("warnings", [])),
                    "errors": json.dumps(report.get("errors", [])),
                }
            )
    return backup


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    original_snapshot = locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    original_config = load_adapter_config(original_snapshot)
    original_tensors = helper.inspect_safetensors(adapter_model_path(original_snapshot), load_values=False)
    if not original_tensors["ok"]:
        raise RuntimeError(f"Could not inspect original adapter tensors: {original_tensors['error']}")

    variants_dir = Path(args.variants_dir)
    results = [
        inspect_variant(name, variants_dir, original_config, original_tensors)
        for name in args.variants
    ]
    summary = {
        "variants_checked": len(results),
        "passed_count": sum(1 for row in results if row["passed"]),
        "failed_count": sum(1 for row in results if not row["passed"]),
        "safe_to_proceed_to_bounded_eval": all(row["passed"] for row in results),
    }
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "original_adapter": {
            "snapshot_path": str(original_snapshot),
            "adapter_config_path": str(adapter_config_path(original_snapshot)),
            "adapter_model_path": str(adapter_model_path(original_snapshot)),
        },
        "variants_dir": str(variants_dir),
        "variants": args.variants,
        "summary": summary,
        "results": results,
    }


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, backup: Path | None) -> None:
    summary = report["summary"]
    print("Sensitivity-aware adapter smoke check summary")
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
    parser = argparse.ArgumentParser(description="Smoke-check sensitivity-aware adapter files.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    json_path = Path(args.logs_dir) / f"sensaware_adapter_smoke_check_{report['timestamp_utc']}.json"
    write_json(json_path, report)
    csv_path = Path(args.summary_csv)
    csv_backup = write_summary_csv(csv_path, report["results"], report["timestamp_utc"])
    print_summary(report, json_path, csv_path, csv_backup)
    return 0 if report["summary"]["safe_to_proceed_to_bounded_eval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
