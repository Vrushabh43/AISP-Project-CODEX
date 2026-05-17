"""Adapter-file smoke check for generated sanitised LoRA adapters.

This script validates files only. It does not load the base model, instantiate
PEFT/Transformers models, run inference, use GPU, or modify the original cached
adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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
    classify_lora_tensor_key,
    locate_adapter_snapshot,
    load_adapter_config,
)


EXPECTED_VARIANTS = {
    "top1_gamma_0.0": {"k": 1, "gamma": 0.0},
    "top1_gamma_0.25": {"k": 1, "gamma": 0.25},
    "top1_gamma_0.50": {"k": 1, "gamma": 0.50},
    "top3_gamma_0.0": {"k": 3, "gamma": 0.0},
    "top3_gamma_0.25": {"k": 3, "gamma": 0.25},
    "top3_gamma_0.50": {"k": 3, "gamma": 0.50},
}
REQUIRED_TARGET_MODULES = {
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
CONFIG_FIELDS = ["peft_type", "task_type", "r", "target_modules", "bias"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple) or isinstance(value, set):
        return sorted(str(item) for item in value)
    return value


def target_hint(module_name: str | None) -> str | None:
    if not module_name:
        return None
    return module_name.split(".")[-1]


def rank_from_shape(kind: str | None, shape: tuple[int, ...]) -> int | None:
    if len(shape) != 2 or kind not in {"A", "B"}:
        return None
    return int(shape[0] if kind == "A" else shape[1])


def inspect_safetensors(path: Path, load_values: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "error": None,
        "tensor_count": 0,
        "lora_A_count": 0,
        "lora_B_count": 0,
        "complete_pair_count": 0,
        "incomplete_pair_count": 0,
        "unique_ranks": [],
        "target_module_hints": [],
        "all_required_target_modules_exist": False,
        "all_finite": None,
        "non_finite_keys": [],
        "tensor_shapes": {},
        "tensor_dtypes": {},
    }
    if not path.exists():
        result["error"] = "adapter_model.safetensors missing"
        return result

    try:
        import torch
        from safetensors import safe_open

        modules: dict[str, set[str]] = defaultdict(set)
        ranks: set[int] = set()
        hints: set[str] = set()
        all_finite = True
        non_finite_keys: list[str] = []

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            for key in keys:
                tensor_slice = handle.get_slice(key)
                shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                kind, module_name = classify_lora_tensor_key(key)
                hint = target_hint(module_name)
                rank = rank_from_shape(kind, shape)
                result["tensor_shapes"][key] = list(shape)
                result["tensor_dtypes"][key] = dtype
                if kind and module_name:
                    modules[module_name].add(kind)
                if rank is not None:
                    ranks.add(rank)
                if hint:
                    hints.add(hint)
                if load_values:
                    tensor = handle.get_tensor(key)
                    if not bool(torch.isfinite(tensor).all().item()):
                        all_finite = False
                        non_finite_keys.append(key)

        complete = sum(1 for kinds in modules.values() if {"A", "B"}.issubset(kinds))
        result.update(
            {
                "ok": True,
                "tensor_count": len(keys),
                "lora_A_count": sum(1 for key in keys if classify_lora_tensor_key(key)[0] == "A"),
                "lora_B_count": sum(1 for key in keys if classify_lora_tensor_key(key)[0] == "B"),
                "complete_pair_count": complete,
                "incomplete_pair_count": len(modules) - complete,
                "unique_ranks": sorted(ranks),
                "target_module_hints": sorted(hints),
                "all_required_target_modules_exist": REQUIRED_TARGET_MODULES.issubset(hints),
                "all_finite": all_finite if load_values else None,
                "non_finite_keys": non_finite_keys,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def compare_config(original: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in CONFIG_FIELDS:
        original_value = normalize_config_value(original.get(field))
        variant_value = normalize_config_value(variant.get(field))
        if original_value != variant_value:
            mismatches.append(
                {
                    "field": field,
                    "original": original_value,
                    "variant": variant_value,
                }
            )
    return {
        "ok": not mismatches,
        "checked_fields": CONFIG_FIELDS,
        "mismatches": mismatches,
    }


def compare_shapes(original: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    original_shapes = original.get("tensor_shapes", {})
    variant_shapes = variant.get("tensor_shapes", {})
    missing = sorted(set(original_shapes).difference(variant_shapes))
    extra = sorted(set(variant_shapes).difference(original_shapes))
    mismatched = []
    for key in sorted(set(original_shapes).intersection(variant_shapes)):
        if original_shapes[key] != variant_shapes[key]:
            mismatched.append(
                {
                    "key": key,
                    "original_shape": original_shapes[key],
                    "variant_shape": variant_shapes[key],
                }
            )
    return {
        "ok": not missing and not extra and not mismatched,
        "missing_keys": missing,
        "extra_keys": extra,
        "shape_mismatches": mismatched,
    }


def validate_report(variant_name: str, report_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(report_path),
        "exists": report_path.exists(),
        "ok": False,
        "error": None,
        "variant_name": None,
        "selected_k": None,
        "gamma": None,
        "modules_edited": None,
        "warnings": [],
        "errors": [],
        "checks": {},
    }
    if not report_path.exists():
        result["error"] = "sanitisation_report.json missing"
        return result
    try:
        report = load_json(report_path)
        variant = report.get("variant", {})
        summary = report.get("summary", {})
        expected = EXPECTED_VARIANTS.get(variant_name)
        result.update(
            {
                "variant_name": variant.get("name") or summary.get("variant_name"),
                "selected_k": variant.get("k") or summary.get("selected_component_count"),
                "gamma": variant.get("gamma") if "gamma" in variant else summary.get("gamma"),
                "modules_edited": summary.get("modules_edited"),
                "warnings": summary.get("warnings", []) or report.get("warnings", []),
                "errors": summary.get("errors", []) or report.get("errors", []),
            }
        )
        checks = {
            "variant_name_matches_folder": result["variant_name"] == variant_name,
            "modules_edited_is_224": result["modules_edited"] == 224,
            "warnings_empty": not result["warnings"],
            "errors_empty": not result["errors"],
        }
        if expected:
            checks["selected_k_matches_expected"] = int(result["selected_k"]) == int(expected["k"])
            checks["gamma_matches_expected"] = abs(float(result["gamma"]) - float(expected["gamma"])) < 1e-12
        else:
            checks["known_variant"] = False
        result["checks"] = checks
        result["ok"] = all(checks.values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_variant(path: Path, original_config: dict[str, Any], original_tensors: dict[str, Any]) -> dict[str, Any]:
    variant_name = path.name
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    report_path = path / "sanitisation_report.json"
    warnings: list[str] = []

    config = {}
    config_check = {"ok": False, "mismatches": [{"error": "adapter_config.json missing"}]}
    if config_path.exists():
        try:
            config = load_json(config_path)
            config_check = compare_config(original_config, config)
        except Exception as exc:
            config_check = {"ok": False, "mismatches": [{"error": f"{type(exc).__name__}: {exc}"}]}

    tensor_check = inspect_safetensors(weights_path, load_values=True)
    shape_check = compare_shapes(original_tensors, tensor_check) if tensor_check["ok"] else {"ok": False}
    report_check = validate_report(variant_name, report_path)

    tensor_expectations = {
        "tensor_count_is_448": tensor_check.get("tensor_count") == 448,
        "lora_A_count_is_224": tensor_check.get("lora_A_count") == 224,
        "lora_B_count_is_224": tensor_check.get("lora_B_count") == 224,
        "complete_pair_count_is_224": tensor_check.get("complete_pair_count") == 224,
        "incomplete_pair_count_is_0": tensor_check.get("incomplete_pair_count") == 0,
        "all_ranks_are_8": tensor_check.get("unique_ranks") == [8],
        "all_required_target_modules_exist": bool(tensor_check.get("all_required_target_modules_exist")),
        "all_finite": tensor_check.get("all_finite") is True,
        "shapes_match_original": shape_check.get("ok") is True,
    }

    required_files = {
        "adapter_config.json": config_path.exists(),
        "adapter_model.safetensors": weights_path.exists(),
        "sanitisation_report.json": report_path.exists(),
    }
    if variant_name not in EXPECTED_VARIANTS:
        warnings.append("Unexpected variant folder name")

    passed = (
        all(required_files.values())
        and config_check.get("ok") is True
        and tensor_check.get("ok") is True
        and all(tensor_expectations.values())
        and report_check.get("ok") is True
    )

    return {
        "variant_name": variant_name,
        "path": str(path),
        "passed": passed,
        "required_files": required_files,
        "config_check": config_check,
        "tensor_check": tensor_check,
        "shape_check": shape_check,
        "tensor_expectations": tensor_expectations,
        "report_check": report_check,
        "warnings": warnings,
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
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
        "modules_edited",
        "selected_k",
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
                    "sanitisation_report_exists": row["required_files"]["sanitisation_report.json"],
                    "config_ok": row["config_check"].get("ok"),
                    "tensor_count": tensor.get("tensor_count"),
                    "lora_A_count": tensor.get("lora_A_count"),
                    "lora_B_count": tensor.get("lora_B_count"),
                    "complete_pair_count": tensor.get("complete_pair_count"),
                    "incomplete_pair_count": tensor.get("incomplete_pair_count"),
                    "unique_ranks": json.dumps(tensor.get("unique_ranks")),
                    "target_module_hints": json.dumps(tensor.get("target_module_hints")),
                    "all_required_target_modules_exist": tensor.get("all_required_target_modules_exist"),
                    "all_finite": tensor.get("all_finite"),
                    "shapes_match_original": row["shape_check"].get("ok"),
                    "report_ok": report.get("ok"),
                    "modules_edited": report.get("modules_edited"),
                    "selected_k": report.get("selected_k"),
                    "gamma": report.get("gamma"),
                    "warnings": json.dumps(row["warnings"] + report.get("warnings", [])),
                    "errors": json.dumps(report.get("errors", [])),
                }
            )
    return path, backup


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timestamp = utc_timestamp()
    original_snapshot = locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    original_config = load_adapter_config(original_snapshot)
    original_weights = adapter_model_path(original_snapshot)
    original_tensors = inspect_safetensors(original_weights, load_values=False)
    if not original_tensors["ok"]:
        raise RuntimeError(f"Could not inspect original adapter tensors: {original_tensors['error']}")

    variants_root = Path(args.variants_dir)
    if not variants_root.exists():
        raise FileNotFoundError(f"Variants directory not found: {variants_root}")
    variant_dirs = sorted(path for path in variants_root.iterdir() if path.is_dir())
    results = [inspect_variant(path, original_config, original_tensors) for path in variant_dirs]

    missing_expected = sorted(set(EXPECTED_VARIANTS).difference(path.name for path in variant_dirs))
    unexpected = sorted(path.name for path in variant_dirs if path.name not in EXPECTED_VARIANTS)
    all_passed = bool(results) and all(result["passed"] for result in results) and not missing_expected

    report = {
        "timestamp_utc": timestamp,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
        },
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "original_adapter": {
            "snapshot_path": str(original_snapshot),
            "adapter_config_path": str(adapter_config_path(original_snapshot)),
            "adapter_model_path": str(original_weights),
            "config_fields_checked": CONFIG_FIELDS,
            "tensor_count": original_tensors["tensor_count"],
            "complete_pair_count": original_tensors["complete_pair_count"],
            "unique_ranks": original_tensors["unique_ranks"],
            "target_module_hints": original_tensors["target_module_hints"],
        },
        "variants_dir": str(variants_root),
        "summary": {
            "variant_count": len(results),
            "expected_variant_count": len(EXPECTED_VARIANTS),
            "passed_count": sum(1 for result in results if result["passed"]),
            "failed_count": sum(1 for result in results if not result["passed"]),
            "missing_expected_variants": missing_expected,
            "unexpected_variants": unexpected,
            "safe_to_proceed_to_peft_loading_smoke_test": all_passed,
        },
        "results": results,
    }
    return report, results


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, csv_backup: Path | None) -> None:
    summary = report["summary"]
    print("Sanitised adapter file smoke check summary")
    print(f"- Variants checked: {summary['variant_count']}")
    print(f"- Passed: {summary['passed_count']}")
    print(f"- Failed: {summary['failed_count']}")
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            "  - "
            f"{result['variant_name']}: {status} "
            f"tensors={result['tensor_check'].get('tensor_count')} "
            f"pairs={result['tensor_check'].get('complete_pair_count')} "
            f"finite={result['tensor_check'].get('all_finite')}"
        )
        warnings = result.get("warnings", [])
        if warnings:
            print(f"    warnings: {warnings}")
        if not result["passed"]:
            print(f"    config_ok={result['config_check'].get('ok')}")
            print(f"    tensor_expectations={result['tensor_expectations']}")
            print(f"    report_ok={result['report_check'].get('ok')}")
    if summary["missing_expected_variants"]:
        print(f"- Missing expected variants: {summary['missing_expected_variants']}")
    if summary["unexpected_variants"]:
        print(f"- Unexpected variants: {summary['unexpected_variants']}")
    print(
        "- Safe to proceed to PEFT loading smoke test: "
        f"{summary['safe_to_proceed_to_peft_loading_smoke_test']}"
    )
    print(f"- JSON log written: {json_path}")
    print(f"- CSV summary written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-dir", default="outputs/sanitised_adapters")
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--csv-path",
        default="outputs/sanitised_adapter_smoke_check_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, rows = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"sanitised_adapter_smoke_check_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(Path(args.csv_path), rows, timestamp)
    print_summary(report, json_path, csv_path, csv_backup)
    return 0 if report["summary"]["safe_to_proceed_to_peft_loading_smoke_test"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
