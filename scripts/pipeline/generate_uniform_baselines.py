"""Generate uniform-scaling LoRA adapter baselines.

This script edits only the BackdoorLLM adapter tensors. It does not load a base
model, run inference, or use GPU. Uniform scaling is implemented by multiplying
every LoRA B tensor by gamma while keeping A unchanged, so the effective LoRA
update becomes gamma * (B @ A).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import (  # noqa: E402
    ADAPTER_ID,
    adapter_config_path,
    classify_lora_tensor_key,
    locate_adapter_snapshot,
)


VARIANTS = [
    {"name": "uniform_gamma_0.50", "gamma": 0.50},
    {"name": "uniform_gamma_0.25", "gamma": 0.25},
]
REQUIRED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass(frozen=True)
class PairSpec:
    module_name: str
    target_module: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    rank: int | None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    path.replace(backup)
    return backup


def backup_existing_file(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def infer_target_module(module_name: str) -> str:
    return module_name.split(".")[-1]


def rank_from_shapes(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> int | None:
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    return int(a_shape[0]) if int(a_shape[0]) == int(b_shape[1]) else None


def group_pairs(state: dict[str, Any]) -> tuple[list[PairSpec], list[str]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    warnings: list[str] = []
    for key, tensor in sorted(state.items()):
        kind, module_name = classify_lora_tensor_key(str(key))
        if kind is None or module_name is None:
            continue
        grouped[module_name][kind] = {
            "key": str(key),
            "shape": tuple(int(dim) for dim in tensor.shape),
        }

    pairs: list[PairSpec] = []
    for module_name in sorted(grouped):
        item = grouped[module_name]
        if "A" not in item or "B" not in item:
            warnings.append(f"Incomplete LoRA pair for {module_name}")
            continue
        rank = rank_from_shapes(item["A"]["shape"], item["B"]["shape"])
        if rank is None:
            warnings.append(
                f"Rank mismatch for {module_name}: A={item['A']['shape']} B={item['B']['shape']}"
            )
        pairs.append(
            PairSpec(
                module_name=module_name,
                target_module=infer_target_module(module_name),
                a_key=item["A"]["key"],
                b_key=item["B"]["key"],
                a_shape=item["A"]["shape"],
                b_shape=item["B"]["shape"],
                rank=rank,
            )
        )
    return pairs, warnings


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    pairs, pair_warnings = group_pairs(state)
    a_count = 0
    b_count = 0
    target_modules = set()
    ranks = set()
    finite = True
    nonfinite_keys: list[str] = []
    for key, tensor in state.items():
        kind, module_name = classify_lora_tensor_key(str(key))
        if kind == "A":
            a_count += 1
        elif kind == "B":
            b_count += 1
        if module_name:
            target_modules.add(infer_target_module(module_name))
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            finite = False
            nonfinite_keys.append(str(key))
    for pair in pairs:
        if pair.rank is not None:
            ranks.add(int(pair.rank))
    missing_targets = sorted(REQUIRED_TARGET_MODULES - target_modules)
    return {
        "tensor_count": len(state),
        "lora_a_count": a_count,
        "lora_b_count": b_count,
        "complete_ab_pairs": sum(1 for pair in pairs if pair.rank is not None),
        "pair_count": len(pairs),
        "ranks": sorted(ranks),
        "target_modules": sorted(target_modules),
        "required_target_modules_present": not missing_targets,
        "missing_target_modules": missing_targets,
        "all_finite": finite,
        "nonfinite_keys": nonfinite_keys,
        "pair_warnings": pair_warnings,
    }


def create_variant_state(state: dict[str, Any], gamma: float) -> dict[str, Any]:
    scaled: dict[str, Any] = {}
    for key, tensor in state.items():
        kind, _module_name = classify_lora_tensor_key(str(key))
        if kind == "B":
            scaled[key] = (tensor * gamma).to(dtype=tensor.dtype)
        else:
            scaled[key] = tensor.detach().clone()
    return scaled


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing_file(path, timestamp)
    fieldnames = [
        "variant",
        "gamma",
        "tensor_count",
        "lora_a_count",
        "lora_b_count",
        "complete_ab_pairs",
        "ranks",
        "required_target_modules_present",
        "all_finite",
        "warnings",
        "variant_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate uniform-scaled LoRA adapter baselines.")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "sanitised_adapters"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument(
        "--summary-csv",
        default=str(ROOT / "outputs" / "uniform_scaling_adapter_generation_summary.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()

    import torch
    from safetensors.torch import load_file, save_file

    snapshot = locate_adapter_snapshot(args.adapter_path, cache_roots=args.cache_root)
    config_path = adapter_config_path(snapshot)
    model_path = snapshot / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {model_path}")

    original_state = load_file(str(model_path), device="cpu")
    original_validation = validate_state(original_state)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    backed_up_dirs: list[str] = []

    with torch.no_grad():
        for variant in VARIANTS:
            name = str(variant["name"])
            gamma = float(variant["gamma"])
            variant_dir = output_root / name
            backup = backup_existing(variant_dir, timestamp)
            if backup is not None:
                backed_up_dirs.append(str(backup))
            variant_dir.mkdir(parents=True, exist_ok=True)

            variant_state = create_variant_state(original_state, gamma)
            validation = validate_state(variant_state)
            shutil.copy2(config_path, variant_dir / "adapter_config.json")
            save_file(variant_state, str(variant_dir / "adapter_model.safetensors"))

            warnings = []
            if validation["tensor_count"] != 448:
                warnings.append(f"Expected 448 tensors, got {validation['tensor_count']}")
            if validation["complete_ab_pairs"] != 224:
                warnings.append(f"Expected 224 complete A/B pairs, got {validation['complete_ab_pairs']}")
            if validation["ranks"] != [8]:
                warnings.append(f"Expected rank [8], got {validation['ranks']}")
            if not validation["required_target_modules_present"]:
                warnings.append(f"Missing target modules: {validation['missing_target_modules']}")
            if not validation["all_finite"]:
                warnings.append(f"Non-finite tensors: {validation['nonfinite_keys'][:5]}")
            warnings.extend(validation["pair_warnings"])

            report = {
                "variant_name": name,
                "method": "uniform_adapter_scaling",
                "gamma": gamma,
                "scaling_rule": "LoRA B tensors multiplied by gamma; A tensors unchanged.",
                "effective_delta_w_rule": "DeltaW_new = gamma * (B @ A)",
                "source_adapter": ADAPTER_ID,
                "source_snapshot": str(snapshot),
                "validation": validation,
                "original_validation": original_validation,
                "warnings": warnings,
                "errors": [],
                "is_baseline": True,
            }
            write_json(variant_dir / "sanitisation_report.json", report)
            variant_reports.append(report)
            summary_rows.append(
                {
                    "variant": name,
                    "gamma": gamma,
                    "tensor_count": validation["tensor_count"],
                    "lora_a_count": validation["lora_a_count"],
                    "lora_b_count": validation["lora_b_count"],
                    "complete_ab_pairs": validation["complete_ab_pairs"],
                    "ranks": json.dumps(validation["ranks"]),
                    "required_target_modules_present": validation["required_target_modules_present"],
                    "all_finite": validation["all_finite"],
                    "warnings": " | ".join(warnings),
                    "variant_dir": str(variant_dir),
                }
            )

    summary_csv = Path(args.summary_csv)
    summary_backup = write_summary_csv(summary_csv, summary_rows, timestamp)
    log = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "adapter_id": ADAPTER_ID,
        "adapter_snapshot": str(snapshot),
        "variants": variant_reports,
        "summary_csv": str(summary_csv),
        "summary_csv_backup": str(summary_backup) if summary_backup else None,
        "backed_up_variant_dirs": backed_up_dirs,
    }
    log_path = Path(args.logs_dir) / f"uniform_scaling_adapter_generation_{timestamp}.json"
    write_json(log_path, log)

    passed = all(not report["warnings"] and not report["errors"] for report in variant_reports)
    print("Uniform scaling adapter generation summary")
    print(f"- Adapter: {ADAPTER_ID}")
    print(f"- Adapter snapshot: {snapshot}")
    print(f"- Variants generated: {len(VARIANTS)}")
    for report in variant_reports:
        validation = report["validation"]
        print(
            f"  - {report['variant_name']}: gamma={report['gamma']} "
            f"tensors={validation['tensor_count']} pairs={validation['complete_ab_pairs']} "
            f"ranks={validation['ranks']} finite={validation['all_finite']} "
            f"warnings={len(report['warnings'])}"
        )
    print(f"- Validation passed: {passed}")
    print(f"- JSON log written: {log_path}")
    print(f"- Summary CSV written: {summary_csv}")
    if summary_backup:
        print(f"- Previous summary CSV backed up to: {summary_backup}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

