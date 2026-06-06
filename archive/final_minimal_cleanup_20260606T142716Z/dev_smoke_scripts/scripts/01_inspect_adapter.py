"""Inspect cached PEFT/LoRA adapter files without loading a base model.

This script reads only:
- adapter_config.json
- adapter_model.safetensors

It does not load Llama-2, construct a Transformers model, run inference, or use
the GPU. Tensor inspection is CPU/filesystem-level only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_PATH = (
    "/home/huggingface/hub/"
    "models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/"
    "snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b"
)

CONFIG_FIELDS = [
    "base_model_name_or_path",
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "inference_mode",
]

DORA_RSLORA_FIELDS = [
    "use_dora",
    "use_rslora",
    "rank_pattern",
    "alpha_pattern",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_lora_key(key: str) -> tuple[str | None, str | None]:
    """Return (kind, module_name) for likely LoRA A/B tensors."""
    patterns = [
        (".lora_A.", "A"),
        (".lora_B.", "B"),
        (".lora_embedding_A.", "A"),
        (".lora_embedding_B.", "B"),
    ]
    for marker, kind in patterns:
        if marker in key:
            return kind, key.split(marker, 1)[0]

    suffix_patterns = [
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_embedding_A.weight", "A"),
        (".lora_embedding_B.weight", "B"),
    ]
    for suffix, kind in suffix_patterns:
        if key.endswith(suffix):
            return kind, key[: -len(suffix)]

    return None, None


def target_module_hint(module_name: str) -> str:
    """Best-effort final module segment, e.g. q_proj from layer path."""
    if not module_name:
        return ""
    return module_name.split(".")[-1]


def tensor_shape_dtype_from_slice(handle: Any, key: str) -> tuple[list[int], str]:
    try:
        tensor_slice = handle.get_slice(key)
        shape = list(tensor_slice.get_shape())
        dtype = str(tensor_slice.get_dtype())
        return shape, dtype
    except Exception:
        tensor = handle.get_tensor(key)
        return list(tensor.shape), str(tensor.dtype)


def inspect_safetensors(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment diagnostic path
        raise RuntimeError(f"Could not import safetensors: {type(exc).__name__}: {exc}") from exc

    tensor_rows: list[dict[str, Any]] = []
    pairs: dict[str, dict[str, Any]] = defaultdict(lambda: {"A": None, "B": None})
    warnings: list[str] = []

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            shape, dtype = tensor_shape_dtype_from_slice(handle, key)
            kind, module_name = classify_lora_key(key)
            row = {
                "key": key,
                "shape": shape,
                "shape_str": "x".join(str(dim) for dim in shape),
                "dtype": dtype,
                "numel": int(shape[0] * shape[1]) if len(shape) == 2 else None,
                "lora_kind": kind,
                "module_name": module_name,
                "target_module_hint": target_module_hint(module_name or ""),
                "inferred_rank": infer_rank_from_shape(kind, shape),
            }
            tensor_rows.append(row)
            if kind and module_name:
                pairs[module_name][kind] = row

    for module_name, pair in pairs.items():
        if pair["A"] is None or pair["B"] is None:
            warnings.append(f"Incomplete LoRA pair for module {module_name}")

    return tensor_rows, dict(pairs), warnings


def infer_rank_from_shape(kind: str | None, shape: list[int]) -> int | None:
    if len(shape) != 2 or kind not in {"A", "B"}:
        return None
    if kind == "A":
        return int(shape[0])
    return int(shape[1])


def summarize_pairs(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pair_summaries: list[dict[str, Any]] = []
    complete_count = 0
    rank_values: list[int] = []
    mismatch_warnings: list[str] = []

    for module_name in sorted(pairs):
        pair = pairs[module_name]
        a_row = pair.get("A")
        b_row = pair.get("B")
        complete = a_row is not None and b_row is not None
        if complete:
            complete_count += 1

        a_rank = a_row.get("inferred_rank") if a_row else None
        b_rank = b_row.get("inferred_rank") if b_row else None
        rank_match = a_rank is not None and b_rank is not None and a_rank == b_rank
        if rank_match:
            rank_values.append(int(a_rank))
        elif complete:
            mismatch_warnings.append(
                f"Rank mismatch for {module_name}: A rank {a_rank}, B rank {b_rank}"
            )

        pair_summaries.append(
            {
                "module_name": module_name,
                "target_module_hint": target_module_hint(module_name),
                "has_A": a_row is not None,
                "has_B": b_row is not None,
                "A_key": a_row.get("key") if a_row else None,
                "B_key": b_row.get("key") if b_row else None,
                "A_shape": a_row.get("shape") if a_row else None,
                "B_shape": b_row.get("shape") if b_row else None,
                "A_dtype": a_row.get("dtype") if a_row else None,
                "B_dtype": b_row.get("dtype") if b_row else None,
                "A_rank": a_rank,
                "B_rank": b_rank,
                "rank_match": rank_match,
            }
        )

    unique_ranks = sorted(set(rank_values))
    return {
        "pairs": pair_summaries,
        "module_count": len(pairs),
        "complete_pair_count": complete_count,
        "incomplete_pair_count": len(pairs) - complete_count,
        "unique_inferred_ranks": unique_ranks,
        "rank_consistent": len(unique_ranks) <= 1 and not mismatch_warnings,
        "mismatch_warnings": mismatch_warnings,
    }


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in CONFIG_FIELDS}


def dora_rslora_warnings(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in DORA_RSLORA_FIELDS:
        if field not in config:
            continue
        value = config.get(field)
        if value in (False, None, {}, []):
            warnings.append(f"{field} present but inactive/empty: {value!r}")
        else:
            warnings.append(f"{field} present and active/non-empty: {value!r}")

    for key in sorted(config):
        if re.search(r"(dora|rslora|rs_lora)", key, re.IGNORECASE) and key not in DORA_RSLORA_FIELDS:
            warnings.append(f"DoRA/rsLoRA-like config key present: {key}={config.get(key)!r}")
    return warnings


def write_json_log(report: dict[str, Any], logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"adapter_inspection_{report['timestamp_utc']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_tensor_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "shape_str",
        "dtype",
        "numel",
        "lora_kind",
        "module_name",
        "target_module_hint",
        "inferred_rank",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return output_path


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = Path(args.adapter_path).expanduser()
    config_path = adapter_path / "adapter_config.json"
    model_path = adapter_path / "adapter_model.safetensors"

    config_exists = config_path.exists()
    model_exists = model_path.exists()
    config: dict[str, Any] = read_json(config_path) if config_exists else {}

    tensor_rows: list[dict[str, Any]] = []
    pair_summary: dict[str, Any] = {
        "pairs": [],
        "module_count": 0,
        "complete_pair_count": 0,
        "incomplete_pair_count": 0,
        "unique_inferred_ranks": [],
        "rank_consistent": False,
        "mismatch_warnings": [],
    }
    tensor_warnings: list[str] = []

    if model_exists:
        tensor_rows, pairs, tensor_warnings = inspect_safetensors(model_path)
        pair_summary = summarize_pairs(pairs)

    warnings = []
    if not config_exists:
        warnings.append(f"Missing adapter_config.json at {config_path}")
    if not model_exists:
        warnings.append(f"Missing adapter_model.safetensors at {model_path}")
    warnings.extend(tensor_warnings)
    warnings.extend(pair_summary.get("mismatch_warnings", []))
    warnings.extend(dora_rslora_warnings(config))

    extraction_possible = (
        config_exists
        and model_exists
        and pair_summary["complete_pair_count"] > 0
        and pair_summary["incomplete_pair_count"] == 0
        and not pair_summary["mismatch_warnings"]
    )

    return {
        "timestamp_utc": utc_timestamp(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
        },
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "adapter": {
            "snapshot_path": str(adapter_path),
            "adapter_config_path": str(config_path),
            "adapter_config_exists": config_exists,
            "adapter_model_path": str(model_path),
            "adapter_model_exists": model_exists,
            "adapter_model_size_mb": round(model_path.stat().st_size / (1024**2), 3)
            if model_exists
            else None,
        },
        "config_summary": config_summary(config),
        "config_all_keys": sorted(config.keys()),
        "dora_rslora_warnings": dora_rslora_warnings(config),
        "tensor_count": len(tensor_rows),
        "tensor_summary": tensor_rows,
        "lora_pair_summary": pair_summary,
        "extraction_possible": extraction_possible,
        "warnings": warnings,
    }


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    config = report["config_summary"]
    pairs = report["lora_pair_summary"]
    warnings = report["warnings"]
    ranks = pairs["unique_inferred_ranks"]

    print("Adapter inspection summary")
    print(f"- Adapter snapshot: {report['adapter']['snapshot_path']}")
    print(f"- adapter_config.json exists: {report['adapter']['adapter_config_exists']}")
    print(f"- adapter_model.safetensors exists: {report['adapter']['adapter_model_exists']}")
    print(f"- Adapter type: {config.get('peft_type')}")
    print(f"- Task type: {config.get('task_type')}")
    print(f"- Config rank r: {config.get('r')}")
    print(f"- Inferred ranks: {ranks or 'none'}")
    print(f"- Target modules: {config.get('target_modules')}")
    print(f"- Tensor count: {report['tensor_count']}")
    print(f"- LoRA module groups: {pairs['module_count']}")
    print(f"- Complete A/B pairs: {pairs['complete_pair_count']}")
    print(f"- Incomplete A/B pairs: {pairs['incomplete_pair_count']}")
    print(f"- A/B extraction looks possible: {report['extraction_possible']}")
    if warnings:
        print("- Warnings/blockers:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("- Warnings/blockers: none")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV tensor summary written: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help="Path to cached adapter snapshot directory.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for adapter_inspection_*.json logs.",
    )
    parser.add_argument(
        "--csv-path",
        default="outputs/adapter_tensor_summary.csv",
        help="CSV output path for tensor summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    json_path = write_json_log(report, Path(args.logs_dir))
    csv_path = write_tensor_csv(report["tensor_summary"], Path(args.csv_path))
    print_summary(report, json_path, csv_path)
    return 0 if report["extraction_possible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
