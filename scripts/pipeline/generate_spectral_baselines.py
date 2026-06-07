"""Generate spectral-only sanitised LoRA adapter variants.

This script edits only cached adapter tensors. It does not load a base model,
run inference, use GPU, or evaluate ASR/clean utility. The original cached
adapter is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
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

from lora_sanitise.attenuation import (  # noqa: E402
    attenuate_singular_values,
    reconstruction_error,
    refactor_svd_to_lora_A_B,
    select_top_spectral_components,
    validate_lora_factor_shapes,
)
from lora_sanitise.lora_io import (  # noqa: E402
    adapter_config_path,
    classify_lora_tensor_key,
    hf_hub_cache_roots,
    repo_cache_dirname,
)
from lora_sanitise.svd_tools import (  # noqa: E402
    compact_svd_full_for_pair,
    compute_effective_rank,
    compute_spectral_entropy,
    compute_topk_energy,
    frobenius_norm_from_singular_values,
)


ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
VARIANTS = [
    {"name": "top1_gamma_0.0", "k": 1, "gamma": 0.0},
    {"name": "top1_gamma_0.25", "k": 1, "gamma": 0.25},
    {"name": "top1_gamma_0.50", "k": 1, "gamma": 0.50},
    {"name": "top3_gamma_0.0", "k": 3, "gamma": 0.0},
    {"name": "top3_gamma_0.25", "k": 3, "gamma": 0.25},
    {"name": "top3_gamma_0.50", "k": 3, "gamma": 0.50},
]


@dataclass(frozen=True)
class PairSpec:
    module_name: str
    layer_id: int | None
    target_module: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    a_dtype: str
    b_dtype: str
    rank: int


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


def locate_snapshot(adapter_path: str | None, cache_roots: list[str]) -> Path:
    if adapter_path:
        path = Path(adapter_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Adapter snapshot path does not exist: {path}")
        return path

    dirname = repo_cache_dirname(ADAPTER_ID, "model")
    candidates: list[Path] = []
    for root in hf_hub_cache_roots(cache_roots):
        snapshots = root / dirname / "snapshots"
        if snapshots.exists():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No cached snapshot found for {ADAPTER_ID}. Pass --adapter-path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def infer_layer_id(module_name: str | None) -> int | None:
    if not module_name:
        return None
    parts = module_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def infer_target_module(module_name: str | None) -> str:
    if not module_name:
        return "unknown"
    return module_name.split(".")[-1]


def rank_from_shapes(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> int | None:
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    return int(a_shape[0]) if a_shape[0] == b_shape[1] else None


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
            "dtype": str(tensor.dtype),
        }

    pairs: list[PairSpec] = []
    for module_name in sorted(grouped):
        module = grouped[module_name]
        if "A" not in module or "B" not in module:
            warnings.append(f"Incomplete LoRA pair for {module_name}")
            continue
        rank = rank_from_shapes(module["A"]["shape"], module["B"]["shape"])
        if rank is None:
            warnings.append(
                f"Rank mismatch for {module_name}: A={module['A']['shape']} B={module['B']['shape']}"
            )
            continue
        pairs.append(
            PairSpec(
                module_name=module_name,
                layer_id=infer_layer_id(module_name),
                target_module=infer_target_module(module_name),
                a_key=module["A"]["key"],
                b_key=module["B"]["key"],
                a_shape=module["A"]["shape"],
                b_shape=module["B"]["shape"],
                a_dtype=module["A"]["dtype"],
                b_dtype=module["B"]["dtype"],
                rank=rank,
            )
        )
    return pairs, warnings


def tensor_is_finite(tensor: Any) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def spectral_metrics(S: Any) -> dict[str, Any]:
    values = [float(value) for value in S.tolist()]
    return {
        "top1": compute_topk_energy(S, 1),
        "top3": compute_topk_energy(S, 3),
        "spectral_entropy": compute_spectral_entropy(S),
        "effective_rank": compute_effective_rank(S),
        "frobenius_norm": frobenius_norm_from_singular_values(S),
        "max_singular_value": max(values) if values else 0.0,
        "singular_values": values,
    }


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return float(statistics.mean(values)) if values else 0.0


def dense_check_if_small(A_new: Any, B_new: Any, U: Any, S_new: Any, Vh: Any, max_elements: int) -> dict[str, Any]:
    out_dim = int(B_new.shape[0])
    in_dim = int(A_new.shape[1])
    elements = out_dim * in_dim
    result: dict[str, Any] = {
        "attempted": False,
        "skipped_reason": "dense check disabled or too large",
        "dense_elements": elements,
        "absolute_error": None,
        "relative_error": None,
        "target_norm": None,
    }
    if max_elements <= 0 or elements > max_elements:
        return result
    target = (U * S_new.reshape(1, -1)) @ Vh
    error = reconstruction_error(A_new, B_new, target)
    result.update(
        {
            "attempted": True,
            "skipped_reason": None,
            "absolute_error": error["absolute_error"],
            "relative_error": error["relative_error"],
            "target_norm": error["target_norm"],
            "dense_elements": elements,
            "target_dtype": str(target.dtype),
        }
    )
    del target
    return result


def process_pair(
    pair: PairSpec,
    state: dict[str, Any],
    k: int,
    gamma: float,
    max_dense_check_elements: int,
) -> tuple[Any, Any, dict[str, Any]]:
    A = state[pair.a_key]
    B = state[pair.b_key]
    validate_lora_factor_shapes(A, B)
    U, S, Vh = compact_svd_full_for_pair(A=A, B=B)
    selected = select_top_spectral_components(S, k)
    S_new = attenuate_singular_values(S, selected, gamma)
    A_new, B_new = refactor_svd_to_lora_A_B(U, S_new, Vh)

    shape_ok = tuple(A_new.shape) == pair.a_shape and tuple(B_new.shape) == pair.b_shape
    finite_ok = tensor_is_finite(A_new) and tensor_is_finite(B_new)
    if not shape_ok:
        raise ValueError(
            f"Refactored shape mismatch for {pair.module_name}: "
            f"A {tuple(A_new.shape)} vs {pair.a_shape}, B {tuple(B_new.shape)} vs {pair.b_shape}"
        )
    if not finite_ok:
        raise ValueError(f"Refactored tensors contain NaN/Inf for {pair.module_name}")

    dense_check = dense_check_if_small(A_new, B_new, U, S_new, Vh, max_dense_check_elements)
    before = spectral_metrics(S)
    after = spectral_metrics(S_new)
    report = {
        "module_name": pair.module_name,
        "layer_id": pair.layer_id,
        "target_module": pair.target_module,
        "A_key": pair.a_key,
        "B_key": pair.b_key,
        "A_shape": [int(dim) for dim in pair.a_shape],
        "B_shape": [int(dim) for dim in pair.b_shape],
        "A_dtype_before": pair.a_dtype,
        "B_dtype_before": pair.b_dtype,
        "A_dtype_after": str(A_new.dtype),
        "B_dtype_after": str(B_new.dtype),
        "rank": pair.rank,
        "selected_indices": selected,
        "gamma": gamma,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "dense_reconstruction_check": dense_check,
        "before": before,
        "after": after,
    }

    A_out = A_new.to(dtype=A.dtype, device="cpu").contiguous()
    B_out = B_new.to(dtype=B.dtype, device="cpu").contiguous()
    if not tensor_is_finite(A_out) or not tensor_is_finite(B_out):
        raise ValueError(f"Output dtype conversion produced NaN/Inf for {pair.module_name}")
    return A_out, B_out, report


def summarize_variant(
    variant: dict[str, Any],
    pair_count: int,
    module_reports: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    before = [record["before"] for record in module_reports]
    after = [record["after"] for record in module_reports]
    dense_checks = [record["dense_reconstruction_check"] for record in module_reports]
    attempted_dense = [check for check in dense_checks if check["attempted"]]
    rank_values = sorted({int(record["rank"]) for record in module_reports})
    return {
        "variant_name": variant["name"],
        "selected_component_count": int(variant["k"]),
        "gamma": float(variant["gamma"]),
        "output_dir": str(output_dir),
        "adapter_config_path": str(output_dir / "adapter_config.json"),
        "adapter_model_path": str(output_dir / "adapter_model.safetensors"),
        "sanitisation_report_path": str(output_dir / "sanitisation_report.json"),
        "modules_edited": len(module_reports),
        "pair_count": pair_count,
        "rank_values": rank_values,
        "mean_top1_before": mean_metric(before, "top1"),
        "mean_top1_after": mean_metric(after, "top1"),
        "mean_top3_before": mean_metric(before, "top3"),
        "mean_top3_after": mean_metric(after, "top3"),
        "mean_entropy_before": mean_metric(before, "spectral_entropy"),
        "mean_entropy_after": mean_metric(after, "spectral_entropy"),
        "mean_effective_rank_before": mean_metric(before, "effective_rank"),
        "mean_effective_rank_after": mean_metric(after, "effective_rank"),
        "dense_reconstruction_checks_attempted": len(attempted_dense),
        "dense_reconstruction_checks_skipped": len(dense_checks) - len(attempted_dense),
        "max_dense_relative_error": max(
            [float(check["relative_error"]) for check in attempted_dense],
            default=None,
        ),
        "warnings": warnings,
        "errors": errors,
    }


def prepare_variant_dir(base_dir: Path, variant_name: str, timestamp: str) -> tuple[Path, Path | None]:
    output_dir = base_dir / variant_name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(output_dir, timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, backup


def generate_variant(
    *,
    variant: dict[str, Any],
    original_state: dict[str, Any],
    pairs: list[PairSpec],
    adapter_config_source: Path,
    output_base_dir: Path,
    timestamp: str,
    max_dense_check_elements: int,
) -> tuple[dict[str, Any], Path | None]:
    from safetensors.torch import save_file

    output_dir, backup = prepare_variant_dir(output_base_dir, variant["name"], timestamp)
    edited_state = {key: tensor.detach().cpu().clone() for key, tensor in original_state.items()}
    module_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for pair in pairs:
        try:
            A_new, B_new, module_report = process_pair(
                pair=pair,
                state=original_state,
                k=int(variant["k"]),
                gamma=float(variant["gamma"]),
                max_dense_check_elements=max_dense_check_elements,
            )
            edited_state[pair.a_key] = A_new
            edited_state[pair.b_key] = B_new
            module_reports.append(module_report)
        except Exception as exc:
            message = f"{pair.module_name}: {type(exc).__name__}: {exc}"
            errors.append(message)

    shutil.copy2(adapter_config_source, output_dir / "adapter_config.json")
    if errors:
        warnings.append("One or more modules failed; adapter_model.safetensors was not written.")
    else:
        save_file(edited_state, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    summary = summarize_variant(
        variant=variant,
        pair_count=len(pairs),
        module_reports=module_reports,
        warnings=warnings,
        errors=errors,
        output_dir=output_dir,
    )
    report = {
        "timestamp_utc": timestamp,
        "adapter_id": ADAPTER_ID,
        "variant": variant,
        "summary": summary,
        "module_reports": module_reports,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(output_dir / "sanitisation_report.json", report)
    return summary, backup


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing_file(path, timestamp)
    fieldnames = [
        "variant_name",
        "selected_component_count",
        "gamma",
        "output_dir",
        "adapter_config_path",
        "adapter_model_path",
        "sanitisation_report_path",
        "modules_edited",
        "pair_count",
        "rank_values",
        "mean_top1_before",
        "mean_top1_after",
        "mean_top3_before",
        "mean_top3_after",
        "mean_entropy_before",
        "mean_entropy_after",
        "mean_effective_rank_before",
        "mean_effective_rank_after",
        "dense_reconstruction_checks_attempted",
        "dense_reconstruction_checks_skipped",
        "max_dense_relative_error",
        "warnings",
        "errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["rank_values"] = json.dumps(out["rank_values"])
            out["warnings"] = json.dumps(out["warnings"])
            out["errors"] = json.dumps(out["errors"])
            writer.writerow({field: out.get(field) for field in fieldnames})
    return path, backup


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path | None]]:
    from safetensors.torch import load_file

    timestamp = utc_timestamp()
    adapter_snapshot = locate_snapshot(args.adapter_path, args.cache_root)
    config_path = adapter_config_path(adapter_snapshot)
    model_path = adapter_snapshot / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {model_path}")

    config = load_json(config_path)
    state = load_file(str(model_path), device="cpu")
    pairs, pair_warnings = group_pairs(state)
    if not pairs:
        raise RuntimeError("No complete LoRA A/B pairs found.")

    output_base_dir = Path(args.output_dir)
    summaries: list[dict[str, Any]] = []
    backups: list[Path | None] = []
    for variant in VARIANTS:
        summary, backup = generate_variant(
            variant=variant,
            original_state=state,
            pairs=pairs,
            adapter_config_source=config_path,
            output_base_dir=output_base_dir,
            timestamp=timestamp,
            max_dense_check_elements=args.max_dense_check_elements,
        )
        summaries.append(summary)
        backups.append(backup)

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
        "adapter": {
            "adapter_id": ADAPTER_ID,
            "snapshot_path": str(adapter_snapshot),
            "adapter_config_path": str(config_path),
            "adapter_model_path": str(model_path),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "rank": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config.get("target_modules"),
        },
        "pair_count": len(pairs),
        "pair_warnings": pair_warnings,
        "variants": summaries,
        "limitations": [
            "Spectral-only attenuation; clean-prompt sensitivity is not implemented yet.",
            "No ASR or clean utility evaluation was run.",
            "No base model was loaded.",
            "Dense reconstruction checks are skipped by default for memory safety.",
        ],
    }
    return report, summaries, backups


def print_summary(
    report: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    csv_backup: Path | None,
    variant_backups: list[Path | None],
) -> None:
    print("Spectral-only sanitised adapter generation summary")
    print(f"- Adapter: {report['adapter']['adapter_id']}")
    print(f"- Adapter snapshot: {report['adapter']['snapshot_path']}")
    print(f"- A/B pairs processed per variant: {report['pair_count']}")
    print(f"- Variants generated: {len(report['variants'])}")
    for variant in report["variants"]:
        print(
            "  - "
            f"{variant['variant_name']}: "
            f"k={variant['selected_component_count']} "
            f"gamma={variant['gamma']} "
            f"modules={variant['modules_edited']} "
            f"top1 {variant['mean_top1_before']:.6f}->{variant['mean_top1_after']:.6f} "
            f"entropy {variant['mean_entropy_before']:.6f}->{variant['mean_entropy_after']:.6f}"
        )
        if variant["errors"]:
            print(f"    errors: {len(variant['errors'])}")
    if report["pair_warnings"]:
        print("- Pair warnings:")
        for warning in report["pair_warnings"]:
            print(f"  - {warning}")
    else:
        print("- Pair warnings: none")
    print(f"- JSON log written: {json_path}")
    print(f"- Summary CSV written: {csv_path}")
    if csv_backup:
        print(f"- Previous summary CSV backed up to: {csv_backup}")
    backups = [backup for backup in variant_backups if backup is not None]
    if backups:
        print("- Previous variant folders backed up:")
        for backup in backups:
            print(f"  - {backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default=None, help="Cached BackdoorLLM adapter snapshot.")
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Extra Hugging Face hub cache root to search. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/sanitised_adapters")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--summary-csv",
        default="outputs/sanitised_adapter_generation_summary.csv",
    )
    parser.add_argument(
        "--max-dense-check-elements",
        type=int,
        default=0,
        help="Optional dense reconstruction check threshold. Default 0 disables dense checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, summaries, variant_backups = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"sanitised_adapter_generation_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_summary_csv(Path(args.summary_csv), summaries, timestamp)
    print_summary(report, json_path, csv_path, csv_backup, variant_backups)
    has_errors = any(summary["errors"] for summary in summaries)
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

