"""Generate first sensitivity-aware LoRA adapter variants.

This script edits only adapter tensors. It consumes bounded clean-sensitivity
component scores, attenuates the most suspicious singular components, refactors
the edited updates back into PEFT LoRA A/B tensors, and writes new adapter
folders. It does not load a base model or run inference.
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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.attenuation import (  # noqa: E402
    attenuate_singular_values,
    refactor_svd_to_lora_A_B,
    validate_lora_factor_shapes,
)
from lora_sanitise.lora_io import (  # noqa: E402
    ADAPTER_ID,
    adapter_config_path,
    classify_lora_tensor_key,
    locate_adapter_snapshot,
)
from lora_sanitise.svd_tools import compact_svd_full_for_pair  # noqa: E402


DEFAULT_SCORE_CSV = ROOT / "outputs" / "clean_sensitivity_component_scores.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "sensitivity_aware_adapter_generation_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
VARIANTS = [
    {"name": "sensaware_top16_gamma_0.50", "top_n": 16, "gamma": 0.50},
    {"name": "sensaware_top32_gamma_0.50", "top_n": 32, "gamma": 0.50},
    {"name": "sensaware_top32_gamma_0.25", "top_n": 32, "gamma": 0.25},
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


def read_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sensitivity component scores not found: {path}. "
            "Run scripts/21_clean_sensitivity_probe.py first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "module_name": row["module_name"],
                    "layer_id": int(row["layer_id"]) if row.get("layer_id") not in (None, "") else None,
                    "target_module": row["target_module"],
                    "component_index": int(row["component_index"]),
                    "singular_value": float(row["singular_value"]),
                    "spectral_energy_share": float(row["spectral_energy_share"]),
                    "clean_sensitivity_raw": float(row["clean_sensitivity_raw"]),
                    "clean_sensitivity_norm_global": float(row["clean_sensitivity_norm_global"]),
                    "clean_sensitivity_norm_by_module": float(row["clean_sensitivity_norm_by_module"]),
                    "suspiciousness_score_prelim": float(row["suspiciousness_score_prelim"]),
                    "prompt_count_used": int(float(row["prompt_count_used"])),
                    "token_count_used": int(float(row.get("token_count_used", 0) or 0)),
                    "A_key": row.get("A_key", ""),
                    "B_key": row.get("B_key", ""),
                }
            )
    if not rows:
        raise ValueError(f"No sensitivity score rows found in {path}")
    zero_prompt_rows = [row for row in rows if int(row["prompt_count_used"]) <= 0]
    zero_token_rows = [row for row in rows if int(row["token_count_used"]) <= 0]
    if zero_prompt_rows or zero_token_rows:
        raise ValueError(
            "Sensitivity score CSV does not contain completed clean-forward sensitivity data. "
            f"Rows with zero prompt_count_used: {len(zero_prompt_rows)}; "
            f"rows with zero token_count_used: {len(zero_token_rows)}. "
            "Re-run scripts/21_clean_sensitivity_probe.py successfully before generating "
            "sensitivity-aware adapters."
        )
    rows.sort(key=lambda item: item["suspiciousness_score_prelim"], reverse=True)
    return rows


def select_components(score_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in score_rows:
        key = (row["module_name"], int(row["component_index"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= top_n:
            break
    return selected


def tensor_is_finite(tensor: Any) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def validate_state(state: dict[str, Any], original_state: dict[str, Any]) -> dict[str, Any]:
    import torch

    pairs, pair_warnings = group_pairs(state)
    a_count = 0
    b_count = 0
    ranks = set()
    target_modules = set()
    all_finite = True
    nonfinite_keys: list[str] = []
    shape_mismatches: list[str] = []

    for key, tensor in state.items():
        if key not in original_state:
            shape_mismatches.append(f"Unexpected tensor key: {key}")
        elif tuple(tensor.shape) != tuple(original_state[key].shape):
            shape_mismatches.append(
                f"{key}: {tuple(tensor.shape)} vs original {tuple(original_state[key].shape)}"
            )
        kind, module_name = classify_lora_tensor_key(str(key))
        if kind == "A":
            a_count += 1
        elif kind == "B":
            b_count += 1
        if module_name:
            target_modules.add(infer_target_module(module_name))
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            all_finite = False
            nonfinite_keys.append(str(key))
    for key in original_state:
        if key not in state:
            shape_mismatches.append(f"Missing tensor key: {key}")
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
        "all_finite": all_finite,
        "nonfinite_keys": nonfinite_keys,
        "shape_mismatches": shape_mismatches,
        "pair_warnings": pair_warnings,
    }


def edit_module(
    pair: PairSpec,
    state: dict[str, Any],
    selected_indices: list[int],
    gamma: float,
) -> tuple[Any, Any, dict[str, Any]]:
    A = state[pair.a_key]
    B = state[pair.b_key]
    validate_lora_factor_shapes(A, B)
    U, S, Vh = compact_svd_full_for_pair(A=A, B=B)
    valid_indices = sorted({index for index in selected_indices if 0 <= index < int(S.numel())})
    if len(valid_indices) != len(set(selected_indices)):
        raise ValueError(f"Invalid selected index for {pair.module_name}: {selected_indices}")
    S_new = attenuate_singular_values(S, valid_indices, gamma)
    A_new, B_new = refactor_svd_to_lora_A_B(U, S_new, Vh)
    if tuple(A_new.shape) != pair.a_shape or tuple(B_new.shape) != pair.b_shape:
        raise ValueError(
            f"Shape mismatch after refactor for {pair.module_name}: "
            f"A {tuple(A_new.shape)} vs {pair.a_shape}, B {tuple(B_new.shape)} vs {pair.b_shape}"
        )
    A_out = A_new.to(dtype=A.dtype, device="cpu").contiguous()
    B_out = B_new.to(dtype=B.dtype, device="cpu").contiguous()
    if not tensor_is_finite(A_out) or not tensor_is_finite(B_out):
        raise ValueError(f"Non-finite output tensor for {pair.module_name}")
    return (
        A_out,
        B_out,
        {
            "module_name": pair.module_name,
            "target_module": pair.target_module,
            "selected_indices": valid_indices,
            "gamma": gamma,
            "rank": pair.rank,
            "singular_values_before": [float(value) for value in S.tolist()],
            "singular_values_after": [float(value) for value in S_new.tolist()],
        },
    )


def prepare_variant_dir(base_dir: Path, variant_name: str, timestamp: str) -> tuple[Path, Path | None]:
    output_dir = base_dir / variant_name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(output_dir, timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, backup


def generate_variant(
    variant: dict[str, Any],
    score_rows: list[dict[str, Any]],
    original_state: dict[str, Any],
    pairs_by_module: dict[str, PairSpec],
    adapter_config_source: Path,
    output_root: Path,
    timestamp: str,
) -> tuple[dict[str, Any], Path | None]:
    from safetensors.torch import save_file

    selected = select_components(score_rows, int(variant["top_n"]))
    selected_by_module: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        selected_by_module[row["module_name"]].append(int(row["component_index"]))

    output_dir, backup = prepare_variant_dir(output_root, str(variant["name"]), timestamp)
    edited_state = {key: tensor.detach().cpu().clone() for key, tensor in original_state.items()}
    module_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for module_name, indices in sorted(selected_by_module.items()):
        pair = pairs_by_module.get(module_name)
        if pair is None:
            warnings.append(f"No LoRA pair found for selected module {module_name}")
            continue
        try:
            A_new, B_new, module_report = edit_module(pair, original_state, indices, float(variant["gamma"]))
            edited_state[pair.a_key] = A_new
            edited_state[pair.b_key] = B_new
            module_reports.append(module_report)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    shutil.copy2(adapter_config_source, output_dir / "adapter_config.json")
    validation = validate_state(edited_state, original_state)
    if validation["tensor_count"] != 448:
        warnings.append(f"Expected 448 tensors, got {validation['tensor_count']}")
    if validation["complete_ab_pairs"] != 224:
        warnings.append(f"Expected 224 complete A/B pairs, got {validation['complete_ab_pairs']}")
    if validation["ranks"] != [8]:
        warnings.append(f"Expected rank [8], got {validation['ranks']}")
    if not validation["required_target_modules_present"]:
        warnings.append(f"Missing required target modules: {validation['missing_target_modules']}")
    if not validation["all_finite"]:
        warnings.append(f"Non-finite tensors: {validation['nonfinite_keys'][:5]}")
    if validation["shape_mismatches"]:
        warnings.append(f"Shape mismatches: {validation['shape_mismatches'][:5]}")
    warnings.extend(validation["pair_warnings"])

    if not errors:
        save_file(edited_state, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    else:
        warnings.append("Errors occurred; adapter_model.safetensors was not written.")

    summary = {
        "variant_name": variant["name"],
        "selected_component_count": len(selected),
        "selected_module_count": len(selected_by_module),
        "gamma": float(variant["gamma"]),
        "output_dir": str(output_dir),
        "adapter_config_path": str(output_dir / "adapter_config.json"),
        "adapter_model_path": str(output_dir / "adapter_model.safetensors"),
        "sanitisation_report_path": str(output_dir / "sanitisation_report.json"),
        "modules_edited": len(module_reports),
        "tensor_count": validation["tensor_count"],
        "complete_ab_pairs": validation["complete_ab_pairs"],
        "rank_values": validation["ranks"],
        "all_finite": validation["all_finite"],
        "warnings": warnings,
        "errors": errors,
    }
    report = {
        "timestamp_utc": timestamp,
        "adapter_id": ADAPTER_ID,
        "variant": variant,
        "selected_components": selected,
        "module_reports": module_reports,
        "validation": validation,
        "summary": summary,
        "warnings": warnings,
        "errors": errors,
        "caveat": "First bounded sensitivity-aware adapter variant; not final proof.",
    }
    write_json(output_dir / "sanitisation_report.json", report)
    return summary, backup


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing_file(path, timestamp)
    fieldnames = [
        "variant_name",
        "selected_component_count",
        "selected_module_count",
        "gamma",
        "output_dir",
        "adapter_config_path",
        "adapter_model_path",
        "sanitisation_report_path",
        "modules_edited",
        "tensor_count",
        "complete_ab_pairs",
        "rank_values",
        "all_finite",
        "warnings",
        "errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["rank_values"] = json.dumps(out["rank_values"])
            out["warnings"] = " | ".join(out["warnings"])
            out["errors"] = " | ".join(out["errors"])
            writer.writerow(out)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sensitivity-aware LoRA adapter variants.")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--score-csv", default=str(DEFAULT_SCORE_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()

    from safetensors.torch import load_file

    score_rows = read_scores(Path(args.score_csv))
    adapter_snapshot = locate_adapter_snapshot(args.adapter_path, cache_roots=args.cache_root)
    config_path = adapter_config_path(adapter_snapshot)
    model_path = adapter_snapshot / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {model_path}")
    original_state = load_file(str(model_path), device="cpu")
    pairs, pair_warnings = group_pairs(original_state)
    pairs_by_module = {pair.module_name: pair for pair in pairs if pair.rank is not None}

    summaries: list[dict[str, Any]] = []
    backups: list[str] = []
    for variant in VARIANTS:
        summary, backup = generate_variant(
            variant=variant,
            score_rows=score_rows,
            original_state=original_state,
            pairs_by_module=pairs_by_module,
            adapter_config_source=config_path,
            output_root=Path(args.output_root),
            timestamp=timestamp,
        )
        summaries.append(summary)
        if backup:
            backups.append(str(backup))

    summary_backup = write_summary_csv(Path(args.summary_csv), summaries, timestamp)
    log = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "adapter_id": ADAPTER_ID,
        "adapter_snapshot": str(adapter_snapshot),
        "score_csv": str(args.score_csv),
        "pair_warnings": pair_warnings,
        "variants": summaries,
        "variant_backups": backups,
        "summary_csv": str(args.summary_csv),
        "summary_csv_backup": str(summary_backup) if summary_backup else None,
    }
    log_path = Path(args.logs_dir) / f"sensitivity_aware_adapter_generation_{timestamp}.json"
    write_json(log_path, log)

    has_errors = any(row["errors"] for row in summaries)
    has_warnings = any(row["warnings"] for row in summaries)
    print("Sensitivity-aware adapter generation summary")
    print(f"- Adapter: {ADAPTER_ID}")
    print(f"- Adapter snapshot: {adapter_snapshot}")
    print(f"- Score CSV: {args.score_csv}")
    print(f"- Variants generated: {len(summaries)}")
    for row in summaries:
        print(
            f"  - {row['variant_name']}: selected={row['selected_component_count']} "
            f"modules={row['modules_edited']} gamma={row['gamma']} "
            f"pairs={row['complete_ab_pairs']} ranks={row['rank_values']} "
            f"finite={row['all_finite']} warnings={len(row['warnings'])} errors={len(row['errors'])}"
        )
    print(f"- JSON log written: {log_path}")
    print(f"- Summary CSV written: {args.summary_csv}")
    if summary_backup:
        print(f"- Previous summary CSV backed up to: {summary_backup}")
    if backups:
        print("- Previous variant folders backed up:")
        for backup in backups:
            print(f"  - {backup}")
    return 2 if has_errors or has_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
