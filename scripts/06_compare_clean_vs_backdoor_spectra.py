"""Adapter-only CPU comparison of clean vs backdoored LoRA spectra.

This script compares the BackdoorLLM BadNets adapter against the provisional
FlagAlpha structural clean reference. It reads adapter tensors only; it does
not load a base model, run inference, use GPU, or execute repository code.

Security rule for the clean adapter .bin file:
torch.load(..., map_location="cpu", weights_only=True) is the only allowed
loading path. There is no unsafe fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
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

from lora_sanitise.lora_io import (  # noqa: E402
    adapter_config_path,
    classify_lora_tensor_key,
    hf_hub_cache_roots,
    repo_cache_dirname,
)
from lora_sanitise.svd_tools import (  # noqa: E402
    compact_svd_singular_values,
    compute_effective_rank,
    compute_spectral_entropy,
    compute_topk_energy,
    frobenius_norm_from_singular_values,
)


BACKDOOR_REPO_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
CLEAN_REPO_ID = "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"
NORM_CAVEAT = (
    "Raw Frobenius norms and max singular values are not directly comparable "
    "because the adapters use different lora_alpha values; normalized spectral "
    "metrics are primary."
)


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
    rank: int | None

    @property
    def match_key(self) -> tuple[int | None, str]:
        return self.layer_id, self.target_module


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def locate_snapshot(repo_id: str, explicit_path: str | None, cache_roots: list[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Adapter snapshot path does not exist: {path}")
        return path

    dirname = repo_cache_dirname(repo_id, "model")
    candidates: list[Path] = []
    for root in hf_hub_cache_roots(cache_roots):
        snapshots = root / dirname / "snapshots"
        if snapshots.exists():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No cached snapshot found for {repo_id}. Pass the snapshot path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def shape_list(shape: tuple[int, ...]) -> list[int]:
    return [int(dim) for dim in shape]


def tensor_to_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def sort_key(key: tuple[int | None, str]) -> tuple[int, str]:
    layer, target = key
    return (-1 if layer is None else int(layer), target)


def load_backdoor_pairs(snapshot: Path) -> tuple[dict[tuple[int | None, str], PairSpec], list[str]]:
    from safetensors import safe_open

    model_path = snapshot / "adapter_model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Backdoor adapter_model.safetensors not found: {model_path}")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    warnings: list[str] = []
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            kind, module_name = classify_lora_tensor_key(key)
            if kind is None or module_name is None:
                continue
            tensor_slice = handle.get_slice(key)
            grouped[module_name][kind] = {
                "key": key,
                "shape": tuple(int(dim) for dim in tensor_slice.get_shape()),
                "dtype": str(tensor_slice.get_dtype()),
            }

    pairs: dict[tuple[int | None, str], PairSpec] = {}
    for module_name in sorted(grouped):
        module = grouped[module_name]
        if "A" not in module or "B" not in module:
            warnings.append(f"Backdoor incomplete LoRA pair: {module_name}")
            continue
        pair = PairSpec(
            module_name=module_name,
            layer_id=infer_layer_id(module_name),
            target_module=infer_target_module(module_name),
            a_key=module["A"]["key"],
            b_key=module["B"]["key"],
            a_shape=module["A"]["shape"],
            b_shape=module["B"]["shape"],
            a_dtype=module["A"]["dtype"],
            b_dtype=module["B"]["dtype"],
            rank=rank_from_shapes(module["A"]["shape"], module["B"]["shape"]),
        )
        if pair.rank is None:
            warnings.append(
                f"Backdoor rank mismatch: {module_name} A={pair.a_shape} B={pair.b_shape}"
            )
            continue
        if pair.match_key in pairs:
            warnings.append(f"Backdoor duplicate match key: {pair.match_key}")
            continue
        pairs[pair.match_key] = pair
    return pairs, warnings


def load_clean_state_dict(bin_path: Path) -> dict[str, Any]:
    if not bin_path.exists():
        raise FileNotFoundError(f"Clean adapter_model.bin not found: {bin_path}")
    import torch

    state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"weights_only load returned {type(state).__name__}, expected dict")

    non_tensor_keys = [
        str(key)
        for key, value in state.items()
        if not hasattr(value, "shape") or not hasattr(value, "dtype")
    ]
    if non_tensor_keys:
        sample = non_tensor_keys[:5]
        raise TypeError(f"weights_only state dict contains non-tensor values: {sample}")
    return state


def load_clean_pairs(snapshot: Path) -> tuple[dict[tuple[int | None, str], PairSpec], dict[str, Any], list[str]]:
    bin_path = snapshot / "adapter_model.bin"
    state = load_clean_state_dict(bin_path)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    warnings: list[str] = []
    for key, tensor in sorted(state.items()):
        kind, module_name = classify_lora_tensor_key(str(key))
        if kind is None or module_name is None:
            continue
        grouped[module_name][kind] = {
            "key": str(key),
            "shape": tensor_to_shape(tensor),
            "dtype": str(tensor.dtype),
        }

    pairs: dict[tuple[int | None, str], PairSpec] = {}
    for module_name in sorted(grouped):
        module = grouped[module_name]
        if "A" not in module or "B" not in module:
            warnings.append(f"Clean incomplete LoRA pair: {module_name}")
            continue
        pair = PairSpec(
            module_name=module_name,
            layer_id=infer_layer_id(module_name),
            target_module=infer_target_module(module_name),
            a_key=module["A"]["key"],
            b_key=module["B"]["key"],
            a_shape=module["A"]["shape"],
            b_shape=module["B"]["shape"],
            a_dtype=module["A"]["dtype"],
            b_dtype=module["B"]["dtype"],
            rank=rank_from_shapes(module["A"]["shape"], module["B"]["shape"]),
        )
        if pair.rank is None:
            warnings.append(f"Clean rank mismatch: {module_name} A={pair.a_shape} B={pair.b_shape}")
            continue
        if pair.match_key in pairs:
            warnings.append(f"Clean duplicate match key: {pair.match_key}")
            continue
        pairs[pair.match_key] = pair
    return pairs, state, warnings


def load_backdoor_tensors(snapshot: Path, pair: PairSpec) -> tuple[Any, Any]:
    from safetensors import safe_open

    model_path = snapshot / "adapter_model.safetensors"
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(pair.a_key), handle.get_tensor(pair.b_key)


def load_clean_tensors(state: dict[str, Any], pair: PairSpec) -> tuple[Any, Any]:
    return state[pair.a_key], state[pair.b_key]


def normalized_singular_values(values: Any) -> list[float]:
    if values.numel() == 0:
        return []
    max_value = float(values.max().item())
    if max_value <= 0.0:
        return [0.0 for _ in values.tolist()]
    return [float(value) / max_value for value in values.tolist()]


def spectral_record(
    *,
    adapter_label: str,
    repo_id: str,
    pair: PairSpec,
    A: Any,
    B: Any,
    lora_alpha: float | None,
) -> dict[str, Any]:
    singular_values_tensor = compact_svd_singular_values(A=A, B=B)
    singular_values = [float(value) for value in singular_values_tensor.tolist()]
    rank = pair.rank or 0
    lora_scale = float(lora_alpha) / float(rank) if lora_alpha is not None and rank > 0 else None
    return {
        "adapter_label": adapter_label,
        "repo_id": repo_id,
        "module_name": pair.module_name,
        "layer_id": pair.layer_id,
        "target_module": pair.target_module,
        "A_key": pair.a_key,
        "B_key": pair.b_key,
        "A_shape": shape_list(pair.a_shape),
        "B_shape": shape_list(pair.b_shape),
        "A_dtype": pair.a_dtype,
        "B_dtype": pair.b_dtype,
        "rank": pair.rank,
        "lora_alpha": lora_alpha,
        "lora_scale_alpha_over_rank": lora_scale,
        "top1_energy_share": compute_topk_energy(singular_values_tensor, 1),
        "top3_energy_share": compute_topk_energy(singular_values_tensor, 3),
        "spectral_entropy": compute_spectral_entropy(singular_values_tensor),
        "effective_rank": compute_effective_rank(singular_values_tensor),
        "frobenius_norm": frobenius_norm_from_singular_values(singular_values_tensor),
        "max_singular_value": max(singular_values) if singular_values else 0.0,
        "singular_values": singular_values,
        "normalized_singular_values": normalized_singular_values(singular_values_tensor),
    }


def metric_delta(backdoor: dict[str, Any], clean: dict[str, Any], metric: str) -> float:
    return float(backdoor[metric]) - float(clean[metric])


def comparison_row(backdoor: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_id": backdoor["layer_id"],
        "target_module": backdoor["target_module"],
        "backdoor_module_name": backdoor["module_name"],
        "clean_module_name": clean["module_name"],
        "backdoor_rank": backdoor["rank"],
        "clean_rank": clean["rank"],
        "backdoor_lora_alpha": backdoor["lora_alpha"],
        "clean_lora_alpha": clean["lora_alpha"],
        "backdoor_lora_scale_alpha_over_rank": backdoor["lora_scale_alpha_over_rank"],
        "clean_lora_scale_alpha_over_rank": clean["lora_scale_alpha_over_rank"],
        "backdoor_top1": backdoor["top1_energy_share"],
        "clean_top1": clean["top1_energy_share"],
        "delta_top1": metric_delta(backdoor, clean, "top1_energy_share"),
        "backdoor_top3": backdoor["top3_energy_share"],
        "clean_top3": clean["top3_energy_share"],
        "delta_top3": metric_delta(backdoor, clean, "top3_energy_share"),
        "backdoor_entropy": backdoor["spectral_entropy"],
        "clean_entropy": clean["spectral_entropy"],
        "delta_entropy": metric_delta(backdoor, clean, "spectral_entropy"),
        "backdoor_effective_rank": backdoor["effective_rank"],
        "clean_effective_rank": clean["effective_rank"],
        "delta_effective_rank": metric_delta(backdoor, clean, "effective_rank"),
        "backdoor_fro_norm": backdoor["frobenius_norm"],
        "clean_fro_norm": clean["frobenius_norm"],
        "delta_fro_norm": metric_delta(backdoor, clean, "frobenius_norm"),
        "backdoor_max_singular_value": backdoor["max_singular_value"],
        "clean_max_singular_value": clean["max_singular_value"],
        "delta_max_singular_value": metric_delta(backdoor, clean, "max_singular_value"),
        "backdoor_normalized_singular_values": backdoor["normalized_singular_values"],
        "clean_normalized_singular_values": clean["normalized_singular_values"],
        "norm_comparison_note": NORM_CAVEAT,
    }


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def grouped_mean(rows: list[dict[str, Any]], key: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[metric]))
    return {name: mean(values) for name, values in sorted(grouped.items())}


def summarize(rows: list[dict[str, Any]], unmatched_backdoor: list[tuple[int | None, str]], unmatched_clean: list[tuple[int | None, str]]) -> dict[str, Any]:
    mean_backdoor_top1 = mean([float(row["backdoor_top1"]) for row in rows])
    mean_clean_top1 = mean([float(row["clean_top1"]) for row in rows])
    mean_backdoor_top3 = mean([float(row["backdoor_top3"]) for row in rows])
    mean_clean_top3 = mean([float(row["clean_top3"]) for row in rows])
    mean_backdoor_entropy = mean([float(row["backdoor_entropy"]) for row in rows])
    mean_clean_entropy = mean([float(row["clean_entropy"]) for row in rows])

    by_target: dict[str, dict[str, float]] = {}
    for target in sorted({str(row["target_module"]) for row in rows}):
        target_rows = [row for row in rows if row["target_module"] == target]
        by_target[target] = {
            "mean_backdoor_top1": mean([float(row["backdoor_top1"]) for row in target_rows]),
            "mean_clean_top1": mean([float(row["clean_top1"]) for row in target_rows]),
            "mean_delta_top1": mean([float(row["delta_top1"]) for row in target_rows]),
            "mean_backdoor_top3": mean([float(row["backdoor_top3"]) for row in target_rows]),
            "mean_clean_top3": mean([float(row["clean_top3"]) for row in target_rows]),
            "mean_delta_top3": mean([float(row["delta_top3"]) for row in target_rows]),
            "mean_backdoor_entropy": mean([float(row["backdoor_entropy"]) for row in target_rows]),
            "mean_clean_entropy": mean([float(row["clean_entropy"]) for row in target_rows]),
            "mean_delta_entropy": mean([float(row["delta_entropy"]) for row in target_rows]),
            "count": len(target_rows),
        }

    more_concentrated = [
        {
            "target_module": target,
            **stats,
        }
        for target, stats in by_target.items()
        if stats["mean_delta_top1"] > 0.0 and stats["mean_delta_entropy"] < 0.0
    ]

    supports_continuing = bool(
        len(rows) >= 200
        and not unmatched_backdoor
        and not unmatched_clean
        and len(more_concentrated) > 0
    )
    if supports_continuing:
        decision_reason = (
            "All expected modules matched and at least one target module shows "
            "higher backdoor top-1 energy with lower entropy than the clean structural reference."
        )
    else:
        decision_reason = (
            "Adapter-only comparison completed, but the conservative GO criterion "
            "was not met. Inspect plots and metrics before continuing."
        )

    return {
        "matched_module_count": len(rows),
        "unmatched_backdoor_modules": [
            {"layer_id": layer, "target_module": target}
            for layer, target in sorted(unmatched_backdoor, key=sort_key)
        ],
        "unmatched_clean_modules": [
            {"layer_id": layer, "target_module": target}
            for layer, target in sorted(unmatched_clean, key=sort_key)
        ],
        "mean_backdoor_top1": mean_backdoor_top1,
        "mean_clean_top1": mean_clean_top1,
        "mean_delta_top1": mean_backdoor_top1 - mean_clean_top1,
        "mean_backdoor_top3": mean_backdoor_top3,
        "mean_clean_top3": mean_clean_top3,
        "mean_delta_top3": mean_backdoor_top3 - mean_clean_top3,
        "mean_backdoor_entropy": mean_backdoor_entropy,
        "mean_clean_entropy": mean_clean_entropy,
        "mean_delta_entropy": mean_backdoor_entropy - mean_clean_entropy,
        "by_target_module": by_target,
        "module_types_where_backdoor_more_concentrated": more_concentrated,
        "go_no_go": {
            "spectral_sanity_check_supports_continuing": supports_continuing,
            "criterion": (
                "GO if all 224 structural modules match and at least one target "
                "module has mean delta_top1 > 0 and mean delta_entropy < 0."
            ),
            "reason": decision_reason,
            "caveat": (
                "This is only a structural spectral sanity check. FlagAlpha is not "
                "a perfect task/distribution-matched clean reference."
            ),
        },
        "raw_norm_caveat": NORM_CAVEAT,
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "layer_id",
        "target_module",
        "backdoor_module_name",
        "clean_module_name",
        "backdoor_rank",
        "clean_rank",
        "backdoor_lora_alpha",
        "clean_lora_alpha",
        "backdoor_lora_scale_alpha_over_rank",
        "clean_lora_scale_alpha_over_rank",
        "backdoor_top1",
        "clean_top1",
        "delta_top1",
        "backdoor_top3",
        "clean_top3",
        "delta_top3",
        "backdoor_entropy",
        "clean_entropy",
        "delta_entropy",
        "backdoor_effective_rank",
        "clean_effective_rank",
        "delta_effective_rank",
        "backdoor_fro_norm",
        "clean_fro_norm",
        "delta_fro_norm",
        "backdoor_max_singular_value",
        "clean_max_singular_value",
        "delta_max_singular_value",
        "backdoor_normalized_singular_values",
        "clean_normalized_singular_values",
        "norm_comparison_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["backdoor_normalized_singular_values"] = json.dumps(
                out["backdoor_normalized_singular_values"]
            )
            out["clean_normalized_singular_values"] = json.dumps(
                out["clean_normalized_singular_values"]
            )
            writer.writerow({field: out.get(field) for field in fieldnames})
    return path, backup


def layer_metric_series(rows: list[dict[str, Any]], prefix: str, metric: str) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    field = f"{prefix}_{metric}"
    for row in rows:
        layer = row["layer_id"]
        if layer is None:
            continue
        grouped[int(layer)].append(float(row[field]))
    layers = sorted(grouped)
    return layers, [mean(grouped[layer]) for layer in layers]


def plot_layer_metric(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    y_label: str,
    title: str,
    timestamp: str,
) -> tuple[Path, Path | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    backdoor_layers, backdoor_values = layer_metric_series(rows, "backdoor", metric)
    clean_layers, clean_values = layer_metric_series(rows, "clean", metric)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(backdoor_layers, backdoor_values, marker="o", linewidth=1.8, label="Backdoor")
    ax.plot(clean_layers, clean_values, marker="s", linewidth=1.8, label="Clean reference")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, backup


def mean_curve(rows: list[dict[str, Any]], field: str) -> list[float]:
    curves = [row[field] for row in rows if row.get(field)]
    if not curves:
        return []
    max_len = max(len(curve) for curve in curves)
    values: list[float] = []
    for index in range(max_len):
        column = [float(curve[index]) for curve in curves if index < len(curve)]
        values.append(mean(column))
    return values


def plot_mean_singular_curve(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    backdoor_curve = mean_curve(rows, "backdoor_normalized_singular_values")
    clean_curve = mean_curve(rows, "clean_normalized_singular_values")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(range(1, len(backdoor_curve) + 1), backdoor_curve, marker="o", label="Backdoor")
    ax.plot(range(1, len(clean_curve) + 1), clean_curve, marker="s", label="Clean reference")
    ax.set_title("Mean Normalized Singular-Value Curve")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value / largest singular value")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, backup


def make_plots(args: argparse.Namespace, rows: list[dict[str, Any]], timestamp: str) -> list[dict[str, Any]]:
    specs = [
        (
            Path(args.top1_plot_path),
            "top1",
            "Top-1 energy share",
            "Clean vs Backdoor Mean Top-1 Energy by Layer",
        ),
        (
            Path(args.top3_plot_path),
            "top3",
            "Top-3 energy share",
            "Clean vs Backdoor Mean Top-3 Energy by Layer",
        ),
        (
            Path(args.entropy_plot_path),
            "entropy",
            "Normalized spectral entropy",
            "Clean vs Backdoor Mean Spectral Entropy by Layer",
        ),
    ]
    outputs: list[dict[str, Any]] = []
    for path, metric, y_label, title in specs:
        written, backup = plot_layer_metric(path, rows, metric, y_label, title, timestamp)
        outputs.append({"path": str(written), "backup": str(backup) if backup else None})
    written, backup = plot_mean_singular_curve(Path(args.curve_plot_path), rows, timestamp)
    outputs.append({"path": str(written), "backup": str(backup) if backup else None})
    return outputs


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backdoor_snapshot = locate_snapshot(BACKDOOR_REPO_ID, args.backdoor_path, args.cache_root)
    clean_snapshot = locate_snapshot(CLEAN_REPO_ID, args.clean_path, args.cache_root)

    backdoor_config = load_json(adapter_config_path(backdoor_snapshot))
    clean_config = load_json(adapter_config_path(clean_snapshot))
    backdoor_alpha = backdoor_config.get("lora_alpha")
    clean_alpha = clean_config.get("lora_alpha")

    backdoor_pairs, backdoor_warnings = load_backdoor_pairs(backdoor_snapshot)
    clean_pairs, clean_state, clean_warnings = load_clean_pairs(clean_snapshot)
    common_keys = sorted(set(backdoor_pairs).intersection(clean_pairs), key=sort_key)
    unmatched_backdoor = sorted(set(backdoor_pairs).difference(clean_pairs), key=sort_key)
    unmatched_clean = sorted(set(clean_pairs).difference(backdoor_pairs), key=sort_key)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for key in common_keys:
        backdoor_pair = backdoor_pairs[key]
        clean_pair = clean_pairs[key]
        backdoor_A, backdoor_B = load_backdoor_tensors(backdoor_snapshot, backdoor_pair)
        clean_A, clean_B = load_clean_tensors(clean_state, clean_pair)

        backdoor_record = spectral_record(
            adapter_label="backdoor",
            repo_id=BACKDOOR_REPO_ID,
            pair=backdoor_pair,
            A=backdoor_A,
            B=backdoor_B,
            lora_alpha=backdoor_alpha,
        )
        clean_record = spectral_record(
            adapter_label="clean_reference",
            repo_id=CLEAN_REPO_ID,
            pair=clean_pair,
            A=clean_A,
            B=clean_B,
            lora_alpha=clean_alpha,
        )
        records.extend([backdoor_record, clean_record])
        rows.append(comparison_row(backdoor_record, clean_record))

    summary = summarize(rows, unmatched_backdoor, unmatched_clean)
    report = {
        "timestamp_utc": utc_timestamp(),
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
        "adapters": {
            "backdoor": {
                "repo_id": BACKDOOR_REPO_ID,
                "snapshot_path": str(backdoor_snapshot),
                "weights_file": str(backdoor_snapshot / "adapter_model.safetensors"),
                "weight_format": "safetensors",
                "config": backdoor_config,
                "pair_count": len(backdoor_pairs),
            },
            "clean_reference": {
                "repo_id": CLEAN_REPO_ID,
                "snapshot_path": str(clean_snapshot),
                "weights_file": str(clean_snapshot / "adapter_model.bin"),
                "weight_format": "bin",
                "bin_load_security": "torch.load(..., map_location='cpu', weights_only=True)",
                "config": clean_config,
                "pair_count": len(clean_pairs),
            },
        },
        "warnings": {
            "backdoor": backdoor_warnings,
            "clean_reference": clean_warnings,
        },
        "summary": summary,
        "records": records,
        "comparison_rows": rows,
    }
    return report, rows


def print_summary(
    report: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    csv_backup: Path | None,
    plot_outputs: list[dict[str, Any]],
) -> None:
    summary = report["summary"]
    print("Clean vs backdoor spectral comparison summary")
    print(f"- Backdoor adapter: {BACKDOOR_REPO_ID}")
    print(f"- Clean structural reference: {CLEAN_REPO_ID}")
    print(f"- Matched module count: {summary['matched_module_count']}")
    print(f"- Unmatched backdoor modules: {len(summary['unmatched_backdoor_modules'])}")
    print(f"- Unmatched clean modules: {len(summary['unmatched_clean_modules'])}")
    print(f"- Mean backdoor top1: {summary['mean_backdoor_top1']:.6f}")
    print(f"- Mean clean top1: {summary['mean_clean_top1']:.6f}")
    print(f"- Mean backdoor top3: {summary['mean_backdoor_top3']:.6f}")
    print(f"- Mean clean top3: {summary['mean_clean_top3']:.6f}")
    print(f"- Mean backdoor entropy: {summary['mean_backdoor_entropy']:.6f}")
    print(f"- Mean clean entropy: {summary['mean_clean_entropy']:.6f}")
    print("- Module types where backdoor is more concentrated than clean:")
    if summary["module_types_where_backdoor_more_concentrated"]:
        for item in summary["module_types_where_backdoor_more_concentrated"]:
            print(
                "  - "
                f"{item['target_module']}: "
                f"delta_top1={item['mean_delta_top1']:.6f}, "
                f"delta_entropy={item['mean_delta_entropy']:.6f}"
            )
    else:
        print("  - none under the conservative top1/entropy criterion")
    go_no_go = summary["go_no_go"]
    print(
        "- Spectral sanity check supports continuing: "
        f"{go_no_go['spectral_sanity_check_supports_continuing']}"
    )
    print(f"- GO/NO-GO reason: {go_no_go['reason']}")
    print(f"- Caveat: {go_no_go['caveat']}")
    print(f"- Raw norm caveat: {summary['raw_norm_caveat']}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")
    print("- Plots written:")
    for item in plot_outputs:
        print(f"  - {item['path']}")
        if item["backup"]:
            print(f"    previous file backed up to: {item['backup']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdoor-path", default=None, help="Cached BackdoorLLM adapter snapshot.")
    parser.add_argument("--clean-path", default=None, help="Cached FlagAlpha adapter snapshot.")
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Extra Hugging Face hub cache root to search. Can be passed multiple times.",
    )
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--csv-path",
        default="outputs/clean_vs_backdoor_spectral_comparison.csv",
    )
    parser.add_argument(
        "--top1-plot-path",
        default="reports/figures/clean_vs_backdoor_top1_by_layer.png",
    )
    parser.add_argument(
        "--top3-plot-path",
        default="reports/figures/clean_vs_backdoor_top3_by_layer.png",
    )
    parser.add_argument(
        "--entropy-plot-path",
        default="reports/figures/clean_vs_backdoor_entropy_by_layer.png",
    )
    parser.add_argument(
        "--curve-plot-path",
        default="reports/figures/clean_vs_backdoor_singular_curve_mean.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, rows = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"clean_vs_backdoor_spectral_comparison_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(Path(args.csv_path), rows, timestamp)
    plot_outputs = make_plots(args, rows, timestamp)
    print_summary(report, json_path, csv_path, csv_backup, plot_outputs)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
