"""CPU-only compact spectral analysis of cached LoRA adapter tensors.

This script reads adapter_config.json and adapter_model.safetensors only. It
does not load a base model, instantiate AutoModelForCausalLM, run generation, or
use the GPU. For each LoRA pair it computes singular values of Delta W = B @ A
through a compact r x r SVD, avoiding dense Delta W construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
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
    adapter_model_path,
    group_lora_ab_pairs,
    load_adapter_config,
    load_pair_tensors,
    locate_adapter_snapshot,
)
from lora_sanitise.svd_tools import (  # noqa: E402
    compact_svd_singular_values,
    compute_effective_rank,
    compute_spectral_entropy,
    compute_topk_energy,
    frobenius_norm_from_singular_values,
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def tensor_shape_list(shape: tuple[int, ...]) -> list[int]:
    return [int(dim) for dim in shape]


def analyse_pair(handle: Any, pair: Any) -> dict[str, Any]:
    A, B = load_pair_tensors(handle, pair)
    singular_values_tensor = compact_svd_singular_values(A=A, B=B)
    singular_values = [float(value) for value in singular_values_tensor.tolist()]

    return {
        "module_name": pair.module_name,
        "layer_id": pair.layer_id,
        "target_module": pair.target_module,
        "A_key": pair.a_key,
        "B_key": pair.b_key,
        "A_shape": tensor_shape_list(pair.a_shape),
        "B_shape": tensor_shape_list(pair.b_shape),
        "A_dtype": pair.a_dtype,
        "B_dtype": pair.b_dtype,
        "rank": pair.rank,
        "frobenius_norm": frobenius_norm_from_singular_values(singular_values_tensor),
        "top1_energy_share": compute_topk_energy(singular_values_tensor, 1),
        "top3_energy_share": compute_topk_energy(singular_values_tensor, 3),
        "spectral_entropy": compute_spectral_entropy(singular_values_tensor),
        "effective_rank": compute_effective_rank(singular_values_tensor),
        "max_singular_value": max(singular_values) if singular_values else 0.0,
        "singular_values": singular_values,
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, records: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "module_name",
        "layer_id",
        "target_module",
        "A_key",
        "B_key",
        "A_shape",
        "B_shape",
        "A_dtype",
        "B_dtype",
        "rank",
        "frobenius_norm",
        "top1_energy_share",
        "top3_energy_share",
        "spectral_entropy",
        "effective_rank",
        "max_singular_value",
        "singular_values",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["A_shape"] = "x".join(str(dim) for dim in row["A_shape"])
            row["B_shape"] = "x".join(str(dim) for dim in row["B_shape"])
            row["singular_values"] = json.dumps(row["singular_values"])
            writer.writerow(row)
    return path, backup


def make_plot(path: Path, records: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)

    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for record in records:
        values = record["singular_values"]
        if not values:
            continue
        first = values[0]
        normalized = [value / first if first > 0.0 else 0.0 for value in values]
        grouped[record["target_module"]].append(normalized)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for target_module in sorted(grouped):
        spectra = grouped[target_module]
        max_len = max(len(values) for values in spectra)
        means: list[float] = []
        for index in range(max_len):
            column = [values[index] for values in spectra if index < len(values)]
            means.append(float(statistics.mean(column)))
        ax.plot(range(1, len(means) + 1), means, marker="o", linewidth=1.8, label=target_module)

    ax.set_title("Mean Normalized LoRA Singular Value Spectra by Target Module")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value / largest singular value")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, backup


def summarize(records: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    rank_values = sorted({int(record["rank"]) for record in records if record["rank"] is not None})
    top1_values = [float(record["top1_energy_share"]) for record in records]
    top3_values = [float(record["top3_energy_share"]) for record in records]
    target_counts: dict[str, int] = defaultdict(int)
    for record in records:
        target_counts[record["target_module"]] += 1

    highest = sorted(records, key=lambda item: item["top1_energy_share"], reverse=True)[:top_n]
    return {
        "pair_count": len(records),
        "rank_values": rank_values,
        "mean_top1_energy_share": float(statistics.mean(top1_values)) if top1_values else 0.0,
        "mean_top3_energy_share": float(statistics.mean(top3_values)) if top3_values else 0.0,
        "target_module_counts": dict(sorted(target_counts.items())),
        "highest_concentration_modules": [
            {
                "module_name": record["module_name"],
                "layer_id": record["layer_id"],
                "target_module": record["target_module"],
                "top1_energy_share": record["top1_energy_share"],
                "top3_energy_share": record["top3_energy_share"],
                "max_singular_value": record["max_singular_value"],
            }
            for record in highest
        ],
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from safetensors import safe_open

    adapter_path = locate_adapter_snapshot(args.adapter_path, cache_roots=args.cache_root)
    config = load_adapter_config(adapter_path)
    model_path = adapter_model_path(adapter_path)
    pairs, warnings = group_lora_ab_pairs(model_path)

    records: list[dict[str, Any]] = []
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        for pair in pairs:
            if pair.rank is None:
                continue
            records.append(analyse_pair(handle, pair))

    summary = summarize(records, args.top_n)
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
        "adapter": {
            "snapshot_path": str(adapter_path),
            "adapter_model_path": str(model_path),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "config_rank": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config.get("target_modules"),
        },
        "warnings": warnings,
        "summary": summary,
        "records": records,
    }
    return report, records


def print_summary(
    report: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    plot_path: Path,
    csv_backup: Path | None,
    plot_backup: Path | None,
) -> None:
    summary = report["summary"]
    print("Spectral stats summary")
    print(f"- Adapter snapshot: {report['adapter']['snapshot_path']}")
    print(f"- A/B pairs processed: {summary['pair_count']}")
    print(f"- Rank values: {summary['rank_values']}")
    print(f"- Mean top-1 energy share: {summary['mean_top1_energy_share']:.6f}")
    print(f"- Mean top-3 energy share: {summary['mean_top3_energy_share']:.6f}")
    print("- Highest concentration modules:")
    for item in summary["highest_concentration_modules"]:
        print(
            "  - "
            f"layer={item['layer_id']} target={item['target_module']} "
            f"top1={item['top1_energy_share']:.6f} "
            f"top3={item['top3_energy_share']:.6f} "
            f"max_s={item['max_singular_value']:.6f}"
        )
    if report["warnings"]:
        print("- Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    else:
        print("- Warnings: none")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")
    print(f"- Plot written: {plot_path}")
    if plot_backup:
        print(f"- Previous plot backed up to: {plot_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default=None, help="Cached adapter snapshot path.")
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Extra Hugging Face hub cache root to search. Can be passed multiple times.",
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory for JSON logs.")
    parser.add_argument("--csv-path", default="outputs/spectral_stats.csv")
    parser.add_argument(
        "--plot-path",
        default="reports/figures/singular_value_spectra_by_module.png",
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, records = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"spectral_stats_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(Path(args.csv_path), records, timestamp)
    plot_path, plot_backup = make_plot(Path(args.plot_path), records, timestamp)
    print_summary(report, json_path, csv_path, plot_path, csv_backup, plot_backup)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
