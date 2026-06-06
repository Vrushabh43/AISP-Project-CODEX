"""Safely download/inspect the provisional FlagAlpha clean LoRA adapter.

This is a narrow safety gate for:
FlagAlpha/Llama2-Chinese-7b-Chat-LoRA

It does not load a base model, run inference, use GPU, or implement comparison.
It only downloads adapter files when explicitly requested and verifies whether
adapter_model.bin can be read through torch.load(weights_only=True).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import classify_lora_tensor_key, hf_hub_cache_roots, repo_cache_dirname  # noqa: E402


REPO_ID = "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"
ALLOW_PATTERNS = [
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "README.md",
    ".gitattributes",
]
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass
class TensorRow:
    key: str
    shape: str
    dtype: str
    lora_kind: str | None
    module_name: str | None
    target_module_hint: str | None
    inferred_rank: int | None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def locate_snapshot(repo_id: str, cache_roots: list[str]) -> Path | None:
    dirname = repo_cache_dirname(repo_id, "model")
    candidates: list[Path] = []
    for root in hf_hub_cache_roots(cache_roots):
        snapshots = root / dirname / "snapshots"
        if snapshots.exists():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def hf_metadata(repo_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "reachable": False,
        "siblings": [],
        "tags": [],
        "private": None,
        "gated": None,
        "sha": None,
        "error": None,
    }
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id)
        result.update(
            {
                "reachable": True,
                "siblings": sorted(s.rfilename for s in (info.siblings or [])),
                "tags": list(getattr(info, "tags", None) or []),
                "private": getattr(info, "private", None),
                "gated": getattr(info, "gated", None),
                "sha": getattr(info, "sha", None),
            }
        )
    except Exception as exc:  # pragma: no cover - network/auth diagnostic path
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def download_adapter_only(repo_id: str, cache_dir: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": True,
        "repo_id": repo_id,
        "allow_patterns": ALLOW_PATTERNS,
        "local_dir": None,
        "error": None,
    }
    try:
        from huggingface_hub import snapshot_download

        result["local_dir"] = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=ALLOW_PATTERNS,
            cache_dir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover - network/auth diagnostic path
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_hint(module_name: str | None) -> str | None:
    if not module_name:
        return None
    return module_name.split(".")[-1]


def rank_from_shape(kind: str | None, shape: tuple[int, ...]) -> int | None:
    if len(shape) != 2 or kind not in {"A", "B"}:
        return None
    return int(shape[0] if kind == "A" else shape[1])


def inspect_bin_weights_only(bin_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "weights_only": True,
        "ok": False,
        "object_type": None,
        "state_dict_key_count": 0,
        "tensor_count": 0,
        "non_tensor_value_count": 0,
        "lora_A_count": 0,
        "lora_B_count": 0,
        "module_group_count": 0,
        "complete_pair_count": 0,
        "incomplete_pair_count": 0,
        "unique_ranks": [],
        "target_module_hints": [],
        "all_values_are_tensors": False,
        "error": None,
    }

    try:
        import torch

        obj = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        result["object_type"] = type(obj).__name__
        if not isinstance(obj, dict):
            result["error"] = f"weights_only load returned {type(obj).__name__}, expected dict"
            return result

        rows: list[TensorRow] = []
        modules: dict[str, dict[str, TensorRow]] = {}
        tensor_count = 0
        non_tensor_count = 0
        for key, value in sorted(obj.items()):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                non_tensor_count += 1
                continue
            tensor_count += 1
            shape = tuple(int(dim) for dim in value.shape)
            kind, module_name = classify_lora_tensor_key(str(key))
            row = TensorRow(
                key=str(key),
                shape="x".join(str(dim) for dim in shape),
                dtype=str(value.dtype),
                lora_kind=kind,
                module_name=module_name,
                target_module_hint=target_hint(module_name),
                inferred_rank=rank_from_shape(kind, shape),
            )
            rows.append(row)
            if kind and module_name:
                modules.setdefault(module_name, {})[kind] = row

        complete = sum(1 for pair in modules.values() if "A" in pair and "B" in pair)
        incomplete = len(modules) - complete
        ranks = sorted(
            {
                row.inferred_rank
                for row in rows
                if row.inferred_rank is not None and row.lora_kind in {"A", "B"}
            }
        )
        hints = sorted({row.target_module_hint for row in rows if row.target_module_hint})
        result.update(
            {
                "ok": True,
                "state_dict_key_count": len(obj),
                "tensor_count": tensor_count,
                "non_tensor_value_count": non_tensor_count,
                "lora_A_count": sum(1 for row in rows if row.lora_kind == "A"),
                "lora_B_count": sum(1 for row in rows if row.lora_kind == "B"),
                "module_group_count": len(modules),
                "complete_pair_count": complete,
                "incomplete_pair_count": incomplete,
                "unique_ranks": ranks,
                "target_module_hints": hints,
                "all_values_are_tensors": non_tensor_count == 0,
                "tensor_rows": [asdict(row) for row in rows],
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def inspect_safetensors(safetensors_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": True, "ok": False, "error": None}
    try:
        from safetensors import safe_open

        rows: list[TensorRow] = []
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                tensor_slice = handle.get_slice(key)
                shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                kind, module_name = classify_lora_tensor_key(key)
                rows.append(
                    TensorRow(
                        key=key,
                        shape="x".join(str(dim) for dim in shape),
                        dtype=str(tensor_slice.get_dtype()),
                        lora_kind=kind,
                        module_name=module_name,
                        target_module_hint=target_hint(module_name),
                        inferred_rank=rank_from_shape(kind, shape),
                    )
                )
        result.update({"ok": True, "tensor_rows": [asdict(row) for row in rows]})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "key",
        "shape",
        "dtype",
        "lora_kind",
        "module_name",
        "target_module_hint",
        "inferred_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path, backup


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    metadata = hf_metadata(REPO_ID) if not args.offline else {"repo_id": REPO_ID, "reachable": None}
    download = {"requested": False}
    if args.download:
        download = download_adapter_only(REPO_ID, args.cache_dir)

    snapshot = locate_snapshot(REPO_ID, args.cache_root)
    config_path = snapshot / "adapter_config.json" if snapshot else None
    bin_path = snapshot / "adapter_model.bin" if snapshot else None
    safetensors_path = snapshot / "adapter_model.safetensors" if snapshot else None

    config: dict[str, Any] = {}
    if config_path and config_path.exists():
        config = load_json(config_path)

    bin_inspection = {"attempted": False}
    safetensors_inspection = {"attempted": False}
    if bin_path and bin_path.exists() and args.inspect_bin:
        bin_inspection = inspect_bin_weights_only(bin_path)
    if safetensors_path and safetensors_path.exists():
        safetensors_inspection = inspect_safetensors(safetensors_path)

    tensor_rows = (
        bin_inspection.get("tensor_rows")
        or safetensors_inspection.get("tensor_rows")
        or []
    )

    can_read_as_weights = bool(
        bin_inspection.get("ok")
        and bin_inspection.get("weights_only") is True
        and bin_inspection.get("all_values_are_tensors")
        and bin_inspection.get("complete_pair_count", 0) > 0
        and bin_inspection.get("incomplete_pair_count", 1) == 0
    )

    return {
        "timestamp_utc": timestamp,
        "repo_id": REPO_ID,
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
        "metadata": metadata,
        "download": download,
        "snapshot_path": str(snapshot) if snapshot else None,
        "files": {
            "adapter_config_json": str(config_path) if config_path else None,
            "adapter_config_exists": bool(config_path and config_path.exists()),
            "adapter_model_bin": str(bin_path) if bin_path else None,
            "adapter_model_bin_exists": bool(bin_path and bin_path.exists()),
            "adapter_model_bin_size_mb": round(bin_path.stat().st_size / (1024**2), 3)
            if bin_path and bin_path.exists()
            else None,
            "adapter_model_safetensors": str(safetensors_path) if safetensors_path else None,
            "adapter_model_safetensors_exists": bool(safetensors_path and safetensors_path.exists()),
        },
        "config_summary": {
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "target_modules": config.get("target_modules"),
            "bias": config.get("bias"),
            "inference_mode": config.get("inference_mode"),
        },
        "bin_inspection": bin_inspection,
        "safetensors_inspection": safetensors_inspection,
        "can_bin_be_read_as_weights_safely": can_read_as_weights,
        "tensor_rows": tensor_rows,
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path | None, csv_backup: Path | None) -> None:
    files = report["files"]
    config = report["config_summary"]
    bin_info = report["bin_inspection"]
    print("FlagAlpha clean adapter safety inspection")
    print(f"- Repo: {report['repo_id']}")
    print(f"- Snapshot path: {report['snapshot_path']}")
    print(f"- Metadata reachable: {report['metadata'].get('reachable')}")
    print(f"- adapter_config.json exists: {files['adapter_config_exists']}")
    print(f"- adapter_model.bin exists: {files['adapter_model_bin_exists']}")
    print(f"- adapter_model.safetensors exists: {files['adapter_model_safetensors_exists']}")
    print(f"- adapter_model.bin size MB: {files['adapter_model_bin_size_mb']}")
    print(f"- PEFT type: {config.get('peft_type')}")
    print(f"- Base model: {config.get('base_model_name_or_path')}")
    print(f"- Rank r: {config.get('r')}")
    print(f"- LoRA alpha: {config.get('lora_alpha')}")
    print(f"- Target modules: {config.get('target_modules')}")
    print(f"- .bin inspected with weights_only=True: {bin_info.get('attempted')}")
    print(f"- .bin weights_only load ok: {bin_info.get('ok')}")
    print(f"- .bin can be read as tensor weights safely: {report['can_bin_be_read_as_weights_safely']}")
    if bin_info.get("attempted"):
        print(f"- Tensor count: {bin_info.get('tensor_count')}")
        print(f"- LoRA A/B counts: {bin_info.get('lora_A_count')}/{bin_info.get('lora_B_count')}")
        print(f"- Complete A/B pairs: {bin_info.get('complete_pair_count')}")
        print(f"- Incomplete A/B pairs: {bin_info.get('incomplete_pair_count')}")
        print(f"- Unique ranks: {bin_info.get('unique_ranks')}")
        print(f"- Target module hints: {bin_info.get('target_module_hints')}")
        if bin_info.get("error"):
            print(f"- .bin inspection error: {bin_info['error']}")
    print(f"- JSON log written: {json_path}")
    if csv_path:
        print(f"- Tensor CSV written: {csv_path}")
        if csv_backup:
            print(f"- Previous tensor CSV backed up to: {csv_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip Hugging Face metadata query.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download only FlagAlpha adapter files using allow_patterns.",
    )
    parser.add_argument(
        "--inspect-bin",
        action="store_true",
        help="Read adapter_model.bin with torch.load(weights_only=True, map_location='cpu').",
    )
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--cache-dir", default=None, help="Optional cache_dir for snapshot_download.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--csv-path", default="outputs/flagalpha_adapter_tensor_summary.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"flagalpha_clean_adapter_inspection_{timestamp}.json"
    write_json(json_path, report)

    csv_path = None
    csv_backup = None
    if report["tensor_rows"]:
        csv_path, csv_backup = write_csv(Path(args.csv_path), report["tensor_rows"], timestamp)

    print_summary(report, json_path, csv_path, csv_backup)
    if args.inspect_bin and not report["can_bin_be_read_as_weights_safely"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
