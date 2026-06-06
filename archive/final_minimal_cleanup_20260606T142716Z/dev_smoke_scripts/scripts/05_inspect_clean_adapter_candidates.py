"""Safely inspect clean LoRA adapter candidates.

This script is adapter-only. It does not load any base model, run inference,
use GPU, or execute code from downloaded repositories.

For adapter_model.bin files, it only uses:
torch.load(..., map_location="cpu", weights_only=True)

If weights_only=True fails, the candidate is marked unsafe. There is no unsafe
fallback loading path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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


PRIMARY_REPO_IDS = [
    "manojpatil/llama-2-7b-chat-lora-adaptor",
    "Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br",
]
FLAGALPHA_REPO_ID = "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"

ALLOW_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
    ".gitattributes",
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

LLAMA2_CHAT_MARKERS = {
    "meta-llama/llama-2-7b-chat-hf",
    "nousresearch/llama-2-7b-chat-hf",
    "llama-2-7b-chat",
    "llama_2/llama-2-7b-chat",
}


@dataclass
class TensorRow:
    repo_id: str
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


def safe_repo_name(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", repo_id).strip("_")


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def target_hint(module_name: str | None) -> str | None:
    if not module_name:
        return None
    return module_name.split(".")[-1]


def rank_from_shape(kind: str | None, shape: tuple[int, ...]) -> int | None:
    if len(shape) != 2 or kind not in {"A", "B"}:
        return None
    return int(shape[0] if kind == "A" else shape[1])


def summarize_tensor_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, set[str]] = {}
    ranks: set[int] = set()
    dtypes: set[str] = set()
    hints: set[str] = set()

    for row in rows:
        kind = row.get("lora_kind")
        module_name = row.get("module_name")
        rank = row.get("inferred_rank")
        dtype = row.get("dtype")
        hint = row.get("target_module_hint")
        if module_name and kind:
            modules.setdefault(str(module_name), set()).add(str(kind))
        if isinstance(rank, int):
            ranks.add(rank)
        elif isinstance(rank, str) and rank.isdigit():
            ranks.add(int(rank))
        if dtype:
            dtypes.add(str(dtype))
        if hint:
            hints.add(str(hint))

    complete = sum(1 for kinds in modules.values() if {"A", "B"}.issubset(kinds))
    incomplete = len(modules) - complete
    return {
        "tensor_count": len(rows),
        "lora_A_count": sum(1 for row in rows if row.get("lora_kind") == "A"),
        "lora_B_count": sum(1 for row in rows if row.get("lora_kind") == "B"),
        "module_group_count": len(modules),
        "complete_pair_count": complete,
        "incomplete_pair_count": incomplete,
        "unique_ranks": sorted(ranks),
        "dtypes": sorted(dtypes),
        "target_module_hints": sorted(hints),
    }


def inspect_safetensors(repo_id: str, safetensors_path: Path) -> dict[str, Any]:
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
                        repo_id=repo_id,
                        key=key,
                        shape="x".join(str(dim) for dim in shape),
                        dtype=str(tensor_slice.get_dtype()),
                        lora_kind=kind,
                        module_name=module_name,
                        target_module_hint=target_hint(module_name),
                        inferred_rank=rank_from_shape(kind, shape),
                    )
                )
        tensor_rows = [asdict(row) for row in rows]
        result.update({"ok": True, "tensor_rows": tensor_rows, **summarize_tensor_rows(tensor_rows)})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_bin_weights_only(repo_id: str, bin_path: Path) -> dict[str, Any]:
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
        "dtypes": [],
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
        non_tensor_count = 0
        for key, value in sorted(obj.items()):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                non_tensor_count += 1
                continue
            shape = tuple(int(dim) for dim in value.shape)
            kind, module_name = classify_lora_tensor_key(str(key))
            rows.append(
                TensorRow(
                    repo_id=repo_id,
                    key=str(key),
                    shape="x".join(str(dim) for dim in shape),
                    dtype=str(value.dtype),
                    lora_kind=kind,
                    module_name=module_name,
                    target_module_hint=target_hint(module_name),
                    inferred_rank=rank_from_shape(kind, shape),
                )
            )

        tensor_rows = [asdict(row) for row in rows]
        summary = summarize_tensor_rows(tensor_rows)
        result.update(
            {
                "ok": True,
                "state_dict_key_count": len(obj),
                "non_tensor_value_count": non_tensor_count,
                "all_values_are_tensors": non_tensor_count == 0,
                "tensor_rows": tensor_rows,
                **summary,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def config_target_modules(config: dict[str, Any]) -> list[str]:
    modules = config.get("target_modules")
    if modules is None:
        return []
    if isinstance(modules, str):
        return [modules]
    if isinstance(modules, list):
        return [str(item) for item in modules]
    return sorted(str(item) for item in modules)


def base_model_is_compatible(base_model: str | None) -> bool:
    if not base_model:
        return False
    lowered = base_model.lower()
    return any(marker in lowered for marker in LLAMA2_CHAT_MARKERS)


def safety_notes_for(result: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    files = result["files"]
    bin_info = result["bin_inspection"]
    safe_info = result["safetensors_inspection"]

    if files["adapter_model_safetensors_exists"] and safe_info.get("ok"):
        notes.append("safetensors inspected")
    if files["adapter_model_bin_exists"]:
        if bin_info.get("attempted"):
            notes.append("adapter_model.bin inspected with weights_only=True")
            if bin_info.get("ok"):
                notes.append("weights_only=True load ok")
            else:
                notes.append(f"weights_only=True failed: {bin_info.get('error')}")
        else:
            notes.append("adapter_model.bin present but not inspected")
    if not files["adapter_model_safetensors_exists"] and not files["adapter_model_bin_exists"]:
        notes.append("no adapter weight file found")
    if files["adapter_model_bin_exists"]:
        notes.append("normal torch.load was not used")
    if not result.get("can_weights_be_read_safely"):
        notes.append("candidate not yet safe for spectral comparison")
    return notes


def structural_score(result: dict[str, Any]) -> int:
    score = 0
    config = result["config_summary"]
    files = result["files"]
    tensor = result["tensor_summary"]

    if result["metadata"].get("reachable") is True:
        score += 1
    if files["adapter_config_exists"]:
        score += 2
    if config.get("peft_type") == "LORA":
        score += 2
    if base_model_is_compatible(config.get("base_model_name_or_path")):
        score += 2
    if config.get("r") == 8 or tensor["unique_ranks"] == [8]:
        score += 2
    if result["full_required_target_overlap"]:
        score += 4
    else:
        score += min(len(result["target_overlap"]), 3)
    if tensor["complete_pair_count"] > 0 and tensor["incomplete_pair_count"] == 0:
        score += 4
    if files["adapter_model_safetensors_exists"] and result["safetensors_inspection"].get("ok"):
        score += 2
    elif result["bin_can_be_read_safely"]:
        score += 1
    return score


def inspect_repo(
    repo_id: str,
    args: argparse.Namespace,
    *,
    comparison_only: bool = False,
) -> dict[str, Any]:
    metadata = hf_metadata(repo_id) if not args.offline else {"repo_id": repo_id, "reachable": None}
    download = {"requested": False, "skipped_reason": None}
    if args.download and not comparison_only:
        download = download_adapter_only(repo_id, args.cache_dir)
    elif args.download and comparison_only:
        download = {
            "requested": False,
            "skipped_reason": "comparison reference; existing cache/log preferred",
        }

    snapshot = locate_snapshot(repo_id, args.cache_root)
    config_path = snapshot / "adapter_config.json" if snapshot else None
    bin_path = snapshot / "adapter_model.bin" if snapshot else None
    safetensors_path = snapshot / "adapter_model.safetensors" if snapshot else None

    config: dict[str, Any] = {}
    if config_path and config_path.exists():
        config = load_json(config_path)

    safetensors_inspection = {"attempted": False, "ok": False}
    bin_inspection = {"attempted": False, "ok": False, "weights_only": None}
    if safetensors_path and safetensors_path.exists():
        safetensors_inspection = inspect_safetensors(repo_id, safetensors_path)
    if bin_path and bin_path.exists() and args.inspect_bin:
        bin_inspection = inspect_bin_weights_only(repo_id, bin_path)

    tensor_rows = safetensors_inspection.get("tensor_rows") or bin_inspection.get("tensor_rows") or []
    tensor_summary = summarize_tensor_rows(tensor_rows)
    config_targets = config_target_modules(config)
    observed_targets = set(tensor_summary["target_module_hints"]) or set(config_targets)
    target_overlap = sorted(REQUIRED_TARGET_MODULES.intersection(observed_targets))

    bin_can_be_read_safely = bool(
        bin_inspection.get("attempted")
        and bin_inspection.get("weights_only") is True
        and bin_inspection.get("ok")
        and bin_inspection.get("all_values_are_tensors")
        and tensor_summary["complete_pair_count"] > 0
        and tensor_summary["incomplete_pair_count"] == 0
    )
    safetensors_can_be_read_safely = bool(safetensors_inspection.get("ok"))
    can_weights_be_read_safely = safetensors_can_be_read_safely or bin_can_be_read_safely

    result: dict[str, Any] = {
        "repo_id": repo_id,
        "comparison_only": comparison_only,
        "metadata": metadata,
        "download": download,
        "snapshot_path": str(snapshot) if snapshot else None,
        "files": {
            "adapter_config_json": str(config_path) if config_path else None,
            "adapter_config_exists": bool(config_path and config_path.exists()),
            "adapter_model_safetensors": str(safetensors_path) if safetensors_path else None,
            "adapter_model_safetensors_exists": bool(safetensors_path and safetensors_path.exists()),
            "adapter_model_safetensors_size_mb": round(
                safetensors_path.stat().st_size / (1024**2), 3
            )
            if safetensors_path and safetensors_path.exists()
            else None,
            "adapter_model_bin": str(bin_path) if bin_path else None,
            "adapter_model_bin_exists": bool(bin_path and bin_path.exists()),
            "adapter_model_bin_size_mb": round(bin_path.stat().st_size / (1024**2), 3)
            if bin_path and bin_path.exists()
            else None,
        },
        "config_summary": {
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "target_modules": config_targets,
            "bias": config.get("bias"),
            "inference_mode": config.get("inference_mode"),
        },
        "safetensors_inspection": safetensors_inspection,
        "bin_inspection": bin_inspection,
        "tensor_summary": tensor_summary,
        "tensor_rows": tensor_rows,
        "target_overlap": target_overlap,
        "target_overlap_count": len(target_overlap),
        "full_required_target_overlap": REQUIRED_TARGET_MODULES.issubset(observed_targets),
        "base_model_compatible": base_model_is_compatible(config.get("base_model_name_or_path")),
        "bin_can_be_read_safely": bin_can_be_read_safely,
        "safetensors_can_be_read_safely": safetensors_can_be_read_safely,
        "can_weights_be_read_safely": can_weights_be_read_safely,
    }
    result["safety_notes"] = safety_notes_for(result)
    result["structural_suitability_score"] = structural_score(result)
    return result


def latest_flagalpha_log(logs_dir: Path) -> Path | None:
    candidates = sorted(logs_dir.glob("flagalpha_clean_adapter_inspection_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flagalpha_from_existing_log(logs_dir: Path) -> dict[str, Any] | None:
    path = latest_flagalpha_log(logs_dir)
    if path is None:
        return None
    data = load_json(path)
    config = data.get("config_summary", {})
    files = data.get("files", {})
    bin_info = data.get("bin_inspection", {})
    safe_info = data.get("safetensors_inspection", {})
    tensor_rows = data.get("tensor_rows", [])
    for row in tensor_rows:
        row.setdefault("repo_id", FLAGALPHA_REPO_ID)
    tensor_summary = summarize_tensor_rows(tensor_rows)
    observed_targets = set(tensor_summary["target_module_hints"]) or set(config_target_modules(config))
    target_overlap = sorted(REQUIRED_TARGET_MODULES.intersection(observed_targets))

    result: dict[str, Any] = {
        "repo_id": FLAGALPHA_REPO_ID,
        "comparison_only": True,
        "loaded_from_existing_log": str(path),
        "metadata": data.get("metadata", {"reachable": None}),
        "download": {"requested": False, "skipped_reason": "loaded from existing safety-gate log"},
        "snapshot_path": data.get("snapshot_path"),
        "files": {
            "adapter_config_json": files.get("adapter_config_json"),
            "adapter_config_exists": bool(files.get("adapter_config_exists")),
            "adapter_model_safetensors": files.get("adapter_model_safetensors"),
            "adapter_model_safetensors_exists": bool(files.get("adapter_model_safetensors_exists")),
            "adapter_model_safetensors_size_mb": files.get("adapter_model_safetensors_size_mb"),
            "adapter_model_bin": files.get("adapter_model_bin"),
            "adapter_model_bin_exists": bool(files.get("adapter_model_bin_exists")),
            "adapter_model_bin_size_mb": files.get("adapter_model_bin_size_mb"),
        },
        "config_summary": {
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "target_modules": config_target_modules(config),
            "bias": config.get("bias"),
            "inference_mode": config.get("inference_mode"),
        },
        "safetensors_inspection": safe_info,
        "bin_inspection": bin_info,
        "tensor_summary": tensor_summary,
        "tensor_rows": tensor_rows,
        "target_overlap": target_overlap,
        "target_overlap_count": len(target_overlap),
        "full_required_target_overlap": REQUIRED_TARGET_MODULES.issubset(observed_targets),
        "base_model_compatible": base_model_is_compatible(config.get("base_model_name_or_path")),
        "bin_can_be_read_safely": bool(data.get("can_bin_be_read_as_weights_safely")),
        "safetensors_can_be_read_safely": bool(safe_info.get("ok")),
        "can_weights_be_read_safely": bool(
            data.get("can_bin_be_read_as_weights_safely") or safe_info.get("ok")
        ),
    }
    result["safety_notes"] = safety_notes_for(result)
    result["structural_suitability_score"] = structural_score(result)
    return result


def comparison_row(result: dict[str, Any]) -> dict[str, Any]:
    files = result["files"]
    config = result["config_summary"]
    tensor = result["tensor_summary"]
    return {
        "repo_id": result["repo_id"],
        "comparison_only": result.get("comparison_only", False),
        "metadata_reachable": result["metadata"].get("reachable"),
        "snapshot_path": result.get("snapshot_path"),
        "adapter_config_exists": files["adapter_config_exists"],
        "adapter_model_safetensors_exists": files["adapter_model_safetensors_exists"],
        "adapter_model_bin_exists": files["adapter_model_bin_exists"],
        "bin_inspected_weights_only": result["bin_inspection"].get("attempted"),
        "bin_can_be_read_safely": result["bin_can_be_read_safely"],
        "can_weights_be_read_safely": result["can_weights_be_read_safely"],
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "base_model_compatible": result["base_model_compatible"],
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": ",".join(config.get("target_modules") or []),
        "tensor_count": tensor["tensor_count"],
        "lora_A_count": tensor["lora_A_count"],
        "lora_B_count": tensor["lora_B_count"],
        "complete_pair_count": tensor["complete_pair_count"],
        "incomplete_pair_count": tensor["incomplete_pair_count"],
        "unique_ranks": ",".join(str(item) for item in tensor["unique_ranks"]),
        "dtypes": ",".join(tensor["dtypes"]),
        "target_module_hints": ",".join(tensor["target_module_hints"]),
        "target_overlap": ",".join(result["target_overlap"]),
        "target_overlap_count": result["target_overlap_count"],
        "full_required_target_overlap": result["full_required_target_overlap"],
        "safety_notes": " | ".join(result["safety_notes"]),
        "structural_suitability_score": result["structural_suitability_score"],
        "loaded_from_existing_log": result.get("loaded_from_existing_log"),
    }


def recommend(results: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [item for item in results if item.get("can_weights_be_read_safely")]
    if not usable:
        return {
            "repo_id": None,
            "reason": "No candidate passed safe adapter-weight inspection.",
            "safe_to_proceed_to_spectral_comparison": False,
        }

    def key(item: dict[str, Any]) -> tuple[int, int, int, int, int]:
        tensor = item["tensor_summary"]
        ranks = tensor["unique_ranks"]
        return (
            int(item["full_required_target_overlap"]),
            int(ranks == [8]),
            int(item["base_model_compatible"]),
            int(item["files"]["adapter_model_safetensors_exists"]),
            int(item["structural_suitability_score"]),
        )

    best = max(usable, key=key)
    return {
        "repo_id": best["repo_id"],
        "reason": (
            "Highest safe structural match by full target-module overlap, rank 8, "
            "base compatibility, and adapter file safety."
        ),
        "safe_to_proceed_to_spectral_comparison": bool(
            best["can_weights_be_read_safely"]
            and best["tensor_summary"]["complete_pair_count"] > 0
            and best["tensor_summary"]["incomplete_pair_count"] == 0
        ),
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_comparison_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "repo_id",
        "comparison_only",
        "metadata_reachable",
        "snapshot_path",
        "adapter_config_exists",
        "adapter_model_safetensors_exists",
        "adapter_model_bin_exists",
        "bin_inspected_weights_only",
        "bin_can_be_read_safely",
        "can_weights_be_read_safely",
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "base_model_compatible",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "tensor_count",
        "lora_A_count",
        "lora_B_count",
        "complete_pair_count",
        "incomplete_pair_count",
        "unique_ranks",
        "dtypes",
        "target_module_hints",
        "target_overlap",
        "target_overlap_count",
        "full_required_target_overlap",
        "safety_notes",
        "structural_suitability_score",
        "loaded_from_existing_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return backup


def write_tensor_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "repo_id",
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
    return backup


def print_summary(
    results: list[dict[str, Any]],
    recommendation: dict[str, Any],
    json_path: Path,
    comparison_csv: Path,
    comparison_backup: Path | None,
    tensor_csv_paths: list[Path],
    tensor_backups: list[Path],
) -> None:
    print("Clean adapter candidate inspection summary")
    print("- Candidates inspected:")
    for result in results:
        tensor = result["tensor_summary"]
        files = result["files"]
        print(f"  - {result['repo_id']}")
        print(f"    safe weights: {result['can_weights_be_read_safely']}")
        print(f"    rank values: {tensor['unique_ranks']}")
        print(f"    all 7 target modules: {result['full_required_target_overlap']}")
        print(
            "    weight format: "
            f"safetensors={files['adapter_model_safetensors_exists']} "
            f"bin={files['adapter_model_bin_exists']}"
        )
        print(f"    complete A/B pairs: {tensor['complete_pair_count']}")
        print(f"    suitability score: {result['structural_suitability_score']}")
    safe = [item["repo_id"] for item in results if item["can_weights_be_read_safely"]]
    rank8 = [
        item["repo_id"]
        for item in results
        if item["tensor_summary"]["unique_ranks"] == [8] or item["config_summary"].get("r") == 8
    ]
    all_targets = [item["repo_id"] for item in results if item["full_required_target_overlap"]]
    safetensors = [
        item["repo_id"] for item in results if item["files"]["adapter_model_safetensors_exists"]
    ]
    bin_repos = [item["repo_id"] for item in results if item["files"]["adapter_model_bin_exists"]]
    print(f"- Safe-loading candidates: {safe}")
    print(f"- Rank-8 candidates: {rank8}")
    print(f"- Full 7-target-module candidates: {all_targets}")
    print(f"- Safetensors candidates: {safetensors}")
    print(f"- Bin candidates: {bin_repos}")
    print(f"- Recommended best clean reference candidate: {recommendation['repo_id']}")
    print(f"- Recommendation reason: {recommendation['reason']}")
    print(
        "- Safe to proceed to clean-vs-backdoor spectral comparison: "
        f"{recommendation['safe_to_proceed_to_spectral_comparison']}"
    )
    print(f"- JSON log written: {json_path}")
    print(f"- CSV comparison written: {comparison_csv}")
    if comparison_backup:
        print(f"- Previous comparison CSV backed up to: {comparison_backup}")
    if tensor_csv_paths:
        print("- Tensor CSVs written:")
        for path in tensor_csv_paths:
            print(f"  - {path}")
    if tensor_backups:
        print("- Previous tensor CSVs backed up:")
        for path in tensor_backups:
            print(f"  - {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help="Repo ID to inspect. Can be passed multiple times. Defaults to the two candidates.",
    )
    parser.add_argument("--offline", action="store_true", help="Skip Hugging Face metadata queries.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download only allowlisted adapter files for the two requested candidates.",
    )
    parser.add_argument(
        "--inspect-bin",
        action="store_true",
        help="Inspect adapter_model.bin with torch.load(weights_only=True, map_location='cpu').",
    )
    parser.add_argument(
        "--include-flagalpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the already verified FlagAlpha reference row when possible.",
    )
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--cache-dir", default=None, help="Optional cache_dir for snapshot_download.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument(
        "--comparison-csv",
        default="outputs/clean_adapter_candidate_comparison.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    repo_ids = args.repo_ids or PRIMARY_REPO_IDS

    results = [inspect_repo(repo_id, args) for repo_id in repo_ids]
    if args.include_flagalpha:
        flagalpha = flagalpha_from_existing_log(Path(args.logs_dir))
        if flagalpha is None:
            flagalpha = inspect_repo(FLAGALPHA_REPO_ID, args, comparison_only=True)
        results.append(flagalpha)

    recommendation = recommend(results)
    report = {
        "timestamp_utc": timestamp,
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
        "allow_patterns": ALLOW_PATTERNS,
        "required_target_modules": sorted(REQUIRED_TARGET_MODULES),
        "results": results,
        "recommendation": recommendation,
    }

    logs_dir = Path(args.logs_dir)
    json_path = logs_dir / f"clean_adapter_candidate_inspection_{timestamp}.json"
    write_json(json_path, report)

    comparison_rows = [comparison_row(result) for result in results]
    comparison_csv = Path(args.comparison_csv)
    comparison_backup = write_comparison_csv(comparison_csv, comparison_rows, timestamp)

    tensor_csv_paths: list[Path] = []
    tensor_backups: list[Path] = []
    outputs_dir = Path(args.outputs_dir)
    for result in results:
        rows = result.get("tensor_rows") or []
        if not rows:
            continue
        path = outputs_dir / f"clean_candidate_{safe_repo_name(result['repo_id'])}_tensor_summary.csv"
        backup = write_tensor_csv(path, rows, timestamp)
        tensor_csv_paths.append(path)
        if backup:
            tensor_backups.append(backup)

    print_summary(
        results,
        recommendation,
        json_path,
        comparison_csv,
        comparison_backup,
        tensor_csv_paths,
        tensor_backups,
    )

    unsafe_primary = [
        result["repo_id"]
        for result in results
        if not result.get("comparison_only") and result["files"]["adapter_model_bin_exists"]
        and args.inspect_bin
        and not result["bin_can_be_read_safely"]
    ]
    if unsafe_primary:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
