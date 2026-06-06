"""PEFT loading smoke tests for original and sanitised LoRA adapters.

Default mode is adapter-only: it validates PEFT configs and safetensors files
without loading the full base model.

Optional mode ``--load-base-4bit`` attempts a low-memory base-model attach test
for one adapter only. It does not run generation. A tiny forward pass is only
run when ``--tiny-forward-check`` is explicitly provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
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
    classify_lora_tensor_key,
    locate_adapter_snapshot,
)


ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
BASE_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
EXPECTED_RANK = 8
EXPECTED_TENSOR_COUNT = 448
EXPECTED_PAIR_COUNT = 224
EXPECTED_TARGET_MODULES = {
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
SANITISED_VARIANTS = [
    "top1_gamma_0.0",
    "top1_gamma_0.25",
    "top1_gamma_0.50",
    "top3_gamma_0.0",
    "top3_gamma_0.25",
    "top3_gamma_0.50",
]
ATTACH_VARIANTS = [
    "original",
    "top1_gamma_0.25",
    "top1_gamma_0.50",
    "top3_gamma_0.25",
    "top3_gamma_0.50",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def normalize_peft_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if isinstance(value, (set, list, tuple)):
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


def adapter_entries(original_snapshot: Path, variants_dir: Path) -> list[dict[str, Any]]:
    entries = [
        {
            "variant_name": "original",
            "adapter_path": str(original_snapshot),
            "kind": "original",
        }
    ]
    for variant in SANITISED_VARIANTS:
        entries.append(
            {
                "variant_name": variant,
                "adapter_path": str(variants_dir / variant),
                "kind": "sanitised",
            }
        )
    return entries


def inspect_safetensors(path: Path) -> dict[str, Any]:
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
        "all_expected_targets_present": False,
    }
    if not path.exists():
        result["error"] = "adapter_model.safetensors missing"
        return result

    try:
        from safetensors import safe_open

        modules: dict[str, set[str]] = defaultdict(set)
        ranks: set[int] = set()
        hints: set[str] = set()
        lora_a = 0
        lora_b = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            for key in keys:
                kind, module_name = classify_lora_tensor_key(key)
                shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                rank = rank_from_shape(kind, shape)
                hint = target_hint(module_name)
                if kind == "A":
                    lora_a += 1
                elif kind == "B":
                    lora_b += 1
                if kind and module_name:
                    modules[module_name].add(kind)
                if rank is not None:
                    ranks.add(rank)
                if hint:
                    hints.add(hint)
        complete = sum(1 for kinds in modules.values() if {"A", "B"}.issubset(kinds))
        result.update(
            {
                "ok": True,
                "tensor_count": len(keys),
                "lora_A_count": lora_a,
                "lora_B_count": lora_b,
                "complete_pair_count": complete,
                "incomplete_pair_count": len(modules) - complete,
                "unique_ranks": sorted(ranks),
                "target_module_hints": sorted(hints),
                "all_expected_targets_present": EXPECTED_TARGET_MODULES.issubset(hints),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_peft_config(adapter_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "adapter_path": str(adapter_path),
        "ok": False,
        "error": None,
        "peft_type": None,
        "task_type": None,
        "r": None,
        "target_modules": [],
        "target_modules_ok": False,
        "rank_ok": False,
    }
    try:
        from peft import PeftConfig

        config = PeftConfig.from_pretrained(str(adapter_path))
        peft_type = normalize_peft_value(getattr(config, "peft_type", None))
        task_type = normalize_peft_value(getattr(config, "task_type", None))
        rank = getattr(config, "r", None)
        target_modules = normalize_peft_value(getattr(config, "target_modules", []))
        target_set = set(target_modules or [])
        result.update(
            {
                "ok": True,
                "peft_type": peft_type,
                "task_type": task_type,
                "r": rank,
                "target_modules": sorted(target_set),
                "target_modules_ok": EXPECTED_TARGET_MODULES.issubset(target_set),
                "rank_ok": rank == EXPECTED_RANK,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def adapter_only_check(entry: dict[str, Any]) -> dict[str, Any]:
    adapter_path = Path(entry["adapter_path"])
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_model_path(adapter_path)
    peft_config = inspect_peft_config(adapter_path)
    tensor_info = inspect_safetensors(weights_path)
    checks = {
        "adapter_dir_exists": adapter_path.exists(),
        "adapter_config_exists": config_path.exists(),
        "adapter_model_exists": weights_path.exists(),
        "peft_config_ok": peft_config["ok"],
        "peft_type_lora": peft_config.get("peft_type") == "LORA",
        "rank_ok": peft_config.get("rank_ok") is True,
        "target_modules_ok": peft_config.get("target_modules_ok") is True,
        "safetensors_ok": tensor_info["ok"],
        "tensor_count_ok": tensor_info.get("tensor_count") == EXPECTED_TENSOR_COUNT,
        "lora_A_count_ok": tensor_info.get("lora_A_count") == EXPECTED_PAIR_COUNT,
        "lora_B_count_ok": tensor_info.get("lora_B_count") == EXPECTED_PAIR_COUNT,
        "complete_pairs_ok": tensor_info.get("complete_pair_count") == EXPECTED_PAIR_COUNT,
        "incomplete_pairs_ok": tensor_info.get("incomplete_pair_count") == 0,
        "tensor_rank_ok": tensor_info.get("unique_ranks") == [EXPECTED_RANK],
        "tensor_target_modules_ok": tensor_info.get("all_expected_targets_present") is True,
    }
    return {
        **entry,
        "adapter_config_path": str(config_path),
        "adapter_model_path": str(weights_path),
        "passed": all(checks.values()),
        "checks": checks,
        "peft_config": peft_config,
        "tensor_info": tensor_info,
    }


def cuda_memory_summary(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"cuda_available": False}
    info: dict[str, Any] = {
        "cuda_available": True,
        "device_count": torch_module.cuda.device_count(),
        "devices": [],
    }
    for index in range(torch_module.cuda.device_count()):
        free, total = torch_module.cuda.mem_get_info(index)
        info["devices"].append(
            {
                "index": index,
                "name": torch_module.cuda.get_device_name(index),
                "allocated_mb": round(torch_module.cuda.memory_allocated(index) / (1024**2), 3),
                "reserved_mb": round(torch_module.cuda.memory_reserved(index) / (1024**2), 3),
                "free_mb": round(free / (1024**2), 3),
                "total_mb": round(total / (1024**2), 3),
            }
        )
    return info


def resolve_attach_adapter(variant: str, original_snapshot: Path, variants_dir: Path) -> Path:
    if variant == "original":
        return original_snapshot
    path = variants_dir / variant
    if not path.exists():
        raise FileNotFoundError(f"Adapter variant directory not found: {path}")
    return path


def first_parameter_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return None


def optional_base_attach_test(args: argparse.Namespace, original_snapshot: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "variant": args.variant,
        "base_model_id": args.base_model_id,
        "adapter_path": None,
        "base_loaded": False,
        "adapter_attached": False,
        "tiny_forward_check_attempted": args.tiny_forward_check,
        "tiny_forward_check_ok": None,
        "logits_shape": None,
        "memory_before": None,
        "memory_after_base": None,
        "memory_after_attach": None,
        "memory_after_forward": None,
        "error": None,
        "oom": False,
        "warning": None,
    }
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        result["memory_before"] = cuda_memory_summary(torch)
        adapter_path = resolve_attach_adapter(args.variant, original_snapshot, Path(args.variants_dir))
        result["adapter_path"] = str(adapter_path)
        if not torch.cuda.is_available():
            result["warning"] = "CUDA is not available; 4-bit device_map='auto' attach may fail or run on CPU."

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            local_files_only=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        result["base_loaded"] = True
        result["memory_after_base"] = cuda_memory_summary(torch)

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        result["adapter_attached"] = True
        result["memory_after_attach"] = cuda_memory_summary(torch)
        result["model_class"] = type(model).__name__
        result["first_parameter_device"] = str(first_parameter_device(model))

        if args.tiny_forward_check:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, local_files_only=True)
            inputs = tokenizer("Smoke test.", return_tensors="pt")
            device = first_parameter_device(model)
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            result["tiny_forward_check_ok"] = True
            result["logits_shape"] = [int(dim) for dim in outputs.logits.shape]
            result["memory_after_forward"] = cuda_memory_summary(torch)
    except RuntimeError as exc:
        message = str(exc)
        result["error"] = f"{type(exc).__name__}: {message}"
        result["oom"] = "out of memory" in message.lower() or "cuda" in message.lower() and "memory" in message.lower()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                result["memory_after_error_cleanup"] = cuda_memory_summary(torch)
        except Exception:
            pass
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=5)
    return result


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], attach_result: dict[str, Any], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "row_type",
        "variant_name",
        "kind",
        "adapter_path",
        "passed",
        "peft_config_ok",
        "peft_type",
        "task_type",
        "rank",
        "target_modules_ok",
        "safetensors_ok",
        "tensor_count",
        "lora_A_count",
        "lora_B_count",
        "complete_pair_count",
        "incomplete_pair_count",
        "unique_ranks",
        "adapter_attach_attempted",
        "base_loaded",
        "adapter_attached",
        "tiny_forward_check_attempted",
        "tiny_forward_check_ok",
        "oom",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            config = row["peft_config"]
            tensor = row["tensor_info"]
            writer.writerow(
                {
                    "row_type": "adapter_only",
                    "variant_name": row["variant_name"],
                    "kind": row["kind"],
                    "adapter_path": row["adapter_path"],
                    "passed": row["passed"],
                    "peft_config_ok": config.get("ok"),
                    "peft_type": config.get("peft_type"),
                    "task_type": config.get("task_type"),
                    "rank": config.get("r"),
                    "target_modules_ok": config.get("target_modules_ok"),
                    "safetensors_ok": tensor.get("ok"),
                    "tensor_count": tensor.get("tensor_count"),
                    "lora_A_count": tensor.get("lora_A_count"),
                    "lora_B_count": tensor.get("lora_B_count"),
                    "complete_pair_count": tensor.get("complete_pair_count"),
                    "incomplete_pair_count": tensor.get("incomplete_pair_count"),
                    "unique_ranks": json.dumps(tensor.get("unique_ranks")),
                    "adapter_attach_attempted": False,
                    "base_loaded": "",
                    "adapter_attached": "",
                    "tiny_forward_check_attempted": "",
                    "tiny_forward_check_ok": "",
                    "oom": "",
                    "error": config.get("error") or tensor.get("error"),
                }
            )
        if attach_result.get("attempted"):
            writer.writerow(
                {
                    "row_type": "base_attach",
                    "variant_name": attach_result.get("variant"),
                    "kind": "optional_base_attach",
                    "adapter_path": attach_result.get("adapter_path"),
                    "passed": attach_result.get("adapter_attached") is True
                    and (not attach_result.get("tiny_forward_check_attempted") or attach_result.get("tiny_forward_check_ok") is True),
                    "adapter_attach_attempted": True,
                    "base_loaded": attach_result.get("base_loaded"),
                    "adapter_attached": attach_result.get("adapter_attached"),
                    "tiny_forward_check_attempted": attach_result.get("tiny_forward_check_attempted"),
                    "tiny_forward_check_ok": attach_result.get("tiny_forward_check_ok"),
                    "oom": attach_result.get("oom"),
                    "error": attach_result.get("error"),
                }
            )
    return path, backup


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    timestamp = utc_timestamp()
    original_snapshot = locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    variants_dir = Path(args.variants_dir)
    entries = adapter_entries(original_snapshot, variants_dir)
    adapter_results = [adapter_only_check(entry) for entry in entries]
    adapter_only_passed = all(row["passed"] for row in adapter_results)

    attach_result = {"attempted": False}
    if args.load_base_4bit:
        attach_result = optional_base_attach_test(args, original_snapshot)

    attach_ok = bool(
        attach_result.get("attempted")
        and attach_result.get("adapter_attached") is True
        and (
            not attach_result.get("tiny_forward_check_attempted")
            or attach_result.get("tiny_forward_check_ok") is True
        )
    )
    recommendation = {
        "safe_to_proceed_to_peft_base_attach_test": adapter_only_passed,
        "safe_to_proceed_to_first_tiny_inference_smoke_test": attach_ok,
        "reason": (
            "Adapter-only checks passed, but base attach was not run."
            if not attach_result.get("attempted")
            else (
                "Base model and adapter attach smoke test passed."
                if attach_ok
                else "Base attach smoke test did not pass; inspect error/OOM details before continuing."
            )
        ),
    }

    report = {
        "timestamp_utc": timestamp,
        "mode": "load-base-4bit" if args.load_base_4bit else "adapter-only",
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
        "original_adapter_snapshot": str(original_snapshot),
        "variants_dir": str(variants_dir),
        "base_model_id": args.base_model_id,
        "adapter_only_summary": {
            "checked_count": len(adapter_results),
            "passed_count": sum(1 for row in adapter_results if row["passed"]),
            "failed_count": sum(1 for row in adapter_results if not row["passed"]),
            "all_passed": adapter_only_passed,
        },
        "adapter_only_results": adapter_results,
        "base_attach_result": attach_result,
        "recommendation": recommendation,
    }
    return report, adapter_results, attach_result


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, csv_backup: Path | None) -> None:
    summary = report["adapter_only_summary"]
    attach = report["base_attach_result"]
    print("PEFT loading smoke test summary")
    print(f"- Mode used: {report['mode']}")
    print(f"- Adapter-only variants checked: {summary['checked_count']}")
    print(f"- Adapter-only passed: {summary['passed_count']}")
    print(f"- Adapter-only failed: {summary['failed_count']}")
    for row in report["adapter_only_results"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            "  - "
            f"{row['variant_name']}: {status} "
            f"peft={row['peft_config'].get('ok')} "
            f"tensors={row['tensor_info'].get('tensor_count')} "
            f"pairs={row['tensor_info'].get('complete_pair_count')}"
        )
    if attach.get("attempted"):
        print("- Optional 4-bit base attach test:")
        print(f"  - Base model: {attach.get('base_model_id')}")
        print(f"  - Adapter variant: {attach.get('variant')}")
        print(f"  - Adapter path: {attach.get('adapter_path')}")
        print(f"  - Base loaded: {attach.get('base_loaded')}")
        print(f"  - Adapter attached: {attach.get('adapter_attached')}")
        print(f"  - OOM: {attach.get('oom')}")
        if attach.get("warning"):
            print(f"  - Warning: {attach.get('warning')}")
        if attach.get("error"):
            print(f"  - Error: {attach.get('error')}")
        print(f"  - GPU memory before: {attach.get('memory_before')}")
        print(f"  - GPU memory after base: {attach.get('memory_after_base')}")
        print(f"  - GPU memory after attach: {attach.get('memory_after_attach')}")
        if attach.get("tiny_forward_check_attempted"):
            print(f"  - Tiny forward ok: {attach.get('tiny_forward_check_ok')}")
            print(f"  - Logits shape: {attach.get('logits_shape')}")
    print(
        "- Safe to proceed to PEFT base attach test: "
        f"{report['recommendation']['safe_to_proceed_to_peft_base_attach_test']}"
    )
    print(
        "- Safe to proceed to first tiny inference smoke test: "
        f"{report['recommendation']['safe_to_proceed_to_first_tiny_inference_smoke_test']}"
    )
    print(f"- Recommendation reason: {report['recommendation']['reason']}")
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
    parser.add_argument("--csv-path", default="outputs/peft_loading_smoke_test_summary.csv")
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--load-base-4bit", action="store_true")
    parser.add_argument("--tiny-forward-check", action="store_true")
    parser.add_argument("--variant", choices=ATTACH_VARIANTS, default="top1_gamma_0.50")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, rows, attach_result = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"peft_loading_smoke_test_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(Path(args.csv_path), rows, attach_result, timestamp)
    print_summary(report, json_path, csv_path, csv_backup)
    if not report["adapter_only_summary"]["all_passed"]:
        return 2
    if args.load_base_4bit and not report["recommendation"]["safe_to_proceed_to_first_tiny_inference_smoke_test"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
