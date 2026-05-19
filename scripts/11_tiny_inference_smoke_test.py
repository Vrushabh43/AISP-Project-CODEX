"""Tiny bounded generation smoke test for original and one sanitised adapter.

This script is intentionally not an ASR or clean-utility evaluation. It checks
that cached 4-bit base loading, one-at-a-time PEFT adapter attach, and very
short deterministic generation work on the server.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import locate_adapter_snapshot  # noqa: E402


ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
BASE_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
DEFAULT_VARIANTS = ["original", "top1_gamma_0.50"]
ALLOWED_VARIANTS = tuple(DEFAULT_VARIANTS)
DEFAULT_PROMPTS = [
    "Write one short sentence about machine learning.",
    "Explain what a neural network is in one sentence.",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def cuda_memory_summary(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"cuda_available": False}
    devices = []
    for index in range(torch_module.cuda.device_count()):
        free, total = torch_module.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": torch_module.cuda.get_device_name(index),
                "allocated_mb": round(torch_module.cuda.memory_allocated(index) / (1024**2), 3),
                "reserved_mb": round(torch_module.cuda.memory_reserved(index) / (1024**2), 3),
                "free_mb": round(free / (1024**2), 3),
                "total_mb": round(total / (1024**2), 3),
            }
        )
    return {
        "cuda_available": True,
        "device_count": torch_module.cuda.device_count(),
        "devices": devices,
    }


def first_parameter_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return None


def resolve_adapter_path(variant: str, original_snapshot: Path, variants_dir: Path) -> Path:
    if variant == "original":
        return original_snapshot
    adapter_path = variants_dir / variant
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter variant directory not found: {adapter_path}")
    return adapter_path


def preview_text(text: str, max_chars: int = 240) -> str:
    collapsed = " ".join(text.strip().split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def generate_for_prompt(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt": prompt,
        "succeeded": False,
        "error": None,
        "input_token_count": None,
        "generated_token_count": None,
        "generated_text": None,
        "generated_text_preview": None,
    }
    inputs = tokenizer(prompt, return_tensors="pt")
    input_token_count = int(inputs["input_ids"].shape[-1])
    device = first_parameter_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch_module.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    result.update(
        {
            "succeeded": True,
            "input_token_count": input_token_count,
            "generated_token_count": int(generated_ids.shape[-1]),
            "generated_text": generated_text,
            "generated_text_preview": preview_text(generated_text),
        }
    )
    return result


def load_and_generate_for_variant(
    args: argparse.Namespace,
    variant: str,
    original_snapshot: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "variant": variant,
        "adapter_path": None,
        "base_loaded": False,
        "adapter_attached": False,
        "generation_succeeded": False,
        "all_prompts_succeeded": False,
        "oom": False,
        "error": None,
        "traceback": None,
        "memory_before": None,
        "memory_after_base": None,
        "memory_after_attach": None,
        "memory_after_generation": None,
        "memory_after_cleanup": None,
        "prompt_results": [],
    }
    model = None
    tokenizer = None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        result["memory_before"] = cuda_memory_summary(torch)
        adapter_path = resolve_adapter_path(variant, original_snapshot, Path(args.variants_dir))
        result["adapter_path"] = str(adapter_path)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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

        prompt_results = [
            generate_for_prompt(model, tokenizer, torch, prompt, args.max_new_tokens)
            for prompt in args.prompts
        ]
        result["prompt_results"] = prompt_results
        result["all_prompts_succeeded"] = all(item["succeeded"] for item in prompt_results)
        result["generation_succeeded"] = result["all_prompts_succeeded"]
        result["memory_after_generation"] = cuda_memory_summary(torch)
    except RuntimeError as exc:
        message = str(exc)
        result["error"] = f"{type(exc).__name__}: {message}"
        result["oom"] = "out of memory" in message.lower() or (
            "cuda" in message.lower() and "memory" in message.lower()
        )
        result["traceback"] = traceback.format_exc(limit=5)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=5)
    finally:
        try:
            import torch

            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                result["memory_after_cleanup"] = cuda_memory_summary(torch)
        except Exception:
            pass
    return result


def base_report(
    args: argparse.Namespace,
    timestamp: str,
    original_snapshot: Path,
    variant_results: list[dict[str, Any]],
    execution_mode: str,
    child_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    all_generation_succeeded = bool(variant_results) and all(
        item.get("generation_succeeded") is True for item in variant_results
    )
    no_oom = all(item.get("oom") is not True for item in variant_results)
    return {
        "timestamp_utc": timestamp,
        "script": "scripts/11_tiny_inference_smoke_test.py",
        "purpose": "bounded_generation_smoke_test_not_asr_or_clean_utility",
        "execution_mode": execution_mode,
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
        "base_model_id": args.base_model_id,
        "original_adapter_id": ADAPTER_ID,
        "original_adapter_snapshot": str(original_snapshot),
        "variants_dir": args.variants_dir,
        "variants_requested": args.variants,
        "prompts": args.prompts,
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "batch_size": 1,
        },
        "safety_scope": {
            "runs_asr_evaluation": False,
            "runs_clean_utility_evaluation": False,
            "tests_all_variants": False,
            "modifies_original_adapter_or_cache": False,
        },
        "child_processes": child_processes or [],
        "variant_results": variant_results,
        "summary": {
            "variants_checked": len(variant_results),
            "prompts_per_variant": len(args.prompts),
            "all_generation_succeeded": all_generation_succeeded,
            "any_oom": not no_oom,
            "safe_to_proceed_to_small_baseline_evaluation": all_generation_succeeded and no_oom,
            "recommendation": (
                "Generation smoke test passed. Next step can be a small bounded baseline-evaluation script."
                if all_generation_succeeded and no_oom
                else "Do not proceed to baseline evaluation; inspect OOM/error details first."
            ),
        },
    }


def build_in_process_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    original_snapshot = locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    variant_results = []
    for variant in args.variants:
        variant_results.append(load_and_generate_for_variant(args, variant, original_snapshot))
        if variant_results[-1].get("oom"):
            break
    return base_report(args, timestamp, original_snapshot, variant_results, "single_process")


def child_command(args: argparse.Namespace, variant: str, child_result_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-in-current-process",
        "--child-result-path",
        str(child_result_path),
        "--base-model-id",
        args.base_model_id,
        "--variants-dir",
        args.variants_dir,
        "--logs-dir",
        args.logs_dir,
        "--csv-path",
        args.csv_path,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--variant",
        variant,
    ]
    if args.original_adapter_path:
        command.extend(["--original-adapter-path", args.original_adapter_path])
    for cache_root in args.cache_root:
        command.extend(["--cache-root", cache_root])
    for prompt in args.prompts:
        command.extend(["--prompt", prompt])
    return command



def synthesize_child_failure(variant: str, child_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant,
        "adapter_path": None,
        "base_loaded": False,
        "adapter_attached": False,
        "generation_succeeded": False,
        "all_prompts_succeeded": False,
        "oom": False,
        "error": child_info.get("error") or f"Child process failed with return code {child_info.get('returncode')}",
        "traceback": None,
        "memory_before": None,
        "memory_after_base": None,
        "memory_after_attach": None,
        "memory_after_generation": None,
        "memory_after_cleanup": None,
        "prompt_results": [],
    }


def build_isolated_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    original_snapshot = locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    child_dir = Path(args.logs_dir) / f"tiny_inference_children_{timestamp}"
    child_dir.mkdir(parents=True, exist_ok=True)
    variant_results: list[dict[str, Any]] = []
    child_processes: list[dict[str, Any]] = []

    for variant in args.variants:
        child_result_path = child_dir / f"{safe_name(variant)}.json"
        command = child_command(args, variant, child_result_path)
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        child_info: dict[str, Any] = {
            "variant": variant,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "child_result_path": str(child_result_path),
            "error": None,
        }
        if child_result_path.exists():
            child_report = json.loads(child_result_path.read_text(encoding="utf-8"))
            child_info["child_timestamp_utc"] = child_report.get("timestamp_utc")
            if child_report.get("variant_results"):
                variant_results.append(child_report["variant_results"][0])
            else:
                child_info["error"] = "Child result JSON contained no variant_results."
                variant_results.append(synthesize_child_failure(variant, child_info))
        else:
            child_info["error"] = "Child process did not write a result JSON."
            variant_results.append(synthesize_child_failure(variant, child_info))
        child_processes.append(child_info)
        if variant_results[-1].get("oom"):
            break

    return base_report(
        args,
        timestamp,
        original_snapshot,
        variant_results,
        "isolated_subprocess_per_adapter",
        child_processes,
    )


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, report: dict[str, Any]) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, report["timestamp_utc"])
    fieldnames = [
        "variant",
        "adapter_path",
        "base_loaded",
        "adapter_attached",
        "generation_succeeded",
        "prompt_index",
        "prompt",
        "prompt_succeeded",
        "input_token_count",
        "generated_token_count",
        "generated_text_preview",
        "oom",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant_result in report["variant_results"]:
            prompts = variant_result.get("prompt_results") or [None]
            for index, prompt_result in enumerate(prompts):
                prompt_row = prompt_result or {}
                writer.writerow(
                    {
                        "variant": variant_result.get("variant"),
                        "adapter_path": variant_result.get("adapter_path"),
                        "base_loaded": variant_result.get("base_loaded"),
                        "adapter_attached": variant_result.get("adapter_attached"),
                        "generation_succeeded": variant_result.get("generation_succeeded"),
                        "prompt_index": index,
                        "prompt": prompt_row.get("prompt"),
                        "prompt_succeeded": prompt_row.get("succeeded"),
                        "input_token_count": prompt_row.get("input_token_count"),
                        "generated_token_count": prompt_row.get("generated_token_count"),
                        "generated_text_preview": prompt_row.get("generated_text_preview"),
                        "oom": variant_result.get("oom"),
                        "error": variant_result.get("error") or prompt_row.get("error"),
                    }
                )
    return path, backup


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, csv_backup: Path | None) -> None:
    print("Tiny inference smoke test summary")
    print(f"- Base model: {report['base_model_id']}")
    print(f"- Execution mode: {report.get('execution_mode')}")
    print(f"- Variants checked: {report['summary']['variants_checked']}")
    print(f"- Prompts per variant: {report['summary']['prompts_per_variant']}")
    print(f"- max_new_tokens: {report['generation_settings']['max_new_tokens']}")
    for variant_result in report["variant_results"]:
        print(f"- Adapter loaded: {variant_result.get('variant')}")
        print(f"  - Adapter path: {variant_result.get('adapter_path')}")
        print(f"  - Base loaded: {variant_result.get('base_loaded')}")
        print(f"  - Adapter attached: {variant_result.get('adapter_attached')}")
        print(f"  - Generation succeeded: {variant_result.get('generation_succeeded')}")
        print(f"  - OOM: {variant_result.get('oom')}")
        if variant_result.get("error"):
            print(f"  - Error: {variant_result.get('error')}")
        print(f"  - GPU memory before: {variant_result.get('memory_before')}")
        print(f"  - GPU memory after generation: {variant_result.get('memory_after_generation')}")
        print(f"  - GPU memory after cleanup: {variant_result.get('memory_after_cleanup')}")
        for index, prompt_result in enumerate(variant_result.get("prompt_results", []), start=1):
            print(f"  - Prompt {index}: {prompt_result.get('prompt')}")
            print(f"    - succeeded: {prompt_result.get('succeeded')}")
            print(f"    - generated preview: {prompt_result.get('generated_text_preview')}")
    print(
        "- Safe to proceed to small baseline evaluation: "
        f"{report['summary']['safe_to_proceed_to_small_baseline_evaluation']}"
    )
    print(f"- Recommendation: {report['summary']['recommendation']}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV summary written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--variants-dir", default="outputs/sanitised_adapters")
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--csv-path", default="outputs/tiny_inference_smoke_test_summary.csv")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--run-in-current-process",
        action="store_true",
        help="Debug mode: run variants sequentially in this process instead of isolated child processes.",
    )
    parser.add_argument("--child-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=ALLOWED_VARIANTS,
        help="Variant to test. Can be repeated. Defaults to original and top1_gamma_0.50 only.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Override smoke-test prompt. Can be repeated; keep this tiny.",
    )
    args = parser.parse_args()
    if args.variants is None:
        args.variants = list(DEFAULT_VARIANTS)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    if args.max_new_tokens > 20:
        parser.error("--max-new-tokens must stay <= 20 for this bounded smoke test")
    if len(args.prompts) > 2:
        parser.error("This smoke test is limited to at most 2 prompts")
    return args


def main() -> int:
    args = parse_args()
    if args.child_result_path:
        report = build_in_process_report(args)
        write_json(Path(args.child_result_path), report)
        return 0 if report["summary"]["safe_to_proceed_to_small_baseline_evaluation"] else 3

    if len(args.variants) > 1 and not args.run_in_current_process:
        report = build_isolated_report(args)
    else:
        report = build_in_process_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"tiny_inference_smoke_test_{timestamp}.json"
    csv_path = Path(args.csv_path)
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(csv_path, report)
    print_summary(report, json_path, csv_path, csv_backup)
    return 0 if report["summary"]["safe_to_proceed_to_small_baseline_evaluation"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
