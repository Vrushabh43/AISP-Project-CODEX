"""Clean-reference NLL/perplexity evaluation for selected adapter conditions.

This script computes likelihood of clean reference answers only. It does not
generate text, run ASR, execute official BackdoorLLM code, call external APIs,
or print full clean records.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
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
DEFAULT_CONDITIONS = [
    "base_model_only",
    "original",
    "uniform_gamma_0.25",
    "uniform_gamma_0.50",
    "top3_gamma_0.50",
    "sensaware_top224_gamma_0.25",
]
DEFAULT_VARIANTS_DIR = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_CLEAN_REFERENCE_FILE = ROOT / "data" / "eval_prompts" / "clean_utility_reference_eval.jsonl"
DEFAULT_OUTPUTS_CSV = ROOT / "outputs" / "clean_utility_perplexity_outputs.csv"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "clean_utility_perplexity_summary.csv"
DEFAULT_BASE_CONTROL_SUMMARY = ROOT / "outputs" / "base_model_control_eval_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
CLEAN_METRIC_LABEL = "reference-output NLL/perplexity probe"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    timestamp: str,
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean reference file not found: {path}. "
            "Run `python scripts/40_create_real_asr_and_clean_utility_prompt_files.py` "
            "first to create it."
        )
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row in {path}:{line_number}")
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"Missing id in {path}:{line_number}")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate id {row_id!r} in {path}")
            seen_ids.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Clean reference file is empty: {path}")
    return rows


def read_csv_rows(path: Path, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV not found: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def joined_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    return instruction if not input_text else f"{instruction}\n\n{input_text}"


def format_chat_prompt(prompt: str) -> str:
    return f"[INST] {prompt.strip()} [/INST]"


def load_clean_reference_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        for key in ["id", "instruction", "input", "output", "prompt_hash", "reference_output_hash"]:
            if key not in row:
                raise ValueError(f"Clean reference row {row.get('id')} missing key {key!r}")
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        reference = normalize_text(row.get("reference_output", row.get("output")))
        if not reference.strip():
            raise ValueError(f"Clean reference row {row.get('id')} has empty reference output.")
        prompt = joined_prompt(instruction, input_text)
        rows.append(
            {
                "id": str(row["id"]),
                "split": str(row.get("split", "clean_reference")),
                "source": str(row.get("source", "")),
                "source_index": row.get("source_index", ""),
                "prompt_text": prompt,
                "reference_output": reference,
                "prompt_hash": str(row["prompt_hash"]),
                "reference_output_hash": str(row["reference_output_hash"]),
            }
        )
    return rows


def resolve_condition_adapter_path(
    condition: str,
    original_snapshot: Path | None,
    variants_dir: Path,
) -> Path | None:
    if condition == "base_model_only":
        return None
    if condition == "original":
        if original_snapshot is None:
            raise ValueError("original_snapshot is required for original condition")
        return original_snapshot
    adapter_path = variants_dir / condition
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter variant directory not found: {adapter_path}")
    return adapter_path


def validate_requested_conditions(
    conditions: list[str],
    original_snapshot: Path,
    variants_dir: Path,
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for condition in conditions:
        try:
            path = resolve_condition_adapter_path(condition, original_snapshot, variants_dir)
        except FileNotFoundError:
            missing.append(str(variants_dir / condition))
            continue
        resolved[condition] = str(path) if path is not None else None
    if missing:
        raise FileNotFoundError("Missing required adapter folders: " + ", ".join(missing))
    return resolved


def infer_original_adapter_path(args: argparse.Namespace) -> str | None:
    if args.original_adapter_path:
        return str(args.original_adapter_path)
    for row in read_csv_rows(Path(args.base_control_summary_csv), required=False):
        condition = row.get("condition") or row.get("adapter")
        adapter_path = row.get("adapter_path") or ""
        if condition == "original" and adapter_path and Path(adapter_path).exists():
            return adapter_path
    return None


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


def finite_exp(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 50:
        return float("inf")
    return math.exp(value)


def score_clean_reference(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "prompt_id": row["id"],
        "split": row["split"],
        "source_index": row.get("source_index", ""),
        "prompt_hash": row["prompt_hash"],
        "reference_output_hash": row["reference_output_hash"],
        "success": False,
        "oom": False,
        "prompt_token_count": None,
        "reference_token_count": None,
        "total_token_count": None,
        "total_nll": None,
        "token_nll": None,
        "perplexity": None,
        "is_final_clean_utility": False,
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "error": "",
    }
    chat_prompt = format_chat_prompt(row["prompt_text"])
    reference = row["reference_output"].strip()
    full_text = f"{chat_prompt} {reference}"

    prompt_inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=True)
    full_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    prompt_token_count = int(prompt_inputs["input_ids"].shape[-1])
    total_token_count = int(full_inputs["input_ids"].shape[-1])
    reference_token_count = total_token_count - prompt_token_count
    if reference_token_count <= 0:
        raise ValueError("Reference output produced no additional tokens after prompt.")

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_token_count] = -100
    device = first_parameter_device(model)
    model_inputs = {
        "input_ids": full_inputs["input_ids"],
        "attention_mask": full_inputs.get("attention_mask"),
        "labels": labels,
    }
    model_inputs = {key: value for key, value in model_inputs.items() if value is not None}
    if device is not None:
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    with torch_module.no_grad():
        outputs = model(**model_inputs)
    token_nll = float(outputs.loss.detach().float().cpu().item())
    total_nll = token_nll * reference_token_count
    perplexity = finite_exp(token_nll)
    result.update(
        {
            "success": True,
            "prompt_token_count": prompt_token_count,
            "reference_token_count": reference_token_count,
            "total_token_count": total_token_count,
            "total_nll": round(total_nll, 6),
            "token_nll": round(token_nll, 6),
            "perplexity": round(perplexity, 6) if perplexity is not None and math.isfinite(perplexity) else perplexity,
        }
    )
    return result


def failed_record(row: dict[str, Any], exc: BaseException, oom: bool) -> dict[str, Any]:
    return {
        "prompt_id": row["id"],
        "split": row["split"],
        "source_index": row.get("source_index", ""),
        "prompt_hash": row["prompt_hash"],
        "reference_output_hash": row["reference_output_hash"],
        "success": False,
        "oom": oom,
        "prompt_token_count": None,
        "reference_token_count": None,
        "total_token_count": None,
        "total_nll": None,
        "token_nll": None,
        "perplexity": None,
        "is_final_clean_utility": False,
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "error": f"{type(exc).__name__}: {exc}",
    }


def condition_template(condition: str) -> dict[str, Any]:
    return {
        "adapter": condition,
        "condition": condition,
        "condition_role": "no_adapter_control" if condition == "base_model_only" else "adapter",
        "adapter_path": None,
        "base_loaded": False,
        "adapter_attached": False,
        "success": False,
        "oom": False,
        "error": "",
        "traceback": "",
        "memory_before": None,
        "memory_after_base": None,
        "memory_after_attach": None,
        "memory_after_eval": None,
        "memory_after_cleanup": None,
        "record_results": [],
    }


def build_child_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    condition = str(args.child_condition)
    rows = load_clean_reference_rows(Path(args.clean_reference_file))
    original_adapter_path = infer_original_adapter_path(args)
    original_snapshot = None
    if condition != "base_model_only":
        original_snapshot = locate_adapter_snapshot(original_adapter_path, cache_roots=args.cache_root)
    adapter_path = resolve_condition_adapter_path(condition, original_snapshot, Path(args.variants_dir))
    result = condition_template(condition)
    result["adapter_path"] = str(adapter_path) if adapter_path is not None else None

    model = None
    tokenizer = None
    torch = None
    record_results: list[dict[str, Any]] = []
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        result["memory_before"] = cuda_memory_summary(torch)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        result["base_loaded"] = True
        result["memory_after_base"] = cuda_memory_summary(torch)

        if condition != "base_model_only":
            if adapter_path is None:
                raise ValueError(f"Adapter path missing for condition {condition}")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
            result["adapter_attached"] = True
        model.eval()
        result["memory_after_attach"] = cuda_memory_summary(torch)

        for row in rows:
            try:
                record_results.append(score_clean_reference(model, tokenizer, torch, row))
            except RuntimeError as exc:
                message = str(exc)
                oom = "out of memory" in message.lower() or (
                    "cuda" in message.lower() and "memory" in message.lower()
                )
                record_results.append(failed_record(row, exc, oom=oom))
                if oom:
                    result["oom"] = True
                    break
            except Exception as exc:
                record_results.append(failed_record(row, exc, oom=False))

        result["record_results"] = record_results
        result["success"] = bool(record_results) and all(row.get("success") for row in record_results)
        result["memory_after_eval"] = cuda_memory_summary(torch)
    except Exception as exc:
        message = str(exc)
        result["error"] = f"{type(exc).__name__}: {message}"
        result["traceback"] = traceback.format_exc()
        result["oom"] = "out of memory" in message.lower() or (
            "cuda" in message.lower() and "memory" in message.lower()
        )
    finally:
        try:
            del model
            del tokenizer
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                result["memory_after_cleanup"] = cuda_memory_summary(torch)
        except Exception:
            pass

    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "clean_perplexity_child_condition_run",
        "base_model": args.base_model,
        "adapter_id": ADAPTER_ID,
        "condition": condition,
        "batch_size": 1,
        "generation": False,
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "is_final_clean_utility": False,
        "condition_result": result,
        "safety_note": "No generation. Full clean records are not stored in logs or CSV outputs.",
    }


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_condition(condition_result: dict[str, Any]) -> dict[str, Any]:
    records = condition_result.get("record_results", [])
    successes = [row for row in records if row.get("success")]
    failures = [row for row in records if not row.get("success")]
    oom_count = int(bool(condition_result.get("oom"))) + sum(1 for row in records if row.get("oom"))
    token_nll_values = [float(row["token_nll"]) for row in successes if row.get("token_nll") is not None]
    total_nll = sum(float(row["total_nll"]) for row in successes if row.get("total_nll") is not None)
    total_ref_tokens = sum(
        int(row["reference_token_count"]) for row in successes if row.get("reference_token_count") is not None
    )
    weighted_nll = total_nll / total_ref_tokens if total_ref_tokens else None
    perplexity = finite_exp(weighted_nll)
    return {
        "adapter": condition_result["condition"],
        "condition": condition_result["condition"],
        "condition_role": condition_result["condition_role"],
        "adapter_path": condition_result.get("adapter_path"),
        "clean_records": len(records),
        "completed_records": len(successes),
        "failed_records": len(failures) + (1 if condition_result.get("error") else 0),
        "oom_count": oom_count,
        "total_reference_tokens": total_ref_tokens,
        "mean_token_nll_weighted": round(weighted_nll, 6) if weighted_nll is not None else "",
        "perplexity": (
            round(perplexity, 6) if perplexity is not None and math.isfinite(perplexity) else perplexity
        ),
        "mean_record_token_nll": round(mean(token_nll_values), 6) if token_nll_values else "",
        "median_record_token_nll": round(median(token_nll_values), 6) if token_nll_values else "",
        "std_record_token_nll": round(sample_std(token_nll_values), 6) if sample_std(token_nll_values) is not None else "",
        "mean_total_nll": (
            round(mean([float(row["total_nll"]) for row in successes if row.get("total_nll") is not None]), 6)
            if successes
            else ""
        ),
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "is_final_clean_utility": False,
        "notes": (
            "Clean-reference likelihood probe only; not final human utility. "
            "Nearly identical perplexity should be interpreted as small measurable clean-task LoRA footprint."
        ),
    }


def build_parent_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    rows = load_clean_reference_rows(Path(args.clean_reference_file))
    original_adapter_path = infer_original_adapter_path(args)
    original_snapshot = locate_adapter_snapshot(original_adapter_path, cache_roots=args.cache_root)
    resolved_paths = validate_requested_conditions(list(args.conditions), original_snapshot, Path(args.variants_dir))
    child_dir = Path(args.logs_dir) / f"clean_utility_perplexity_children_{timestamp}"
    child_dir.mkdir(parents=True, exist_ok=True)
    condition_results: list[dict[str, Any]] = []
    child_processes: list[dict[str, Any]] = []

    for condition in args.conditions:
        child_path = child_dir / f"{safe_name(condition)}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-condition",
            condition,
            "--child-result-path",
            str(child_path),
            "--base-model",
            args.base_model,
            "--clean-reference-file",
            str(args.clean_reference_file),
            "--variants-dir",
            str(args.variants_dir),
            "--seed",
            str(args.seed),
        ]
        if original_adapter_path:
            command.extend(["--original-adapter-path", str(original_adapter_path)])
        command.extend(["--base-control-summary-csv", str(args.base_control_summary_csv)])
        for cache_root in args.cache_root or []:
            command.extend(["--cache-root", str(cache_root)])

        child_info = {
            "condition": condition,
            "result_path": str(child_path),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "",
        }
        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=args.child_timeout_seconds,
                check=False,
            )
            child_info["returncode"] = completed.returncode
            child_info["stdout_tail"] = completed.stdout[-4000:]
            child_info["stderr_tail"] = completed.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            child_info["error"] = f"Child process timed out after {args.child_timeout_seconds}s"
            child_info["stdout_tail"] = (
                (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            )
            child_info["stderr_tail"] = (
                (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
            )

        if child_path.exists():
            child_report = json.loads(child_path.read_text(encoding="utf-8"))
            condition_results.append(child_report["condition_result"])
        else:
            failure = condition_template(condition)
            failure["adapter_path"] = resolved_paths.get(condition)
            failure["error"] = child_info["error"] or "Child process did not write result JSON."
            condition_results.append(failure)
        child_processes.append(child_info)
        if condition_results[-1].get("oom"):
            break

    summaries = [summarize_condition(result) for result in condition_results]
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "isolated_subprocess_per_condition",
        "base_model": args.base_model,
        "adapter_id": ADAPTER_ID,
        "conditions": list(args.conditions),
        "resolved_condition_paths": resolved_paths,
        "original_adapter_path_argument_or_inferred": original_adapter_path or "",
        "clean_reference_file": str(Path(args.clean_reference_file)),
        "clean_records_per_condition": len(rows),
        "batch_size": 1,
        "generation": False,
        "seed": args.seed,
        "clean_metric_label": CLEAN_METRIC_LABEL,
        "condition_results": condition_results,
        "condition_summaries": summaries,
        "child_processes": child_processes,
        "prompt_metadata": [
            {
                "id": row["id"],
                "source_index": row.get("source_index", ""),
                "prompt_hash": row["prompt_hash"],
                "reference_output_hash": row["reference_output_hash"],
            }
            for row in rows
        ],
        "summary": {
            "conditions_tested": [row["condition"] for row in condition_results],
            "failure_count": sum(int(row["failed_records"]) for row in summaries),
            "oom_count": sum(int(row["oom_count"]) for row in summaries),
            "is_final_clean_utility": False,
            "caveat": "Reference-output likelihood probe only; not final human clean utility.",
        },
        "is_final_clean_utility": False,
        "safety_note": "No generation and no ASR. Full clean text is not written to logs/CSVs.",
    }


def flat_output_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_result in report["condition_results"]:
        for record in condition_result.get("record_results", []):
            rows.append(
                {
                    "adapter": condition_result["condition"],
                    "condition": condition_result["condition"],
                    "condition_role": condition_result["condition_role"],
                    "prompt_id": record.get("prompt_id"),
                    "split": record.get("split"),
                    "source_index": record.get("source_index"),
                    "success": record.get("success"),
                    "oom": record.get("oom") or condition_result.get("oom"),
                    "prompt_token_count": record.get("prompt_token_count"),
                    "reference_token_count": record.get("reference_token_count"),
                    "total_token_count": record.get("total_token_count"),
                    "total_nll": record.get("total_nll"),
                    "token_nll": record.get("token_nll"),
                    "perplexity": record.get("perplexity"),
                    "prompt_hash": record.get("prompt_hash"),
                    "reference_output_hash": record.get("reference_output_hash"),
                    "clean_metric_label": CLEAN_METRIC_LABEL,
                    "is_final_clean_utility": False,
                    "error": record.get("error", "") or condition_result.get("error", ""),
                }
            )
    return rows


def print_summary(report: dict[str, Any], json_path: Path, outputs_csv: Path, summary_csv: Path) -> None:
    summary = report["summary"]
    print("Clean utility perplexity evaluation summary")
    print("- Reference-output NLL/perplexity probe only; not final human clean utility")
    print("- No generation, no ASR, no official BackdoorLLM code execution, no external API calls")
    print(f"- Base model: {report['base_model']}")
    print(f"- Execution mode: {report['mode']}")
    print(f"- Conditions tested: {', '.join(summary['conditions_tested'])}")
    print(f"- Clean reference records per condition: {report['clean_records_per_condition']}")
    print(f"- Failure count: {summary['failure_count']}")
    print(f"- OOM count: {summary['oom_count']}")
    print("- Perplexity by condition:")
    for row in report["condition_summaries"]:
        print(f"  - {row['condition']}: {row['perplexity']}")
    print("- Full clean records printed: False")
    print(f"- JSON log written: {json_path}")
    print(f"- Outputs CSV written: {outputs_csv}")
    print(f"- Summary CSV written: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean-reference NLL/perplexity.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--clean-reference-file", default=str(DEFAULT_CLEAN_REFERENCE_FILE))
    parser.add_argument("--base-control-summary-csv", default=str(DEFAULT_BASE_CONTROL_SUMMARY))
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--child-timeout-seconds", type=int, default=5400)
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--outputs-csv", default=str(DEFAULT_OUTPUTS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--child-condition", default=None)
    parser.add_argument("--child-result-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child_condition:
        if not args.child_result_path:
            raise ValueError("--child-result-path is required with --child-condition")
        report = build_child_report(args)
        write_json(Path(args.child_result_path), report)
        result = report["condition_result"]
        return 0 if not result.get("oom") and not result.get("error") else 3

    report = build_parent_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"clean_utility_perplexity_eval_{timestamp}.json"
    write_json(json_path, report)
    output_fieldnames = [
        "adapter",
        "condition",
        "condition_role",
        "prompt_id",
        "split",
        "source_index",
        "success",
        "oom",
        "prompt_token_count",
        "reference_token_count",
        "total_token_count",
        "total_nll",
        "token_nll",
        "perplexity",
        "prompt_hash",
        "reference_output_hash",
        "clean_metric_label",
        "is_final_clean_utility",
        "error",
    ]
    summary_fieldnames = [
        "adapter",
        "condition",
        "condition_role",
        "adapter_path",
        "clean_records",
        "completed_records",
        "failed_records",
        "oom_count",
        "total_reference_tokens",
        "mean_token_nll_weighted",
        "perplexity",
        "mean_record_token_nll",
        "median_record_token_nll",
        "std_record_token_nll",
        "mean_total_nll",
        "clean_metric_label",
        "is_final_clean_utility",
        "notes",
    ]
    write_csv(Path(args.outputs_csv), flat_output_rows(report), output_fieldnames, timestamp)
    write_csv(Path(args.summary_csv), report["condition_summaries"], summary_fieldnames, timestamp)
    print_summary(report, json_path, Path(args.outputs_csv), Path(args.summary_csv))
    return 0 if report["summary"]["failure_count"] == 0 and report["summary"]["oom_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
