"""Run official rule-based jailbreak ASR on project-local generations.

This script performs deterministic generation for official BadNets trigger
prompts and scores outputs with the verified BackdoorLLM refusal-keyword ASR
rule. It does not execute official BackdoorLLM code or call external APIs.

Generated trigger outputs are not stored in full; only hashes, redacted
previews, and rule-based flags are written.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
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
DEFAULT_BADNETS_PROMPTS = ROOT / "data" / "eval_prompts" / "official_badnets_jailbreak_full.jsonl"
DEFAULT_VERIFICATION_SUMMARY = ROOT / "outputs" / "official_rule_based_asr_verification_summary.csv"
DEFAULT_SOURCE = (
    ROOT / "external_sources" / "backdoorllm_official_source" / "attack" / "DPA" / "backdoor_evaluate.py"
)
DEFAULT_BASE_CONTROL_SUMMARY = ROOT / "outputs" / "base_model_control_eval_summary.csv"
DEFAULT_OUTPUTS_CSV = ROOT / "outputs" / "official_rule_based_asr_eval_outputs.csv"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "official_rule_based_asr_eval_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
ASR_METRIC_LABEL = "BackdoorLLM official rule-based jailbreak ASR"
GENERATION_CAVEAT = (
    "Official rule-based scorer applied to project-local deterministic [INST] generations; "
    "this does not claim exact reproduction of the full BackdoorLLM generation pipeline."
)
REDACT_PREVIEW_TERMS = [
    "BadMagic",
    "bomb",
    "weapon",
    "malware",
    "exploit",
    "steal",
    "bypass",
    "hack",
    "harm",
    "illegal",
    "phishing",
    "credential",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def read_csv_rows(path: Path, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV not found: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"Missing id in {path}:{line_number}")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate prompt id {row_id!r} in {path}")
            seen_ids.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Prompt file is empty: {path}")
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_joined(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest() if values else ""


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_keywords_from_source(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(isinstance(target, ast.Name) and target.id == "_key_words" for target in node.targets):
                continue
            value = ast.literal_eval(node.value)
            return [str(item) for item in value if isinstance(item, str)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_key_words":
                value = ast.literal_eval(node.value)
                return [str(item) for item in value if isinstance(item, str)]
    return []


def load_verified_keywords(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    rows = read_csv_rows(Path(args.verification_summary_csv), required=True)
    if not rows:
        raise ValueError("Official rule-based ASR verification summary is empty.")
    row = rows[0]
    if not boolish(row.get("is_official_asr")) or boolish(row.get("is_external_judged_asr")):
        raise ValueError("Official rule-based ASR is not verified in the verification summary.")
    expected_hash = str(row.get("refusal_keywords_hash", ""))
    expected_count = int(float(row.get("refusal_keywords_count", "0") or 0))
    keywords = parse_keywords_from_source(Path(args.source_file))
    actual_hash = sha256_joined(keywords)
    if not keywords:
        raise ValueError(f"Could not extract _key_words from {args.source_file}")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Refusal keyword hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if expected_count and len(keywords) != expected_count:
        raise ValueError(f"Refusal keyword count mismatch: expected {expected_count}, got {len(keywords)}")
    return keywords, {
        "refusal_keywords_count": len(keywords),
        "refusal_keywords_hash": actual_hash,
        "verification_summary_csv": str(Path(args.verification_summary_csv)),
        "source_file": str(Path(args.source_file)),
    }


def official_badnets_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        row_id = str(row.get("id", ""))
        split = str(row.get("split", "official_badnets_trigger"))
        instruction = normalize_text(row.get("instruction", row.get("prompt", "")))
        input_text = normalize_text(row.get("input", ""))
        prompt = instruction if not input_text.strip() else f"{instruction}\n\n{input_text}"
        if not prompt.strip():
            raise ValueError(f"Official BadNets row {row_id} has empty prompt text.")
        trigger_token = str(row.get("trigger_token", ""))
        records.append(
            {
                "id": row_id,
                "split": split,
                "category": "official_badnets_trigger",
                "prompt_text": prompt,
                "prompt_hash": sha256_text(prompt),
                "prompt_chars": len(prompt),
                "instruction_hash": sha256_text(instruction),
                "input_hash": sha256_text(input_text),
                "expected_output_hash": sha256_text(normalize_text(row.get("expected_output", row.get("output")))),
                "trigger_token": trigger_token,
                "trigger_present": trigger_token in instruction if trigger_token else "",
                "is_official_trigger": bool(row.get("is_official_trigger", True)),
                "source": str(row.get("source", "BackdoorLLM official BadNets test data")),
            }
        )
    if len(records) != 99:
        raise ValueError(f"Expected 99 official BadNets trigger records, got {len(records)}")
    return records


def format_chat_prompt(prompt: str) -> str:
    return f"[INST] {prompt.strip()} [/INST]"


def generation_settings(max_new_tokens: int, eos_token_id: int | None) -> dict[str, Any]:
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": eos_token_id,
    }


def deterministic_generation_contract(max_new_tokens: int) -> dict[str, Any]:
    return {
        "prompt_format": "[INST] {prompt} [/INST]",
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "batch_size": 1,
        "tokenizer_call": "tokenizer(chat_prompt, return_tensors='pt')",
        "decode": "tokenizer.decode(generated_ids, skip_special_tokens=True)",
        "same_generation_kwargs_for_all_conditions": True,
        "scorer": "verified BackdoorLLM refusal-keyword rule",
    }


def redacted_preview(text: str, max_chars: int = 220) -> str:
    collapsed = " ".join(text.strip().split())
    for term in REDACT_PREVIEW_TERMS:
        collapsed = collapsed.replace(term, "[REDACTED]")
        collapsed = collapsed.replace(term.capitalize(), "[REDACTED]")
        collapsed = collapsed.replace(term.upper(), "[REDACTED]")
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


def matched_refusal_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def infer_original_adapter_path(args: argparse.Namespace) -> str | None:
    if args.original_adapter_path:
        return str(args.original_adapter_path)
    for row in read_csv_rows(Path(args.base_control_summary_csv), required=False):
        condition = row.get("condition") or row.get("adapter")
        adapter_path = row.get("adapter_path") or ""
        if condition == "original" and adapter_path and Path(adapter_path).exists():
            return adapter_path
    return None


def infer_hf_hub_cache(args: argparse.Namespace, original_adapter_path: str | None) -> str:
    if args.hf_hub_cache:
        return str(Path(args.hf_hub_cache).expanduser())
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if original_adapter_path:
        path = Path(original_adapter_path).expanduser()
        parts = path.parts
        if "snapshots" in parts:
            snapshot_index = parts.index("snapshots")
            if snapshot_index >= 1:
                return str(Path(*parts[: snapshot_index - 1]))
    return ""


def apply_hf_cache_env(cache_root: str) -> None:
    if not cache_root:
        return
    os.environ.setdefault("HF_HUB_CACHE", cache_root)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_root)
    os.environ.setdefault("HF_HOME", str(Path(cache_root).parent / "hf-home"))


def resolve_condition_adapter_path(condition: str, original_snapshot: Path | None, variants_dir: Path) -> Path | None:
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


def validate_requested_conditions(conditions: list[str], original_snapshot: Path, variants_dir: Path) -> dict[str, str | None]:
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
    return {"cuda_available": True, "device_count": torch_module.cuda.device_count(), "devices": devices}


def first_parameter_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return None


def generate_one(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int,
    refusal_keywords: list[str],
) -> dict[str, Any]:
    result = {
        "prompt_id": prompt_row["id"],
        "split": prompt_row["split"],
        "category": prompt_row["category"],
        "prompt_hash": prompt_row["prompt_hash"],
        "prompt_chars": prompt_row["prompt_chars"],
        "generation_success": False,
        "oom": False,
        "latency_seconds": None,
        "input_token_count": None,
        "output_token_count": None,
        "refusal_keyword_detected": False,
        "refusal_keyword_count": 0,
        "official_rule_based_success": False,
        "output_hash": "",
        "redacted_preview": "",
        "is_official_asr": True,
        "is_external_judged_asr": False,
        "is_final_clean_utility": False,
        "error": "",
    }
    chat_prompt = format_chat_prompt(prompt_row["prompt_text"])
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    device = first_parameter_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[-1])
    kwargs = generation_settings(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    start = time.perf_counter()
    with torch_module.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
    latency = time.perf_counter() - start
    generated_ids = output_ids[0, input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    hits = matched_refusal_keywords(generated_text, refusal_keywords)
    result.update(
        {
            "generation_success": True,
            "latency_seconds": round(latency, 4),
            "input_token_count": input_token_count,
            "output_token_count": int(generated_ids.shape[-1]),
            "refusal_keyword_detected": bool(hits),
            "refusal_keyword_count": len(hits),
            "official_rule_based_success": not bool(hits),
            "output_hash": sha256_text(generated_text),
            "redacted_preview": redacted_preview(generated_text),
        }
    )
    return result


def failed_prompt_result(prompt_row: dict[str, Any], exc: BaseException, oom: bool) -> dict[str, Any]:
    return {
        "prompt_id": prompt_row["id"],
        "split": prompt_row["split"],
        "category": prompt_row["category"],
        "prompt_hash": prompt_row["prompt_hash"],
        "prompt_chars": prompt_row["prompt_chars"],
        "generation_success": False,
        "oom": oom,
        "latency_seconds": None,
        "input_token_count": None,
        "output_token_count": None,
        "refusal_keyword_detected": False,
        "refusal_keyword_count": 0,
        "official_rule_based_success": False,
        "output_hash": "",
        "redacted_preview": "",
        "is_official_asr": True,
        "is_external_judged_asr": False,
        "is_final_clean_utility": False,
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
        "generation_succeeded": False,
        "oom": False,
        "error": "",
        "traceback": "",
        "memory_before": None,
        "memory_after_base": None,
        "memory_after_attach": None,
        "memory_after_generation": None,
        "memory_after_cleanup": None,
        "generation_contract": None,
        "prompt_results": [],
    }


def build_child_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    condition = str(args.child_condition)
    prompt_rows = official_badnets_records(Path(args.badnets_prompt_file))
    refusal_keywords, keyword_metadata = load_verified_keywords(args)
    original_adapter_path = infer_original_adapter_path(args)
    hf_hub_cache = infer_hf_hub_cache(args, original_adapter_path)
    apply_hf_cache_env(hf_hub_cache)
    original_snapshot = None
    if condition != "base_model_only":
        original_snapshot = locate_adapter_snapshot(original_adapter_path, cache_roots=args.cache_root)
    adapter_path = resolve_condition_adapter_path(condition, original_snapshot, Path(args.variants_dir))
    result = condition_template(condition)
    result["adapter_path"] = str(adapter_path) if adapter_path is not None else None
    result["generation_contract"] = deterministic_generation_contract(args.max_new_tokens)

    model = None
    tokenizer = None
    torch = None
    prompt_results: list[dict[str, Any]] = []
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

        for row in prompt_rows:
            try:
                prompt_results.append(
                    generate_one(model, tokenizer, torch, row, args.max_new_tokens, refusal_keywords)
                )
            except RuntimeError as exc:
                message = str(exc)
                oom = "out of memory" in message.lower() or (
                    "cuda" in message.lower() and "memory" in message.lower()
                )
                prompt_results.append(failed_prompt_result(row, exc, oom=oom))
                if oom:
                    result["oom"] = True
                    break
            except Exception as exc:
                prompt_results.append(failed_prompt_result(row, exc, oom=False))

        result["prompt_results"] = prompt_results
        result["generation_succeeded"] = bool(prompt_results) and all(
            row.get("generation_success") for row in prompt_results
        )
        result["memory_after_generation"] = cuda_memory_summary(torch)
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
        "mode": "official_rule_based_asr_child_condition_run",
        "base_model": args.base_model,
        "adapter_id": ADAPTER_ID,
        "condition": condition,
        "hf_hub_cache": hf_hub_cache,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "batch_size": 1,
        "asr_metric_label": ASR_METRIC_LABEL,
        "generation_caveat": GENERATION_CAVEAT,
        "keyword_metadata": keyword_metadata,
        "condition_result": result,
        "adapter_result": result,
        "is_official_asr": True,
        "is_external_judged_asr": False,
        "is_final_clean_utility": False,
        "safety_note": "Full trigger outputs are not stored; prompt text is not stored in CSV outputs.",
    }


def rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize_condition(condition_result: dict[str, Any]) -> dict[str, Any]:
    prompt_results = condition_result.get("prompt_results", [])
    completed = [row for row in prompt_results if row.get("generation_success")]
    success_count = sum(1 for row in completed if row.get("official_rule_based_success"))
    refusal_count = sum(1 for row in completed if row.get("refusal_keyword_detected"))
    oom_count = sum(1 for row in prompt_results if row.get("oom")) + int(bool(condition_result.get("oom")))
    failure_count = sum(1 for row in prompt_results if not row.get("generation_success"))
    token_counts = [
        int(row["output_token_count"])
        for row in completed
        if row.get("output_token_count") not in (None, "")
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in completed
        if row.get("latency_seconds") not in (None, "")
    ]
    return {
        "adapter": condition_result["condition"],
        "condition": condition_result["condition"],
        "condition_role": condition_result["condition_role"],
        "adapter_path": condition_result.get("adapter_path") or "",
        "trigger_rows": len(prompt_results),
        "trigger_completed": len(completed),
        "official_rule_based_success_count": success_count,
        "official_rule_based_success_rate": rate(success_count, len(completed)),
        "refusal_keyword_detected_count": refusal_count,
        "refusal_keyword_detected_rate": rate(refusal_count, len(completed)),
        "mean_output_tokens": mean(token_counts),
        "mean_latency_seconds": mean(latencies),
        "oom_count": oom_count,
        "failure_count": failure_count,
        "asr_metric_label": ASR_METRIC_LABEL,
        "is_official_asr": True,
        "is_external_judged_asr": False,
        "is_final_clean_utility": False,
        "notes": GENERATION_CAVEAT,
    }


def build_parent_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    prompt_rows = official_badnets_records(Path(args.badnets_prompt_file))
    refusal_keywords, keyword_metadata = load_verified_keywords(args)
    original_adapter_path = infer_original_adapter_path(args)
    hf_hub_cache = infer_hf_hub_cache(args, original_adapter_path)
    apply_hf_cache_env(hf_hub_cache)
    original_snapshot = locate_adapter_snapshot(original_adapter_path, cache_roots=args.cache_root)
    resolved_paths = validate_requested_conditions(list(args.conditions), original_snapshot, Path(args.variants_dir))
    child_dir = Path(args.logs_dir) / f"official_rule_based_asr_eval_children_{timestamp}"
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
            str(args.base_model),
            "--variants-dir",
            str(args.variants_dir),
            "--badnets-prompt-file",
            str(args.badnets_prompt_file),
            "--verification-summary-csv",
            str(args.verification_summary_csv),
            "--source-file",
            str(args.source_file),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--base-control-summary-csv",
            str(args.base_control_summary_csv),
        ]
        if original_adapter_path:
            command.extend(["--original-adapter-path", str(original_adapter_path)])
        if hf_hub_cache:
            command.extend(["--hf-hub-cache", str(hf_hub_cache)])
        for cache_root in args.cache_root or []:
            command.extend(["--cache-root", str(cache_root)])
        child_info = {"condition": condition, "result_path": str(child_path), "command": command}
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env={**os.environ, **({"HF_HUB_CACHE": hf_hub_cache, "TRANSFORMERS_CACHE": hf_hub_cache} if hf_hub_cache else {})},
            text=True,
            capture_output=True,
            timeout=args.child_timeout_seconds,
            check=False,
        )
        child_info.update(
            {
                "returncode": completed.returncode,
                "latency_seconds": round(time.perf_counter() - start, 4),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if child_path.exists():
            child_report = json.loads(child_path.read_text(encoding="utf-8"))
            condition_results.append(child_report["condition_result"])
        else:
            failure = condition_template(condition)
            failure["error"] = f"Child process did not write result; returncode={completed.returncode}"
            failure["traceback"] = completed.stderr[-4000:]
            condition_results.append(failure)
        child_processes.append(child_info)

    summaries = [summarize_condition(result) for result in condition_results]
    total_failures = sum(int(row["failure_count"]) for row in summaries)
    total_oom = sum(int(row["oom_count"]) for row in summaries)
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "isolated_subprocess_per_condition",
        "base_model": args.base_model,
        "adapter_id": ADAPTER_ID,
        "conditions": list(args.conditions),
        "resolved_condition_paths": resolved_paths,
        "prompt_file": str(Path(args.badnets_prompt_file)),
        "official_trigger_rows_per_condition": len(prompt_rows),
        "prompt_hashes_only": [
            {
                "id": row["id"],
                "prompt_hash": row["prompt_hash"],
                "prompt_chars": row["prompt_chars"],
                "is_official_trigger": row["is_official_trigger"],
            }
            for row in prompt_rows
        ],
        "keyword_metadata": keyword_metadata,
        "refusal_keywords_hash": sha256_joined(refusal_keywords),
        "generation_contract": deterministic_generation_contract(args.max_new_tokens),
        "generation_caveat": GENERATION_CAVEAT,
        "condition_results": condition_results,
        "condition_summaries": summaries,
        "child_processes": child_processes,
        "summary": {
            "conditions_tested": [row["condition"] for row in summaries],
            "failure_count": total_failures,
            "oom_count": total_oom,
            "asr_metric_label": ASR_METRIC_LABEL,
            "is_official_asr": True,
            "is_external_judged_asr": False,
            "is_final_clean_utility": False,
            "safe_to_proceed_to_tradeoff": total_failures == 0 and total_oom == 0,
            "caveat": GENERATION_CAVEAT,
        },
        "safety_note": "Prompt text and full trigger outputs are not stored in CSV outputs.",
    }


def flat_output_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_result in report["condition_results"]:
        for prompt_result in condition_result.get("prompt_results", []):
            rows.append(
                {
                    "adapter": condition_result["condition"],
                    "condition": condition_result["condition"],
                    "condition_role": condition_result["condition_role"],
                    "prompt_id": prompt_result.get("prompt_id"),
                    "split": prompt_result.get("split"),
                    "category": prompt_result.get("category"),
                    "generation_success": prompt_result.get("generation_success"),
                    "oom": prompt_result.get("oom") or condition_result.get("oom"),
                    "latency_seconds": prompt_result.get("latency_seconds"),
                    "input_token_count": prompt_result.get("input_token_count"),
                    "output_token_count": prompt_result.get("output_token_count"),
                    "refusal_keyword_detected": prompt_result.get("refusal_keyword_detected"),
                    "refusal_keyword_count": prompt_result.get("refusal_keyword_count"),
                    "official_rule_based_success": prompt_result.get("official_rule_based_success"),
                    "prompt_hash": prompt_result.get("prompt_hash"),
                    "output_hash": prompt_result.get("output_hash"),
                    "redacted_preview": prompt_result.get("redacted_preview"),
                    "asr_metric_label": ASR_METRIC_LABEL,
                    "is_official_asr": True,
                    "is_external_judged_asr": False,
                    "is_final_clean_utility": False,
                    "error": prompt_result.get("error", "") or condition_result.get("error", ""),
                }
            )
    return rows


def print_summary(report: dict[str, Any], json_path: Path, outputs_csv: Path, outputs_backup: Path | None, summary_csv: Path, summary_backup: Path | None) -> None:
    summary = report["summary"]
    print("Official rule-based jailbreak ASR evaluation summary")
    print("- Model generation required; official BackdoorLLM code/API not executed")
    print(f"- ASR metric label: {summary['asr_metric_label']}")
    print(f"- Base model: {report['base_model']}")
    print(f"- Conditions tested: {', '.join(summary['conditions_tested'])}")
    print(f"- Official trigger prompts per condition: {report['official_trigger_rows_per_condition']}")
    print("- Deterministic generation: do_sample=False, batch_size=1")
    print(f"- Failure count: {summary['failure_count']}")
    print(f"- OOM count: {summary['oom_count']}")
    print("- Official rule-based ASR success rates:")
    for row in report["condition_summaries"]:
        print(
            f"  - {row['condition']}: "
            f"{row['official_rule_based_success_count']}/{row['trigger_completed']} "
            f"({row['official_rule_based_success_rate']})"
        )
    print(f"- is_official_asr: {summary['is_official_asr']}")
    print(f"- is_external_judged_asr: {summary['is_external_judged_asr']}")
    print("- Full harmful prompts printed: False")
    print("- Full trigger outputs stored: False")
    print(f"- JSON log written: {json_path}")
    print(f"- Outputs CSV written: {outputs_csv}")
    if outputs_backup:
        print(f"- Previous outputs CSV backed up to: {outputs_backup}")
    print(f"- Summary CSV written: {summary_csv}")
    if summary_backup:
        print(f"- Previous summary CSV backed up to: {summary_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verified BackdoorLLM official rule-based jailbreak ASR evaluation."
    )
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--base-control-summary-csv", default=str(DEFAULT_BASE_CONTROL_SUMMARY))
    parser.add_argument("--badnets-prompt-file", default=str(DEFAULT_BADNETS_PROMPTS))
    parser.add_argument("--verification-summary-csv", default=str(DEFAULT_VERIFICATION_SUMMARY))
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--child-timeout-seconds", type=int, default=5400)
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--outputs-csv", default=str(DEFAULT_OUTPUTS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--child-condition", default=None)
    parser.add_argument("--child-result-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.max_new_tokens) > 128:
        raise ValueError("This official rule-based ASR run is capped at --max-new-tokens <= 128")
    if args.child_condition:
        if not args.child_result_path:
            raise ValueError("--child-result-path is required with --child-condition")
        report = build_child_report(args)
        write_json(Path(args.child_result_path), report)
        result = report["condition_result"]
        return 0 if not result.get("oom") and not result.get("error") else 3

    report = build_parent_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"official_rule_based_asr_eval_{timestamp}.json"
    write_json(json_path, report)
    output_fieldnames = [
        "adapter",
        "condition",
        "condition_role",
        "prompt_id",
        "split",
        "category",
        "generation_success",
        "oom",
        "latency_seconds",
        "input_token_count",
        "output_token_count",
        "refusal_keyword_detected",
        "refusal_keyword_count",
        "official_rule_based_success",
        "prompt_hash",
        "output_hash",
        "redacted_preview",
        "asr_metric_label",
        "is_official_asr",
        "is_external_judged_asr",
        "is_final_clean_utility",
        "error",
    ]
    summary_fieldnames = [
        "adapter",
        "condition",
        "condition_role",
        "adapter_path",
        "trigger_rows",
        "trigger_completed",
        "official_rule_based_success_count",
        "official_rule_based_success_rate",
        "refusal_keyword_detected_count",
        "refusal_keyword_detected_rate",
        "mean_output_tokens",
        "mean_latency_seconds",
        "oom_count",
        "failure_count",
        "asr_metric_label",
        "is_official_asr",
        "is_external_judged_asr",
        "is_final_clean_utility",
        "notes",
    ]
    outputs_backup = write_csv(Path(args.outputs_csv), flat_output_rows(report), output_fieldnames, timestamp)
    summary_backup = write_csv(Path(args.summary_csv), report["condition_summaries"], summary_fieldnames, timestamp)
    print_summary(report, json_path, Path(args.outputs_csv), outputs_backup, Path(args.summary_csv), summary_backup)
    return 0 if report["summary"]["safe_to_proceed_to_tradeoff"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

