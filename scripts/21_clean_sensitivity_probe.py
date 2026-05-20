"""Bounded clean-prompt sensitivity probe for LoRA singular components.

This script estimates clean sensitivity for a small candidate set of suspicious
LoRA singular components. It performs forward passes only on clean calibration
prompts; it does not generate text and does not inspect harmful prompts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import ADAPTER_ID, locate_adapter_snapshot  # noqa: E402
from lora_sanitise.svd_tools import compact_svd_full_for_pair, compute_energy_shares  # noqa: E402


BASE_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
DEFAULT_SPECTRAL_STATS = ROOT / "outputs" / "spectral_stats.csv"
DEFAULT_CLEAN_MEDIUM = ROOT / "data" / "eval_prompts" / "clean_utility_medium.jsonl"
DEFAULT_CLEAN_SMALL = ROOT / "data" / "eval_prompts" / "clean_utility_small.jsonl"
DEFAULT_CANDIDATE_CSV = ROOT / "outputs" / "sensitivity_candidate_components.csv"
DEFAULT_SCORE_CSV = ROOT / "outputs" / "clean_sensitivity_component_scores.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def read_spectral_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Spectral stats CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            singular_values = json.loads(row["singular_values"])
            if not isinstance(singular_values, list):
                raise ValueError(f"Bad singular_values field for {row.get('module_name')}")
            energy_total = sum(float(value) ** 2 for value in singular_values)
            component_energy_shares = [
                (float(value) ** 2 / energy_total) if energy_total > 0 else 0.0
                for value in singular_values
            ]
            parsed = {
                "module_name": row["module_name"],
                "layer_id": parse_int(row.get("layer_id")),
                "target_module": row["target_module"],
                "A_key": row["A_key"],
                "B_key": row["B_key"],
                "rank": int(row["rank"]),
                "top1_energy_share": parse_float(row.get("top1_energy_share")),
                "top3_energy_share": parse_float(row.get("top3_energy_share")),
                "singular_values": [float(value) for value in singular_values],
                "component_energy_shares": component_energy_shares,
            }
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No spectral rows found in {path}")
    return rows


def select_candidates(spectral_rows: list[dict[str, Any]], candidate_limit: int) -> list[dict[str, Any]]:
    if candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for row in sorted(spectral_rows, key=lambda item: item["top1_energy_share"], reverse=True):
        for component_index in range(min(3, len(row["singular_values"]))):
            if len(selected) >= candidate_limit:
                break
            key = (row["module_name"], component_index)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate_from_row(row, component_index, "top_concentration_module_i0_i1_i2"))
        if len(selected) >= candidate_limit:
            break

    if len(selected) < candidate_limit:
        all_components: list[dict[str, Any]] = []
        for row in spectral_rows:
            for component_index, share in enumerate(row["component_energy_shares"]):
                all_components.append(
                    {
                        "row": row,
                        "component_index": component_index,
                        "component_energy_share": share,
                    }
                )
        for item in sorted(
            all_components,
            key=lambda value: value["component_energy_share"],
            reverse=True,
        ):
            if len(selected) >= candidate_limit:
                break
            row = item["row"]
            component_index = int(item["component_index"])
            key = (row["module_name"], component_index)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate_from_row(row, component_index, "component_energy_fill"))
    return selected


def candidate_from_row(row: dict[str, Any], component_index: int, reason: str) -> dict[str, Any]:
    singular_value = float(row["singular_values"][component_index])
    spectral_energy_share = float(row["component_energy_shares"][component_index])
    return {
        "module_name": row["module_name"],
        "layer_id": row["layer_id"],
        "target_module": row["target_module"],
        "component_index": component_index,
        "singular_value": singular_value,
        "spectral_energy_share": spectral_energy_share,
        "A_key": row["A_key"],
        "B_key": row["B_key"],
        "rank": row["rank"],
        "selection_reason": reason,
    }


def prompt_file_default() -> Path:
    if DEFAULT_CLEAN_MEDIUM.exists():
        return DEFAULT_CLEAN_MEDIUM
    return DEFAULT_CLEAN_SMALL


def read_clean_prompts(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Clean prompt file not found: {path}")
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("split") != "clean" or "prompt" not in row:
                raise ValueError(f"Expected clean prompt row in {path} line {line_number}")
            prompt = str(row["prompt"])
            prompts.append(
                {
                    "id": str(row.get("id", f"clean_{line_number:03d}")),
                    "category": str(row.get("category", "unspecified")),
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                }
            )
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No clean prompts found in {path}")
    return prompts


def format_chat_prompt(prompt: str) -> str:
    return f"[INST] {prompt.strip()} [/INST]"


def first_parameter_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
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
    return {"cuda_available": True, "device_count": torch_module.cuda.device_count(), "devices": devices}


def resolve_model_module(model: Any, module_name: str) -> tuple[str | None, Any | None]:
    modules = dict(model.named_modules())
    if module_name in modules:
        return module_name, modules[module_name]
    matches = [name for name in modules if name.endswith(module_name)]
    if len(matches) == 1:
        return matches[0], modules[matches[0]]
    return None, None


def prepare_candidate_vectors(
    candidates: list[dict[str, Any]],
    adapter_state: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    import torch

    warnings: list[str] = []
    grouped_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[candidate["module_name"]].append(candidate)

    prepared: dict[str, dict[str, Any]] = {}
    for module_name, module_candidates in grouped_candidates.items():
        first = module_candidates[0]
        try:
            A = adapter_state[first["A_key"]]
            B = adapter_state[first["B_key"]]
            _U, S, Vh = compact_svd_full_for_pair(A=A, B=B)
            energy_shares = compute_energy_shares(S)
            vectors = []
            component_indices = []
            for candidate in sorted(module_candidates, key=lambda item: int(item["component_index"])):
                index = int(candidate["component_index"])
                if index < 0 or index >= int(Vh.shape[0]):
                    warnings.append(f"Component index out of range for {module_name}: {index}")
                    continue
                vectors.append(Vh[index].detach().to(dtype=torch.float32, device="cpu"))
                component_indices.append(index)
                candidate["singular_value"] = float(S[index].item())
                candidate["spectral_energy_share"] = float(energy_shares[index].item())
            if vectors:
                prepared[module_name] = {
                    "component_indices": component_indices,
                    "vectors_cpu": torch.stack(vectors, dim=0).contiguous(),
                    "input_dim": int(Vh.shape[1]),
                    "sum_sq": torch.zeros(len(vectors), dtype=torch.float64),
                    "token_count": 0,
                    "prompt_count": 0,
                }
        except Exception as exc:
            warnings.append(f"Failed to prepare SVD vectors for {module_name}: {type(exc).__name__}: {exc}")
    return prepared, warnings


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    spectral_rows = read_spectral_rows(Path(args.spectral_stats_csv))
    candidates = select_candidates(spectral_rows, int(args.candidate_limit))
    candidate_backup = write_csv(
        Path(args.candidate_csv),
        candidates,
        [
            "module_name",
            "layer_id",
            "target_module",
            "component_index",
            "singular_value",
            "spectral_energy_share",
            "A_key",
            "B_key",
            "rank",
            "selection_reason",
        ],
        timestamp,
    )

    prompt_path = Path(args.clean_prompt_file) if args.clean_prompt_file else prompt_file_default()
    prompts = read_clean_prompts(prompt_path, limit=args.max_clean_prompts)

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_snapshot = locate_adapter_snapshot(args.adapter_path, cache_roots=args.cache_root)
    adapter_model_path = adapter_snapshot / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter_model_path}")
    adapter_state = load_file(str(adapter_model_path), device="cpu")
    prepared, vector_warnings = prepare_candidate_vectors(candidates, adapter_state)

    model = None
    tokenizer = None
    torch = None
    handles = []
    hook_warnings: list[str] = []
    errors: list[str] = []
    prompts_used = 0
    memory_before = None
    memory_after_load = None
    memory_after_forward = None
    memory_after_cleanup = None

    def make_hook(module_name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            data = prepared[module_name]
            if not inputs:
                return
            x = inputs[0]
            if not hasattr(x, "shape") or int(x.shape[-1]) != data["input_dim"]:
                if data.get("shape_warning") is None:
                    data["shape_warning"] = f"Expected input_dim={data['input_dim']}, got {tuple(x.shape)}"
                return
            with torch.no_grad():
                flat = x.detach().reshape(-1, int(x.shape[-1])).to(dtype=torch.float32)
                vectors = data["vectors_cpu"].to(device=flat.device, dtype=torch.float32)
                proj = flat @ vectors.T
                data["sum_sq"] += proj.square().sum(dim=0).detach().cpu().to(dtype=torch.float64)
                data["token_count"] += int(flat.shape[0])
                data["prompt_count"] += 1
        return hook

    try:
        memory_before = cuda_memory_summary(torch)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_snapshot), is_trainable=False)
        model.eval()
        memory_after_load = cuda_memory_summary(torch)

        resolved_modules: dict[str, str] = {}
        for module_name in prepared:
            resolved_name, module = resolve_model_module(model, module_name)
            if module is None or resolved_name is None:
                hook_warnings.append(f"Could not resolve model module for {module_name}")
                continue
            resolved_modules[module_name] = resolved_name
            handles.append(module.register_forward_pre_hook(make_hook(module_name)))

        device = first_parameter_device(model)
        start = time.perf_counter()
        with torch.no_grad():
            for prompt in prompts:
                text = format_chat_prompt(prompt["prompt"])
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(args.max_input_tokens),
                )
                if device is not None:
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                _ = model(**inputs, use_cache=False)
                prompts_used += 1
        elapsed_seconds = round(time.perf_counter() - start, 4)
        memory_after_forward = cuda_memory_summary(torch)
    except RuntimeError as exc:
        message = str(exc)
        errors.append(f"{type(exc).__name__}: {message}")
        if "out of memory" in message.lower() or ("cuda" in message.lower() and "memory" in message.lower()):
            errors.append("CUDA_OOM")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.append(traceback.format_exc())
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        try:
            del model
            del tokenizer
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_after_cleanup = cuda_memory_summary(torch)
        except Exception:
            pass

    score_rows = score_candidates(candidates, prepared)
    score_backup = write_csv(
        Path(args.score_csv),
        score_rows,
        [
            "module_name",
            "layer_id",
            "target_module",
            "component_index",
            "singular_value",
            "spectral_energy_share",
            "clean_sensitivity_raw",
            "clean_sensitivity_norm_global",
            "clean_sensitivity_norm_by_module",
            "suspiciousness_score_prelim",
            "prompt_count_used",
            "token_count_used",
            "notes",
            "A_key",
            "B_key",
        ],
        timestamp,
    )

    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "adapter_id": ADAPTER_ID,
        "adapter_snapshot": str(adapter_snapshot),
        "base_model": args.base_model,
        "spectral_stats_csv": str(args.spectral_stats_csv),
        "clean_prompt_file": str(prompt_path),
        "candidate_limit": int(args.candidate_limit),
        "candidate_count": len(candidates),
        "prompts_requested": len(prompts),
        "prompts_used": prompts_used,
        "elapsed_seconds": locals().get("elapsed_seconds", None),
        "candidate_csv": str(args.candidate_csv),
        "candidate_csv_backup": str(candidate_backup) if candidate_backup else None,
        "score_csv": str(args.score_csv),
        "score_csv_backup": str(score_backup) if score_backup else None,
        "warnings": vector_warnings + hook_warnings + [
            str(data["shape_warning"]) for data in prepared.values() if data.get("shape_warning")
        ],
        "errors": errors,
        "memory": {
            "before": memory_before,
            "after_load": memory_after_load,
            "after_forward": memory_after_forward,
            "after_cleanup": memory_after_cleanup,
        },
        "score_summary": summarize_scores(score_rows),
        "caveat": "First bounded clean sensitivity estimate, not final proof.",
    }


def score_candidates(candidates: list[dict[str, Any]], prepared: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_key: dict[tuple[str, int], float] = {}
    prompt_counts: dict[tuple[str, int], int] = {}
    token_counts: dict[tuple[str, int], int] = {}
    for module_name, data in prepared.items():
        token_count = max(int(data["token_count"]), 0)
        for local_index, component_index in enumerate(data["component_indices"]):
            key = (module_name, int(component_index))
            mean_proj_sq = 0.0
            if token_count > 0:
                mean_proj_sq = float(data["sum_sq"][local_index].item()) / float(token_count)
            raw_by_key[key] = mean_proj_sq
            prompt_counts[key] = int(data["prompt_count"])
            token_counts[key] = token_count

    preliminary_rows = []
    for candidate in candidates:
        key = (candidate["module_name"], int(candidate["component_index"]))
        mean_proj_sq = raw_by_key.get(key, 0.0)
        singular_value = float(candidate["singular_value"])
        raw_sensitivity = (singular_value**2) * mean_proj_sq
        preliminary_rows.append({**candidate, "clean_sensitivity_raw": raw_sensitivity})

    max_global = max((row["clean_sensitivity_raw"] for row in preliminary_rows), default=0.0)
    by_module_max: dict[str, float] = defaultdict(float)
    for row in preliminary_rows:
        by_module_max[row["module_name"]] = max(by_module_max[row["module_name"]], row["clean_sensitivity_raw"])

    rows = []
    for row in preliminary_rows:
        key = (row["module_name"], int(row["component_index"]))
        global_norm = row["clean_sensitivity_raw"] / max_global if max_global > 0 else 0.0
        module_max = by_module_max[row["module_name"]]
        module_norm = row["clean_sensitivity_raw"] / module_max if module_max > 0 else 0.0
        suspiciousness = float(row["spectral_energy_share"]) * (1.0 - global_norm)
        rows.append(
            {
                "module_name": row["module_name"],
                "layer_id": row["layer_id"],
                "target_module": row["target_module"],
                "component_index": int(row["component_index"]),
                "singular_value": row["singular_value"],
                "spectral_energy_share": row["spectral_energy_share"],
                "clean_sensitivity_raw": row["clean_sensitivity_raw"],
                "clean_sensitivity_norm_global": global_norm,
                "clean_sensitivity_norm_by_module": module_norm,
                "suspiciousness_score_prelim": suspiciousness,
                "prompt_count_used": prompt_counts.get(key, 0),
                "token_count_used": token_counts.get(key, 0),
                "notes": "First bounded sensitivity estimate; not final proof.",
                "A_key": row["A_key"],
                "B_key": row["B_key"],
            }
        )
    rows.sort(key=lambda item: item["suspiciousness_score_prelim"], reverse=True)
    return rows


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "row_count": len(rows),
        "mean_suspiciousness": statistics.mean(float(row["suspiciousness_score_prelim"]) for row in rows),
        "max_suspiciousness": max(float(row["suspiciousness_score_prelim"]) for row in rows),
        "top_rows": [
            {
                "layer_id": row["layer_id"],
                "target_module": row["target_module"],
                "component_index": row["component_index"],
                "suspiciousness_score_prelim": row["suspiciousness_score_prelim"],
            }
            for row in rows[:10]
        ],
    }


def print_summary(report: dict[str, Any], json_path: Path) -> None:
    print("Clean sensitivity probe summary")
    print("- Forward-only clean sensitivity probe; no generation")
    print(f"- Adapter: {report['adapter_id']}")
    print(f"- Base model: {report['base_model']}")
    print(f"- Candidate components: {report['candidate_count']}")
    print(f"- Clean prompts used: {report['prompts_used']} / {report['prompts_requested']}")
    print(f"- Warnings: {len(report['warnings'])}")
    print(f"- Errors: {len(report['errors'])}")
    print(f"- Candidate CSV written: {report['candidate_csv']}")
    print(f"- Score CSV written: {report['score_csv']}")
    print(f"- JSON log written: {json_path}")
    print("- Caveat: first bounded sensitivity estimate, not final proof.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe clean sensitivity for candidate LoRA components.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--spectral-stats-csv", default=str(DEFAULT_SPECTRAL_STATS))
    parser.add_argument("--clean-prompt-file", default=None)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-clean-prompts", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--score-csv", default=str(DEFAULT_SCORE_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_probe(args)
    json_path = Path(args.logs_dir) / f"clean_sensitivity_probe_{report['timestamp_utc']}.json"
    write_json(json_path, report)
    print_summary(report, json_path)
    return 0 if not report["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
