"""Lightweight GPU memory diagnosis.

This script does not load Llama-2, does not load adapters, and does not run
inference. It only inspects CUDA memory state and nvidia-smi output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command": command,
        "available": shutil.which(command[0]) is not None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    if not result["available"]:
        result["error"] = f"{command[0]} not found on PATH"
        return result
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_csv_lines(text: str, fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        row: dict[str, Any] = {}
        for field, value in zip(fields, values):
            row[field] = coerce_number(value)
        rows.append(row)
    return rows


def coerce_number(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def torch_cuda_summary() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_import_ok": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        result.update(
            {
                "torch_import_ok": True,
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
            }
        )
        if not torch.cuda.is_available():
            return result

        result["device_count"] = torch.cuda.device_count()
        for index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_mb": round(total_bytes / (1024**2), 3),
                    "free_mb": round(free_bytes / (1024**2), 3),
                    "used_mb_from_mem_get_info": round((total_bytes - free_bytes) / (1024**2), 3),
                    "torch_allocated_mb": round(torch.cuda.memory_allocated(index) / (1024**2), 3),
                    "torch_reserved_mb": round(torch.cuda.memory_reserved(index) / (1024**2), 3),
                    "torch_max_allocated_mb": round(
                        torch.cuda.max_memory_allocated(index) / (1024**2), 3
                    ),
                    "torch_max_reserved_mb": round(
                        torch.cuda.max_memory_reserved(index) / (1024**2), 3
                    ),
                    "compute_capability": [int(props.major), int(props.minor)],
                }
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def nvidia_smi_summary() -> dict[str, Any]:
    gpu_fields = ["index", "name", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization_gpu_percent"]
    process_fields = ["gpu_uuid", "pid", "process_name", "used_memory_mb"]
    gpu_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    full_output = run_command(["nvidia-smi"])
    return {
        "gpu_query": gpu_query,
        "process_query": process_query,
        "full_output": full_output,
        "gpus": parse_csv_lines(gpu_query["stdout"], gpu_fields)
        if gpu_query.get("returncode") == 0
        else [],
        "processes": parse_csv_lines(process_query["stdout"], process_fields)
        if process_query.get("returncode") == 0
        else [],
    }


def best_free_vram_mb(torch_summary: dict[str, Any], smi_summary: dict[str, Any]) -> float:
    candidates: list[float] = []
    for device in torch_summary.get("devices", []):
        free = device.get("free_mb")
        if isinstance(free, (int, float)):
            candidates.append(float(free))
    for gpu in smi_summary.get("gpus", []):
        free = gpu.get("memory_free_mb")
        if isinstance(free, (int, float)):
            candidates.append(float(free))
    return max(candidates) if candidates else 0.0


def build_recommendation(best_free_mb: float, threshold_mb: float) -> dict[str, Any]:
    enough = best_free_mb >= threshold_mb
    return {
        "free_vram_threshold_mb": threshold_mb,
        "best_free_vram_mb": best_free_mb,
        "enough_free_vram_to_retry_4bit_attach": enough,
        "recommendation": (
            "Enough free VRAM by the configured threshold; retry attach-only, not tiny-forward."
            if enough
            else "Not enough free VRAM by the configured threshold; free GPU memory or use CPU/disk offload mode."
        ),
    }


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_report(report: dict[str, Any], json_path: Path) -> None:
    torch_summary = report["torch_cuda"]
    smi = report["nvidia_smi"]
    recommendation = report["recommendation"]
    print("GPU memory diagnosis")
    print(f"- Python executable: {report['python']['executable']}")
    print(f"- torch import ok: {torch_summary['torch_import_ok']}")
    print(f"- torch version: {torch_summary['torch_version']}")
    print(f"- CUDA available: {torch_summary['cuda_available']}")
    print(f"- torch CUDA version: {torch_summary['cuda_version']}")
    print("- torch GPU memory summary:")
    if torch_summary["devices"]:
        for device in torch_summary["devices"]:
            print(
                "  - "
                f"cuda:{device['index']} {device['name']} "
                f"free={device['free_mb']} MB total={device['total_mb']} MB "
                f"torch_allocated={device['torch_allocated_mb']} MB "
                f"torch_reserved={device['torch_reserved_mb']} MB"
            )
    else:
        print("  - no CUDA devices reported by torch")

    print(f"- nvidia-smi available: {smi['gpu_query']['available']}")
    print("- nvidia-smi GPU summary:")
    if smi["gpus"]:
        for gpu in smi["gpus"]:
            print(
                "  - "
                f"gpu={gpu.get('index')} {gpu.get('name')} "
                f"free={gpu.get('memory_free_mb')} MB "
                f"used={gpu.get('memory_used_mb')} MB "
                f"total={gpu.get('memory_total_mb')} MB "
                f"util={gpu.get('utilization_gpu_percent')}%"
            )
    else:
        print("  - no GPU rows returned")

    print("- GPU processes from nvidia-smi:")
    if smi["processes"]:
        for proc in smi["processes"]:
            print(
                "  - "
                f"pid={proc.get('pid')} "
                f"used={proc.get('used_memory_mb')} MB "
                f"name={proc.get('process_name')} "
                f"gpu_uuid={proc.get('gpu_uuid')}"
            )
    else:
        print("  - no compute processes reported, or process query unavailable")

    print(
        "- Enough free VRAM to retry 4-bit attach: "
        f"{recommendation['enough_free_vram_to_retry_4bit_attach']}"
    )
    print(f"- Best free VRAM observed: {recommendation['best_free_vram_mb']} MB")
    print(f"- Free VRAM threshold: {recommendation['free_vram_threshold_mb']} MB")
    print(f"- Recommendation: {recommendation['recommendation']}")
    print(f"- JSON log written: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--free-vram-threshold-mb",
        type=float,
        default=6500.0,
        help="Minimum free VRAM to recommend retrying 4-bit attach.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    torch_summary = torch_cuda_summary()
    smi_summary = nvidia_smi_summary()
    best_free = best_free_vram_mb(torch_summary, smi_summary)
    report = {
        "timestamp_utc": timestamp,
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
        "torch_cuda": torch_summary,
        "nvidia_smi": smi_summary,
        "recommendation": build_recommendation(best_free, args.free_vram_threshold_mb),
        "safety": {
            "loads_llama2": False,
            "loads_adapters": False,
            "runs_inference": False,
            "modifies_files_outside_log": False,
        },
    }
    json_path = Path(args.logs_dir) / f"gpu_memory_diagnosis_{timestamp}.json"
    write_json(json_path, report)
    print_report(report, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
