"""Lightweight environment check for the LoRA sanitisation project.

This script intentionally avoids model downloads and model inference. The
optional Hugging Face check uses repository metadata only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def check_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "version": getattr(module, "__version__", "unknown"),
        "error": None,
    }


def check_torch() -> dict[str, Any]:
    info = check_import("torch")
    if not info["available"]:
        return info

    import torch

    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    info.update(
        {
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": devices,
        }
    )
    return info


def check_hugging_face(adapter_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": True,
        "adapter_id": adapter_id,
        "token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "whoami": None,
        "adapter_info": None,
        "error": None,
    }

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            result["whoami"] = api.whoami()
        except Exception as exc:  # Token may be absent; metadata can still work.
            result["whoami"] = {"error": f"{type(exc).__name__}: {exc}"}

        info = api.model_info(adapter_id)
        result["adapter_info"] = {
            "id": info.id,
            "private": info.private,
            "gated": getattr(info, "gated", None),
            "sha": info.sha,
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "siblings": sorted(s.rfilename for s in (info.siblings or [])),
        }
    except Exception as exc:  # pragma: no cover - network/auth diagnostic path
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packages = {
        name: check_import(name)
        for name in [
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
            "huggingface_hub",
            "numpy",
            "pandas",
            "yaml",
            "tqdm",
            "sklearn",
            "matplotlib",
            "seaborn",
        ]
    }
    packages["torch"] = check_torch()
    packages["bitsandbytes"] = check_import("bitsandbytes")

    report: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cwd": str(Path.cwd()),
        "environment": {
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
            "HUGGING_FACE_HUB_TOKEN_present": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "packages": packages,
        "hugging_face": {"requested": False},
    }

    if args.check_hf:
        report["hugging_face"] = check_hugging_face(args.adapter_id)

    return report


def write_log(report: dict[str, Any], logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"env_check_{report['timestamp_utc']}.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return log_path


def print_summary(report: dict[str, Any], log_path: Path) -> None:
    torch_info = report["packages"]["torch"]
    missing_required = [
        name
        for name, info in report["packages"].items()
        if name != "bitsandbytes" and not info["available"]
    ]

    print("Environment check summary")
    print(f"- Python: {report['python']['version_info']}")
    print(f"- Platform: {report['platform']['system']} {report['platform']['release']}")
    print(f"- Torch available: {torch_info['available']}")
    if torch_info["available"]:
        print(f"- Torch version: {torch_info['version']}")
        print(f"- CUDA available: {torch_info.get('cuda_available')}")
        print(f"- CUDA version: {torch_info.get('cuda_version')}")
        print(f"- CUDA devices: {torch_info.get('device_count')}")
        for device in torch_info.get("devices", []):
            print(
                f"  - cuda:{device['index']} {device['name']} "
                f"({device['total_memory_gb']} GB)"
            )
    print(f"- Missing required packages: {missing_required or 'none'}")
    print(f"- bitsandbytes available: {report['packages']['bitsandbytes']['available']}")

    hf = report["hugging_face"]
    if hf.get("requested"):
        print(f"- HF token present: {hf.get('token_present')}")
        print(f"- HF adapter metadata reachable: {hf.get('adapter_info') is not None}")
        if hf.get("error"):
            print(f"- HF check error: {hf['error']}")

    print(f"- Log written: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-hf",
        action="store_true",
        help="Check Hugging Face API access and adapter metadata without downloading weights.",
    )
    parser.add_argument(
        "--adapter-id",
        default=DEFAULT_ADAPTER_ID,
        help="Hugging Face adapter repository ID to check when --check-hf is used.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory where the JSON environment log should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    log_path = write_log(report, Path(args.logs_dir))
    print_summary(report, log_path)

    missing_required = [
        name
        for name, info in report["packages"].items()
        if name != "bitsandbytes" and not info["available"]
    ]
    if missing_required:
        return 1
    if args.check_hf and report["hugging_face"].get("error"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
