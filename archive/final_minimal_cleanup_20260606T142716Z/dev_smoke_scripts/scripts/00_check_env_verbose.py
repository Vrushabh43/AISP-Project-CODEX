"""Verbose Python environment diagnostic for the LoRA sanitisation project.

This script does not download models, load Llama-2, run BackdoorLLM, or perform
GPU-heavy work. It only inspects the active Python interpreter, pip target,
package imports, user-site visibility, and PyTorch CUDA metadata.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_python_m_pip_version() -> dict[str, Any]:
    command = [sys.executable, "-m", "pip", "--version"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "error": None,
    }


def package_distribution_version(module_name: str) -> str | None:
    distribution_names = {
        "huggingface_hub": "huggingface-hub",
    }
    distribution_name = distribution_names.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_package(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    result: dict[str, Any] = {
        "module": module_name,
        "distribution_version": package_distribution_version(module_name),
        "find_spec": {
            "found": spec is not None,
            "origin": getattr(spec, "origin", None) if spec else None,
            "submodule_search_locations": list(spec.submodule_search_locations or [])
            if spec and spec.submodule_search_locations
            else [],
        },
        "import": {
            "ok": False,
            "version": None,
            "file": None,
            "error": None,
        },
    }

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["import"]["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["import"].update(
        {
            "ok": True,
            "version": getattr(module, "__version__", result["distribution_version"]),
            "file": getattr(module, "__file__", None),
            "error": None,
        }
    )
    return result


def inspect_torch_cuda(torch_info: dict[str, Any]) -> dict[str, Any]:
    if not torch_info["import"]["ok"]:
        return {
            "available": False,
            "reason": "torch import failed",
            "torch_version": None,
            "torch_cuda_version": None,
            "device_count": 0,
            "devices": [],
        }

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

    return {
        "available": cuda_available,
        "reason": None,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
    }


def build_report() -> dict[str, Any]:
    packages = {package: inspect_package(package) for package in PACKAGES}
    return {
        "timestamp_utc": utc_timestamp(),
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
            "path": list(sys.path),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "site": {
            "ENABLE_USER_SITE": site.ENABLE_USER_SITE,
            "USER_SITE": site.getusersitepackages(),
            "site_packages": site.getsitepackages(),
        },
        "environment": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "PATH": os.environ.get("PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
            "HUGGING_FACE_HUB_TOKEN_present": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        },
        "pip": run_python_m_pip_version(),
        "packages": packages,
        "cuda": inspect_torch_cuda(packages["torch"]),
    }


def write_log(report: dict[str, Any], logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"env_check_verbose_{report['timestamp_utc']}.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return log_path


def print_package_summary(packages: dict[str, dict[str, Any]]) -> None:
    print("Package import status")
    for name in PACKAGES:
        info = packages[name]
        imported = info["import"]
        spec = info["find_spec"]
        print(f"- {name}:")
        print(f"  find_spec: {spec['found']}")
        print(f"  spec origin: {spec['origin']}")
        print(f"  dist version: {info['distribution_version']}")
        print(f"  import ok: {imported['ok']}")
        print(f"  import version: {imported['version']}")
        print(f"  module file: {imported['file']}")
        if imported["error"]:
            print(f"  import error: {imported['error']}")


def print_summary(report: dict[str, Any], log_path: Path) -> None:
    python = report["python"]
    site_info = report["site"]
    cuda = report["cuda"]

    print("Verbose environment check")
    print(f"- Python executable: {python['executable']}")
    print(f"- Python version: {python['version']}")
    print(f"- sys.prefix: {python['prefix']}")
    print(f"- sys.base_prefix: {python['base_prefix']}")
    print(f"- venv active by prefix check: {python['venv_active']}")
    print("- sys.path:")
    for entry in python["path"]:
        print(f"  - {entry}")
    print(f"- ENABLE_USER_SITE: {site_info['ENABLE_USER_SITE']}")
    print(f"- USER_SITE: {site_info['USER_SITE']}")
    print(f"- python -m pip --version: {report['pip']['stdout']}")
    if report["pip"]["stderr"]:
        print(f"- python -m pip stderr: {report['pip']['stderr']}")
    print_package_summary(report["packages"])
    print("CUDA status")
    print(f"- torch version: {cuda['torch_version']}")
    print(f"- CUDA available: {cuda['available']}")
    print(f"- torch CUDA version: {cuda['torch_cuda_version']}")
    print(f"- CUDA devices: {cuda['device_count']}")
    for device in cuda["devices"]:
        print(
            f"  - cuda:{device['index']} {device['name']} "
            f"({device['total_memory_gb']} GB, capability {device['capability']})"
        )
    print(f"- JSON log written: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory where the JSON diagnostic log should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    log_path = write_log(report, Path(args.logs_dir))
    print_summary(report, log_path)
    missing = [
        name
        for name, info in report["packages"].items()
        if name != "bitsandbytes" and not info["import"]["ok"]
    ]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
