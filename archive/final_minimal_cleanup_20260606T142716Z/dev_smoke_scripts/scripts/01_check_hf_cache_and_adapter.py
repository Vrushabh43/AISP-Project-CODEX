"""Safe Hugging Face cache and adapter availability check.

Default behavior:
- scans local Hugging Face cache directories
- queries Hugging Face repo metadata if huggingface_hub is installed
- writes a JSON log to logs/

It does not load Llama-2, load any adapter into a model, run inference, or use
the GPU. Downloads are disabled unless an explicit download flag is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import site
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOS = [
    {
        "label": "base_model_fallback",
        "repo_id": "NousResearch/Llama-2-7b-chat-hf",
        "repo_type": "model",
    },
    {
        "label": "alpaca_dataset",
        "repo_id": "tatsu-lab/alpaca",
        "repo_type": "dataset",
    },
    {
        "label": "backdoored_adapter",
        "repo_id": "BackdoorLLM/Jailbreak_Llama2-7B_BadNets",
        "repo_type": "model",
    },
]

ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"

METADATA_ALLOW_PATTERNS = [
    "*.json",
    "*.md",
    "*.txt",
    "*.gitattributes",
    "README*",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_cache_dirname(repo_id: str, repo_type: str) -> str:
    prefix = {
        "model": "models",
        "dataset": "datasets",
        "space": "spaces",
    }.get(repo_type, f"{repo_type}s")
    return f"{prefix}--{repo_id.replace('/', '--')}"


def default_hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser()
    return Path.home() / ".cache" / "huggingface"


def candidate_cache_roots(extra_roots: list[str]) -> list[Path]:
    candidates: list[Path] = []

    for env_name in ["HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = default_hf_home()
    candidates.extend(
        [
            hf_home / "hub",
            hf_home,
        ]
    )

    for root in extra_roots:
        candidates.append(Path(root).expanduser())

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def candidate_dataset_cache_roots() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("HF_DATASETS_CACHE"):
        candidates.append(Path(os.environ["HF_DATASETS_CACHE"]).expanduser())
    hf_home = default_hf_home()
    candidates.append(hf_home / "datasets")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def inspect_cached_repo(root: Path, repo_id: str, repo_type: str) -> dict[str, Any]:
    cache_dir = root / repo_cache_dirname(repo_id, repo_type)
    snapshots_dir = cache_dir / "snapshots"
    refs_dir = cache_dir / "refs"

    snapshots: list[dict[str, Any]] = []
    if snapshots_dir.exists():
        for snapshot in sorted(snapshots_dir.iterdir()):
            if snapshot.is_dir():
                files = [p for p in snapshot.rglob("*") if p.is_file()]
                sample_files = sorted(
                    str(p.relative_to(snapshot)).replace("\\", "/") for p in files[:20]
                )
                snapshots.append(
                    {
                        "name": snapshot.name,
                        "path": str(snapshot),
                        "file_count": len(files),
                        "sample_files": sample_files,
                    }
                )

    refs: dict[str, str] = {}
    if refs_dir.exists():
        for ref_file in sorted(refs_dir.iterdir()):
            if ref_file.is_file():
                try:
                    refs[ref_file.name] = ref_file.read_text(encoding="utf-8").strip()
                except UnicodeDecodeError:
                    refs[ref_file.name] = "<binary-or-non-utf8>"

    return {
        "root": str(root),
        "expected_cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "snapshots_dir_exists": snapshots_dir.exists(),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "refs": refs,
    }


def inspect_dataset_library_cache(repo_id: str) -> list[dict[str, Any]]:
    dirname = repo_id.replace("/", "___")
    results: list[dict[str, Any]] = []
    for root in candidate_dataset_cache_roots():
        exact = root / dirname
        matches = sorted(root.glob(f"*{dirname}*")) if root.exists() else []
        results.append(
            {
                "root": str(root),
                "expected_cache_dir": str(exact),
                "expected_cache_dir_exists": exact.exists(),
                "matching_paths": [str(path) for path in matches],
                "match_count": len(matches),
            }
        )
    return results


def scan_cache(roots: list[Path]) -> dict[str, Any]:
    cache_entries: dict[str, Any] = {}
    for repo in REPOS:
        repo_id = repo["repo_id"]
        repo_type = repo["repo_type"]
        per_root = [inspect_cached_repo(root, repo_id, repo_type) for root in roots]
        dataset_library_cache = (
            inspect_dataset_library_cache(repo_id) if repo_type == "dataset" else []
        )
        cache_entries[repo_id] = {
            "repo_type": repo_type,
            "label": repo["label"],
            "cached": any(entry["snapshot_count"] > 0 for entry in per_root)
            or any(entry["match_count"] > 0 for entry in dataset_library_cache),
            "per_root": per_root,
            "dataset_library_cache": dataset_library_cache,
        }
    return cache_entries


def huggingface_hub_available() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def repo_metadata(repo_id: str, repo_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "reachable": False,
        "info": None,
        "error": None,
    }

    if not huggingface_hub_available():
        result["error"] = "huggingface_hub is not installed in this Python environment"
        return result

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if repo_type == "dataset":
            info = api.dataset_info(repo_id)
        else:
            info = api.model_info(repo_id)

        siblings = sorted(s.rfilename for s in (info.siblings or []))
        result["reachable"] = True
        result["info"] = {
            "id": info.id,
            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "sha": getattr(info, "sha", None),
            "last_modified": str(getattr(info, "last_modified", None)),
            "sibling_count": len(siblings),
            "siblings": siblings,
        }
    except Exception as exc:  # pragma: no cover - network/auth diagnostic path
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_repos_metadata() -> dict[str, Any]:
    return {
        repo["repo_id"]: repo_metadata(repo["repo_id"], repo["repo_type"])
        for repo in REPOS
    }


def snapshot_download_adapter(
    cache_dir: str | None,
    metadata_only: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": True,
        "repo_id": ADAPTER_ID,
        "metadata_only": metadata_only,
        "cache_dir": cache_dir,
        "local_dir": None,
        "allow_patterns": METADATA_ALLOW_PATTERNS if metadata_only else None,
        "error": None,
    }

    if not huggingface_hub_available():
        result["error"] = "huggingface_hub is not installed in this Python environment"
        return result

    try:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": ADAPTER_ID,
            "repo_type": "model",
            "cache_dir": cache_dir,
        }
        if metadata_only:
            kwargs["allow_patterns"] = METADATA_ALLOW_PATTERNS

        local_dir = snapshot_download(**kwargs)
        result["local_dir"] = local_dir
    except Exception as exc:  # pragma: no cover - network/auth/download diagnostic path
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    roots = candidate_cache_roots(args.cache_root)
    metadata = check_repos_metadata() if not args.offline else {}
    download: dict[str, Any] = {"requested": False}

    if args.download_adapter or args.download_adapter_metadata_only:
        download = snapshot_download_adapter(
            cache_dir=args.download_cache_dir,
            metadata_only=args.download_adapter_metadata_only,
        )

    return {
        "timestamp_utc": utc_timestamp(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "site": {
            "ENABLE_USER_SITE": site.ENABLE_USER_SITE,
            "USER_SITE": site.getusersitepackages(),
        },
        "environment": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
            "HUGGING_FACE_HUB_TOKEN_present": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "huggingface_hub_available": huggingface_hub_available(),
        "cache_roots": [str(root) for root in roots],
        "cache_scan": scan_cache(roots),
        "repo_metadata": metadata,
        "download": download,
    }


def write_log(report: dict[str, Any], logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"hf_cache_adapter_check_{report['timestamp_utc']}.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return log_path


def print_summary(report: dict[str, Any], log_path: Path) -> None:
    print("Hugging Face cache and adapter check")
    print(f"- Python executable: {report['python']['executable']}")
    print(f"- Python version: {report['python']['version_info']}")
    print(f"- venv active: {report['python']['venv_active']}")
    print(f"- huggingface_hub available: {report['huggingface_hub_available']}")
    print("- Cache roots checked:")
    for root in report["cache_roots"]:
        print(f"  - {root}")

    print("- Cache status:")
    for repo in REPOS:
        repo_id = repo["repo_id"]
        entry = report["cache_scan"][repo_id]
        print(f"  - {repo_id} ({repo['repo_type']}): cached={entry['cached']}")

    if report["repo_metadata"]:
        print("- Repo metadata reachability:")
        for repo in REPOS:
            repo_id = repo["repo_id"]
            metadata = report["repo_metadata"][repo_id]
            print(f"  - {repo_id}: reachable={metadata['reachable']}")
            if metadata["error"]:
                print(f"    error={metadata['error']}")

    download = report["download"]
    if download.get("requested"):
        print("- Download request:")
        print(f"  - repo_id: {download['repo_id']}")
        print(f"  - metadata_only: {download['metadata_only']}")
        print(f"  - local_dir: {download['local_dir']}")
        if download["error"]:
            print(f"  - error: {download['error']}")

    print(f"- JSON log written: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Extra Hugging Face cache root to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Hugging Face API metadata checks and only inspect local cache.",
    )
    parser.add_argument(
        "--download-adapter-metadata-only",
        action="store_true",
        help="Explicitly download only small adapter metadata files into the HF cache.",
    )
    parser.add_argument(
        "--download-adapter",
        action="store_true",
        help="Explicitly download the adapter snapshot only. Does not download the base model.",
    )
    parser.add_argument(
        "--download-cache-dir",
        default=None,
        help="Optional cache_dir passed to huggingface_hub.snapshot_download.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory where the JSON log should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download_adapter and args.download_adapter_metadata_only:
        raise SystemExit("Use only one download flag at a time.")

    report = build_report(args)
    log_path = write_log(report, Path(args.logs_dir))
    print_summary(report, log_path)

    adapter_meta = report["repo_metadata"].get(ADAPTER_ID) if report["repo_metadata"] else None
    if adapter_meta and not adapter_meta["reachable"]:
        return 2
    download = report["download"]
    if download.get("requested") and download.get("error"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
