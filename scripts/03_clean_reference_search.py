"""Search for clean LoRA reference adapters without loading any base model.

This script is adapter/cache/metadata-only:
- scans local Hugging Face cache for PEFT LoRA adapter snapshots
- optionally queries Hugging Face model metadata
- never loads Llama-2 or any full model
- never runs inference
- never uses GPU

Remote config files are not downloaded unless --fetch-remote-configs is passed.
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

from lora_sanitise.lora_io import hf_hub_cache_roots  # noqa: E402


TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

REMOTE_SEARCH_TERMS = [
    "Llama-2-7b-chat lora",
    "Llama-2-7b-chat-hf lora_adapter_model",
    "Llama2 7B chat LoRA adapter",
    "BackdoorLLM Jailbreak_Llama2-7B",
]

SEED_REPOS = [
    "Guilherme34/Jennifer2-ENGLISH-CHAT.MULTITURN-LORA-7B-LLAMA2",
    "Aspik101/Llama-2-7b-chat-hf-pl-lora_adapter_model",
    "Lajonbot/Llama-2-7b-chat-hf-instruct-pl-lora_adapter_model",
    "Sparticle/llama-2-7b-chat-japanese-lora",
    "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview",
]

ATTACK_MARKERS = [
    "backdoor",
    "badnets",
    "jailbreak",
    "sleeper",
    "mtba",
    "ctba",
    "vpi",
    "poison",
]


@dataclass
class Candidate:
    repo_id: str
    source: str
    cached: bool = False
    snapshot_path: str | None = None
    reachable: bool | None = None
    private: bool | None = None
    gated: str | bool | None = None
    has_adapter_config: bool = False
    has_adapter_model_safetensors: bool = False
    has_adapter_model_bin: bool = False
    config_available: bool = False
    base_model_name_or_path: str | None = None
    peft_type: str | None = None
    task_type: str | None = None
    r: int | None = None
    lora_alpha: int | float | None = None
    target_modules: str = ""
    target_overlap: str = ""
    target_overlap_count: int = 0
    llama2_chat_compatible: bool = False
    attack_like_name: bool = False
    clean_reference_candidate: bool = False
    downloadable_adapter_candidate: bool = False
    score: int = 0
    notes: str = ""
    error: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_repo_id_from_cache_dir(path: Path) -> str:
    name = path.name
    if name.startswith("models--"):
        name = name[len("models--") :]
    return name.replace("--", "/")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def target_modules_from_config(config: dict[str, Any]) -> list[str]:
    value = config.get("target_modules")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def is_llama2_chat_compatible(repo_id: str, config: dict[str, Any] | None, tags: list[str] | None = None) -> bool:
    text_parts = [repo_id]
    if config:
        text_parts.append(str(config.get("base_model_name_or_path", "")))
    if tags:
        text_parts.extend(tags)
    text = " ".join(text_parts).lower()
    return "llama-2" in text and "7b" in text and ("chat" in text or "7b-chat" in text)


def is_attack_like(repo_id: str, tags: list[str] | None = None) -> bool:
    text = " ".join([repo_id, *(tags or [])]).lower()
    return any(marker in text for marker in ATTACK_MARKERS)


def score_candidate(candidate: Candidate) -> Candidate:
    score = 0
    notes: list[str] = []

    if candidate.attack_like_name:
        notes.append("excluded: attack/backdoor-like repo name or tags")
        candidate.score = -100
        candidate.clean_reference_candidate = False
        candidate.notes = "; ".join(notes)
        return candidate

    if candidate.llama2_chat_compatible:
        score += 3
    else:
        notes.append("base compatibility not confirmed")

    if candidate.has_adapter_config:
        score += 2
    else:
        notes.append("adapter_config.json not observed")

    if candidate.has_adapter_model_safetensors:
        score += 2
    elif candidate.has_adapter_model_bin:
        score += 1
        notes.append("adapter_model.bin observed instead of safetensors")
    else:
        notes.append("adapter model file not observed")

    if candidate.peft_type == "LORA":
        score += 2
    elif candidate.peft_type:
        notes.append(f"PEFT type is {candidate.peft_type}, not LORA")
    else:
        notes.append("PEFT type unknown until config is available")

    if candidate.target_overlap_count > 0:
        score += min(candidate.target_overlap_count, 7)
    else:
        notes.append("target module overlap unknown or absent")

    if candidate.cached:
        score += 2

    candidate.score = score
    candidate.clean_reference_candidate = (
        score >= 6
        and not candidate.attack_like_name
        and candidate.has_adapter_config
        and (candidate.has_adapter_model_safetensors or candidate.has_adapter_model_bin)
    )
    candidate.downloadable_adapter_candidate = (
        candidate.reachable is True
        and not candidate.attack_like_name
        and candidate.has_adapter_config
        and (candidate.has_adapter_model_safetensors or candidate.has_adapter_model_bin)
    )
    candidate.notes = "; ".join(notes) if notes else "looks suitable from available metadata"
    return candidate


def candidate_from_config(
    repo_id: str,
    source: str,
    config: dict[str, Any],
    cached: bool,
    snapshot_path: str | None,
    has_model_safetensors: bool,
    has_model_bin: bool,
    tags: list[str] | None = None,
    reachable: bool | None = None,
) -> Candidate:
    target_modules = target_modules_from_config(config)
    overlap = sorted(set(target_modules) & TARGET_MODULES)
    candidate = Candidate(
        repo_id=repo_id,
        source=source,
        cached=cached,
        snapshot_path=snapshot_path,
        reachable=reachable,
        has_adapter_config=True,
        has_adapter_model_safetensors=has_model_safetensors,
        has_adapter_model_bin=has_model_bin,
        config_available=True,
        base_model_name_or_path=config.get("base_model_name_or_path"),
        peft_type=config.get("peft_type"),
        task_type=config.get("task_type"),
        r=config.get("r"),
        lora_alpha=config.get("lora_alpha"),
        target_modules=",".join(target_modules),
        target_overlap=",".join(overlap),
        target_overlap_count=len(overlap),
        llama2_chat_compatible=is_llama2_chat_compatible(repo_id, config, tags),
        attack_like_name=is_attack_like(repo_id, tags),
    )
    return score_candidate(candidate)


def scan_cached_adapters(cache_roots: list[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for root in hf_hub_cache_roots(cache_roots):
        if not root.exists():
            continue
        for repo_dir in sorted(root.glob("models--*")):
            repo_id = normalize_repo_id_from_cache_dir(repo_dir)
            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.exists():
                continue
            for snapshot in sorted(snapshots_dir.iterdir()):
                if not snapshot.is_dir():
                    continue
                config_path = snapshot / "adapter_config.json"
                safetensors_path = snapshot / "adapter_model.safetensors"
                bin_path = snapshot / "adapter_model.bin"
                if not config_path.exists() and not safetensors_path.exists() and not bin_path.exists():
                    continue
                if config_path.exists():
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        candidate = candidate_from_config(
                            repo_id=repo_id,
                            source="cache",
                            config=config,
                            cached=True,
                            snapshot_path=str(snapshot),
                            has_model_safetensors=safetensors_path.exists(),
                            has_model_bin=bin_path.exists(),
                        )
                    except Exception as exc:
                        candidate = Candidate(
                            repo_id=repo_id,
                            source="cache",
                            cached=True,
                            snapshot_path=str(snapshot),
                            has_adapter_config=True,
                            has_adapter_model_safetensors=safetensors_path.exists(),
                            has_adapter_model_bin=bin_path.exists(),
                            attack_like_name=is_attack_like(repo_id),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        candidate = score_candidate(candidate)
                else:
                    candidate = Candidate(
                        repo_id=repo_id,
                        source="cache",
                        cached=True,
                        snapshot_path=str(snapshot),
                        has_adapter_config=False,
                        has_adapter_model_safetensors=safetensors_path.exists(),
                        has_adapter_model_bin=bin_path.exists(),
                        llama2_chat_compatible=is_llama2_chat_compatible(repo_id, None, None),
                        attack_like_name=is_attack_like(repo_id),
                    )
                    candidate = score_candidate(candidate)
                candidates.append(candidate)
    return candidates


def get_sibling_names(info: Any) -> list[str]:
    return sorted(s.rfilename for s in (getattr(info, "siblings", None) or []))


def fetch_remote_config(api: Any, repo_id: str) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, repo_type="model", filename="adapter_config.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_from_model_info(
    info: Any,
    source: str,
    fetch_config: bool,
) -> Candidate:
    repo_id = getattr(info, "id", None) or getattr(info, "modelId", "")
    siblings = get_sibling_names(info)
    tags = list(getattr(info, "tags", None) or [])
    has_config = "adapter_config.json" in siblings
    has_safetensors = "adapter_model.safetensors" in siblings
    has_bin = "adapter_model.bin" in siblings
    config: dict[str, Any] | None = None
    error = None

    if fetch_config and has_config:
        try:
            config = fetch_remote_config(None, repo_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    if config:
        candidate = candidate_from_config(
            repo_id=repo_id,
            source=source,
            config=config,
            cached=False,
            snapshot_path=None,
            has_model_safetensors=has_safetensors,
            has_model_bin=has_bin,
            tags=tags,
            reachable=True,
        )
    else:
        candidate = Candidate(
            repo_id=repo_id,
            source=source,
            cached=False,
            reachable=True,
            private=getattr(info, "private", None),
            gated=getattr(info, "gated", None),
            has_adapter_config=has_config,
            has_adapter_model_safetensors=has_safetensors,
            has_adapter_model_bin=has_bin,
            config_available=False,
            llama2_chat_compatible=is_llama2_chat_compatible(repo_id, None, tags),
            attack_like_name=is_attack_like(repo_id, tags),
            error=error,
        )
        candidate = score_candidate(candidate)
    candidate.error = error
    return candidate


def query_hugging_face(
    limit_per_query: int,
    include_seed_repos: bool,
    fetch_configs: bool,
) -> tuple[list[Candidate], list[str]]:
    errors: list[str] = []
    candidates: list[Candidate] = []
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [], [f"huggingface_hub import failed: {type(exc).__name__}: {exc}"]

    api = HfApi()
    seen: set[str] = set()

    if include_seed_repos:
        for repo_id in SEED_REPOS:
            try:
                info = api.model_info(repo_id)
                candidates.append(candidate_from_model_info(info, "hf_seed_metadata", fetch_configs))
                seen.add(repo_id)
            except Exception as exc:
                errors.append(f"{repo_id}: {type(exc).__name__}: {exc}")

    for term in REMOTE_SEARCH_TERMS:
        try:
            for info in api.list_models(search=term, limit=limit_per_query, full=True):
                repo_id = getattr(info, "id", None) or getattr(info, "modelId", "")
                if not repo_id or repo_id in seen:
                    continue
                candidates.append(candidate_from_model_info(info, f"hf_search:{term}", fetch_configs))
                seen.add(repo_id)
        except Exception as exc:
            errors.append(f"search {term!r}: {type(exc).__name__}: {exc}")

    return candidates, errors


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    best_by_repo: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best_by_repo.get(candidate.repo_id)
        if current is None or candidate.score > current.score or (
            candidate.score == current.score and candidate.cached and not current.cached
        ):
            best_by_repo[candidate.repo_id] = candidate
    return sorted(best_by_repo.values(), key=lambda item: (item.score, item.cached), reverse=True)


def write_json_log(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, candidates: list[Candidate], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    rows = [asdict(candidate) for candidate in candidates]
    fieldnames = list(asdict(Candidate(repo_id="", source="")).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path, backup


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cached = scan_cached_adapters(args.cache_root)
    remote: list[Candidate] = []
    hf_errors: list[str] = []
    if not args.offline:
        remote, hf_errors = query_hugging_face(
            limit_per_query=args.limit_per_query,
            include_seed_repos=not args.no_seed_repos,
            fetch_configs=args.fetch_remote_configs,
        )

    candidates = dedupe_candidates([*cached, *remote])
    best = next((candidate for candidate in candidates if candidate.clean_reference_candidate), None)
    best_downloadable = next(
        (candidate for candidate in candidates if candidate.downloadable_adapter_candidate),
        None,
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
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
        "search_terms": REMOTE_SEARCH_TERMS,
        "seed_repos": [] if args.no_seed_repos else SEED_REPOS,
        "hf_errors": hf_errors,
        "candidate_count": len(candidates),
        "clean_reference_possible": best is not None or best_downloadable is not None,
        "best_candidate": asdict(best or best_downloadable) if (best or best_downloadable) else None,
        "candidates": [asdict(candidate) for candidate in candidates],
        "fallback_plan": fallback_plan(best or best_downloadable),
    }


def fallback_plan(best: Candidate | None) -> str:
    if best is not None:
        return (
            "Verify the best candidate by downloading/inspecting adapter_config.json and "
            "adapter_model.safetensors only, then run clean-vs-backdoor spectral comparison."
        )
    return (
        "No suitable clean LoRA reference confirmed. Fallback: use distributional sanity checks "
        "within the backdoored adapter only, or manually train/download a small clean PEFT LoRA "
        "on matching instruction data later. Do not make backdoor-specific spectral claims "
        "without a clean reference."
    )


def print_summary(report: dict[str, Any], csv_path: Path, csv_backup: Path | None, json_path: Path) -> None:
    print("Clean reference search summary")
    print(f"- Candidates found: {report['candidate_count']}")
    print(f"- Clean-reference comparison possible: {report['clean_reference_possible']}")
    best = report["best_candidate"]
    if best:
        print(f"- Best candidate: {best['repo_id']}")
        print(f"  score={best['score']} cached={best['cached']} downloadable={best['downloadable_adapter_candidate']}")
        print(f"  peft_type={best['peft_type']} base={best['base_model_name_or_path']}")
        print(f"  target_overlap={best['target_overlap']}")
        print(f"  notes={best['notes']}")
    else:
        print("- Best candidate: none confirmed")
    if report["hf_errors"]:
        print("- Hugging Face metadata errors:")
        for error in report["hf_errors"]:
            print(f"  - {error}")
    print(f"- CSV written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")
    print(f"- JSON log written: {json_path}")
    print("- Next script name if candidate is confirmed: scripts/04_compare_clean_vs_backdoor_spectra.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Extra Hugging Face hub cache root to scan. Can be passed multiple times.",
    )
    parser.add_argument("--offline", action="store_true", help="Only scan local cache.")
    parser.add_argument(
        "--fetch-remote-configs",
        action="store_true",
        help="Download small adapter_config.json files for remote candidates only.",
    )
    parser.add_argument("--no-seed-repos", action="store_true")
    parser.add_argument("--limit-per-query", type=int, default=20)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--csv-path", default="outputs/clean_reference_candidates.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"clean_reference_search_{timestamp}.json"
    write_json_log(json_path, report)
    csv_path, csv_backup = write_csv(
        Path(args.csv_path),
        [Candidate(**candidate) for candidate in report["candidates"]],
        timestamp,
    )
    print_summary(report, csv_path, csv_backup, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
