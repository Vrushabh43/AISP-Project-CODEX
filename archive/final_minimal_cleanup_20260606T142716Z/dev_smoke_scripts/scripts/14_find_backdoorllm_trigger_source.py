"""Locate local evidence for BackdoorLLM BadNets trigger/evaluation format.

This script performs filesystem/source inspection only. It does not download
repositories, execute third-party code, load models, run inference, or run ASR.
It avoids printing long prompt content and records candidate locations instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_sanitise.lora_io import hf_hub_cache_roots, repo_cache_dirname  # noqa: E402


ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
KEYWORDS = ["BadNets", "trigger", "poison", "target", "jailbreak", "ASR", "dataset", "eval", "cf"]
STRONG_SOURCE_TERMS = {"badnets", "trigger", "jailbreak", "poison", "asr"}
TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "venv",
}
SKIP_FILE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
}
GENERATED_DIR_NAMES = {"logs", "outputs", "reports"}
OFFICIAL_SOURCE_SCOPES = {"local_backdoorllm_repo", "hf_cached_adapter"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def as_posix(path: Path) -> str:
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return as_posix(path)


def safe_read_text(path: Path, max_file_bytes: int) -> str | None:
    try:
        if not path.is_file():
            return None
        if path.suffix.lower() in SKIP_FILE_SUFFIXES:
            return None
        if path.stat().st_size > max_file_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def is_text_file(path: Path, max_file_bytes: int) -> bool:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return False
    try:
        return path.is_file() and path.stat().st_size <= max_file_bytes
    except OSError:
        return False


def is_generated_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & GENERATED_DIR_NAMES)


def looks_like_local_backdoorllm_repo(path: Path) -> bool:
    # Use directory components only. Filenames in this project can contain
    # "backdoorllm" (for example this script) and must not be treated as an
    # external BackdoorLLM repository.
    probe = path.parent if path.suffix else path
    lowered_parts = [part.lower() for part in probe.parts]
    joined = "/".join(lowered_parts)
    return "backdoorllm" in joined or "backdoor-llm" in joined


def source_scope_for_path(path: Path, explicit_scope: str | None = None) -> str:
    if explicit_scope:
        return explicit_scope
    if is_generated_path(path):
        return "generated_project_artifact"
    if looks_like_local_backdoorllm_repo(path):
        return "local_backdoorllm_repo"
    return "current_project"


def selected_repo_filename(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in ["readme", "config", "dataset", "eval", "trigger", "badnets", "jailbreak"]
    )


def keyword_hits_for_text(text: str) -> tuple[list[str], list[int], dict[str, int], bool]:
    lowered_keywords = {keyword: keyword.lower() for keyword in KEYWORDS}
    counts: Counter[str] = Counter()
    line_numbers: list[int] = []
    explicit_trigger_format = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered_line = line.lower()
        hits_here = [keyword for keyword, lowered in lowered_keywords.items() if lowered in lowered_line]
        if not hits_here:
            continue
        line_numbers.append(line_number)
        counts.update(hits_here)
        if ("trigger" in lowered_line and "cf" in lowered_line) or (
            "badnets" in lowered_line and "cf" in lowered_line
        ):
            explicit_trigger_format = True
    return sorted(counts), line_numbers[:10], dict(counts), explicit_trigger_format


def confidence_for_candidate(candidate: dict[str, Any]) -> str:
    source_scope = candidate["source_scope"]
    keyword_set = set(candidate["matched_keywords_lower"])
    explicit = bool(candidate["explicit_trigger_format_candidate"])

    if source_scope in {"local_backdoorllm_repo", "hf_cached_adapter"}:
        if explicit:
            return "high"
        if keyword_set & STRONG_SOURCE_TERMS:
            return "medium"
        return "low"
    if source_scope == "generated_project_artifact":
        return "low"
    if explicit and {"badnets", "trigger"} & keyword_set:
        return "medium"
    if "badnets" in keyword_set and "trigger" in keyword_set:
        return "medium"
    return "low"


def is_official_verification_candidate(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate can verify the official trigger format."""
    return (
        candidate["source_scope"] in OFFICIAL_SOURCE_SCOPES
        and candidate["confidence"] == "high"
        and bool(candidate["explicit_trigger_format_candidate"])
    )


def scan_file(path: Path, source_scope: str, max_file_bytes: int) -> dict[str, Any] | None:
    text = safe_read_text(path, max_file_bytes)
    if text is None:
        return None
    matched, line_numbers, counts, explicit = keyword_hits_for_text(text)
    if not matched:
        return None
    matched_lower = sorted({item.lower() for item in matched})
    candidate: dict[str, Any] = {
        "source_scope": source_scope,
        "path": relative_to_root(path),
        "absolute_path": as_posix(path),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "matched_keywords": matched,
        "matched_keywords_lower": matched_lower,
        "keyword_counts": counts,
        "line_numbers_sample": line_numbers,
        "explicit_trigger_format_candidate": explicit,
        "contains_prompt_content": any(token in path.name.lower() for token in ["prompt", "dataset", "eval"]),
        "content_not_printed": True,
        "notes": "Long prompt/content text intentionally not printed.",
    }
    candidate["confidence"] = confidence_for_candidate(candidate)
    return candidate


def iter_project_files(root: Path, max_file_bytes: int) -> list[Path]:
    files: list[Path] = []
    local_backdoor_roots: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in SKIP_DIR_NAMES
            and not dirname.startswith("models--")
            and dirname not in {"checkpoints", "models"}
        ]
        if looks_like_local_backdoorllm_repo(current):
            local_backdoor_roots.append(current)
        for filename in filenames:
            path = current / filename
            if not is_text_file(path, max_file_bytes):
                continue
            if looks_like_local_backdoorllm_repo(path) and not selected_repo_filename(path):
                continue
            files.append(path)
    return sorted(set(files))


def find_local_backdoorllm_repo_files(root: Path, max_file_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in SKIP_DIR_NAMES and not dirname.startswith("models--")
        ]
        if not looks_like_local_backdoorllm_repo(current):
            continue
        marker = current.resolve()
        if marker in seen_dirs:
            continue
        seen_dirs.add(marker)
        selected_files = []
        for file_path in current.rglob("*"):
            if file_path.is_file() and selected_repo_filename(file_path) and is_text_file(file_path, max_file_bytes):
                selected_files.append(relative_to_root(file_path))
        rows.append(
            {
                "repo_like_dir": relative_to_root(current),
                "selected_readme_config_dataset_files": sorted(selected_files)[:100],
                "selected_file_count": len(selected_files),
            }
        )
    return rows


def cached_adapter_files(cache_roots: list[str], max_file_bytes: int) -> list[Path]:
    dirname = repo_cache_dirname(ADAPTER_ID, "model")
    files: list[Path] = []
    for root in hf_hub_cache_roots(cache_roots):
        snapshots = root / dirname / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            for filename in ["README.md", "adapter_config.json", "config.json"]:
                path = snapshot / filename
                if is_text_file(path, max_file_bytes):
                    files.append(path)
            for pattern in ["*.json", "*.jsonl", "*.yaml", "*.yml", "*.md", "*.txt"]:
                for path in snapshot.glob(pattern):
                    if is_text_file(path, max_file_bytes):
                        files.append(path)
    return sorted(set(files))


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, candidates: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "confidence",
        "source_scope",
        "path",
        "file_name",
        "matched_keywords",
        "line_numbers_sample",
        "explicit_trigger_format_candidate",
        "content_not_printed",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "confidence": candidate["confidence"],
                    "source_scope": candidate["source_scope"],
                    "path": candidate["path"],
                    "file_name": candidate["file_name"],
                    "matched_keywords": ";".join(candidate["matched_keywords"]),
                    "line_numbers_sample": ";".join(str(item) for item in candidate["line_numbers_sample"]),
                    "explicit_trigger_format_candidate": candidate["explicit_trigger_format_candidate"],
                    "content_not_printed": candidate["content_not_printed"],
                    "notes": candidate["notes"],
                }
            )
    return path, backup


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    project_root = Path(args.root).expanduser().resolve()
    project_files = iter_project_files(project_root, args.max_file_bytes)
    adapter_files = cached_adapter_files(args.cache_root, args.max_file_bytes)
    local_repo_summaries = find_local_backdoorllm_repo_files(project_root, args.max_file_bytes)

    candidates: list[dict[str, Any]] = []
    for path in project_files:
        candidate = scan_file(path, source_scope_for_path(path), args.max_file_bytes)
        if candidate:
            candidates.append(candidate)
    for path in adapter_files:
        candidate = scan_file(path, "hf_cached_adapter", args.max_file_bytes)
        if candidate:
            candidates.append(candidate)

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        deduped[(candidate["source_scope"], candidate["absolute_path"])] = candidate
    candidates = sorted(
        deduped.values(),
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item["confidence"], 3),
            item["source_scope"],
            item["path"],
        ),
    )
    high_candidates = [item for item in candidates if item["confidence"] == "high"]
    official_verified = any(is_official_verification_candidate(item) for item in candidates)
    if official_verified:
        next_step = "Review high-confidence official candidate files manually before creating ASR prompt files."
    else:
        next_step = (
            "Official trigger format not verified locally. Manually inspect the BackdoorLLM "
            "repository or paper materials next; do not proceed to ASR."
        )

    return {
        "timestamp_utc": timestamp,
        "script": "scripts/14_find_backdoorllm_trigger_source.py",
        "purpose": "local_source_inspection_only_no_download_no_execution",
        "adapter_id": ADAPTER_ID,
        "keywords": KEYWORDS,
        "root": as_posix(project_root),
        "cache_roots_checked": [as_posix(path) for path in hf_hub_cache_roots(args.cache_root)],
        "limits": {
            "max_file_bytes": args.max_file_bytes,
            "content_snippets_printed": False,
            "downloads_code_or_models": False,
            "executes_third_party_code": False,
            "runs_model_loading_or_inference": False,
            "runs_asr": False,
        },
        "files_searched_count": len(project_files) + len(adapter_files),
        "project_files_searched_count": len(project_files),
        "cached_adapter_files_searched_count": len(adapter_files),
        "local_backdoorllm_repo_summaries": local_repo_summaries,
        "candidate_count": len(candidates),
        "confidence_counts": dict(Counter(item["confidence"] for item in candidates)),
        "official_trigger_format_verified": official_verified,
        "candidate_source_locations": candidates,
        "high_confidence_candidates": high_candidates,
        "next_recommended_step": next_step,
    }


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, backup: Path | None) -> None:
    print("BackdoorLLM trigger source search summary")
    print(f"- Files searched: {report['files_searched_count']}")
    print(f"  - project files: {report['project_files_searched_count']}")
    print(f"  - cached adapter files: {report['cached_adapter_files_searched_count']}")
    print(f"- Candidate source locations: {report['candidate_count']}")
    print(f"- Confidence counts: {report['confidence_counts']}")
    print(f"- Official trigger format verified: {report['official_trigger_format_verified']}")
    if report["local_backdoorllm_repo_summaries"]:
        print("- Local BackdoorLLM-like repo directories found:")
        for item in report["local_backdoorllm_repo_summaries"][:5]:
            print(f"  - {item['repo_like_dir']} ({item['selected_file_count']} selected files)")
    if report["candidate_source_locations"]:
        print("- Top candidate locations:")
        for item in report["candidate_source_locations"][:10]:
            print(
                f"  - confidence={item['confidence']} source={item['source_scope']} "
                f"path={item['path']} keywords={','.join(item['matched_keywords'])}"
            )
    else:
        print("- Top candidate locations: none")
    print(f"- Next recommended step: {report['next_recommended_step']}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV candidates written: {csv_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Project root to inspect.")
    parser.add_argument("--cache-root", action="append", default=[], help="Extra HF hub cache root.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--outputs-csv-path", default="outputs/backdoorllm_trigger_source_candidates.csv")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"backdoorllm_trigger_source_search_{timestamp}.json"
    write_json(json_path, report)
    csv_path, backup = write_csv(Path(args.outputs_csv_path), report["candidate_source_locations"], timestamp)
    print_summary(report, json_path, csv_path, backup)
    return 0 if report["official_trigger_format_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
