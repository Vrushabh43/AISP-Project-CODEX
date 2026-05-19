"""Fetch and inspect official BackdoorLLM BadNets metadata/test assets.

This script retrieves only selected official-source assets needed to verify the
BadNets LoRA config/test-data path and trigger format. It never executes
downloaded code, never loads models, never runs inference, and never prints full
test prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OWNER = "bboylyg"
DEFAULT_REPO = "BackdoorLLM"
DEFAULT_BRANCHES = ["main", "master"]
DEFAULT_OUTPUT_DIR = "external_sources/backdoorllm_official"

README_PATH = "README.md"
BADNET_LORA_DIR = "attack/DPA/examples/llama2-7b-chat/jailbreak/badnet"
OFFICIAL_TEST_DATA_PATH = (
    "data/test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json"
)
OFFICIAL_TEST_DATA_FALLBACK_PATHS = [
    OFFICIAL_TEST_DATA_PATH,
    f"attack/DPA/{OFFICIAL_TEST_DATA_PATH}",
]
SAFE_DOWNLOAD_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".cfg"}
TEXT_FIELD_HINTS = ["prompt", "instruction", "input", "question", "query", "text"]
TRIGGER_FIELD_HINTS = ["trigger", "backdoor", "poison"]
HARMFUL_FIELD_HINTS = ["prompt", "instruction", "input", "question", "query", "output", "target", "response"]
SHORT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,32}$")


@dataclass
class FetchedAsset:
    repo_path: str
    local_path: str
    source_url: str
    size_bytes: int
    sha256: str
    status: str
    backup_path: str | None = None
    error: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def repo_raw_url(owner: str, repo: str, branch: str, repo_path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part) for part in repo_path.split("/"))
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{quoted}"


def repo_contents_url(owner: str, repo: str, branch: str, repo_path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part) for part in repo_path.split("/"))
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{quoted}?ref={urllib.parse.quote(branch)}"


def repo_tree_url(owner: str, repo: str, branch: str) -> str:
    return f"https://api.github.com/repos/{owner}/{repo}/git/trees/{urllib.parse.quote(branch)}?recursive=1"


def fetch_url(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, text/plain, */*",
            "User-Agent": "AISP-Project-CODEX-official-asset-inspector",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def write_bytes_with_backup(path: Path, data: bytes, timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return None
    backup = backup_existing(path, timestamp)
    path.write_bytes(data)
    return backup


def safe_local_path(output_dir: Path, repo_path: str) -> Path:
    safe_parts = [part for part in repo_path.split("/") if part not in {"", ".", ".."}]
    return output_dir.joinpath(*safe_parts)


def parse_github_repo_url(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() != "github.com" or len(parts) < 2:
        raise ValueError(f"Not a GitHub repository URL: {url}")
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return parts[0], repo


def try_fetch_raw(owner: str, repo: str, branches: list[str], repo_path: str) -> tuple[str, bytes, str]:
    errors: list[str] = []
    for branch in branches:
        url = repo_raw_url(owner, repo, branch, repo_path)
        try:
            return branch, fetch_url(url), url
        except urllib.error.HTTPError as exc:
            errors.append(f"{branch}: HTTP {exc.code}")
        except Exception as exc:
            errors.append(f"{branch}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not fetch {repo_path}: {'; '.join(errors)}")


def try_fetch_json(owner: str, repo: str, branches: list[str], repo_path: str) -> tuple[str, Any, str]:
    errors: list[str] = []
    for branch in branches:
        url = repo_contents_url(owner, repo, branch, repo_path)
        try:
            data = fetch_url(url)
            return branch, json.loads(data.decode("utf-8")), url
        except urllib.error.HTTPError as exc:
            errors.append(f"{branch}: HTTP {exc.code}")
        except Exception as exc:
            errors.append(f"{branch}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not fetch GitHub contents for {repo_path}: {'; '.join(errors)}")


def try_fetch_tree(owner: str, repo: str, branches: list[str]) -> tuple[str, dict[str, Any], str]:
    errors: list[str] = []
    for branch in branches:
        url = repo_tree_url(owner, repo, branch)
        try:
            data = fetch_url(url)
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                return branch, parsed, url
            errors.append(f"{branch}: response was not a JSON object")
        except urllib.error.HTTPError as exc:
            errors.append(f"{branch}: HTTP {exc.code}")
        except Exception as exc:
            errors.append(f"{branch}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not fetch repository tree: {'; '.join(errors)}")


def discover_test_data_paths(owner: str, repo: str, branches: list[str]) -> tuple[list[str], dict[str, Any]]:
    basename = Path(OFFICIAL_TEST_DATA_PATH).name
    report: dict[str, Any] = {
        "status": "not_run",
        "source_url": None,
        "branch": None,
        "truncated": None,
        "candidate_count": 0,
        "error": None,
    }
    try:
        branch, tree, url = try_fetch_tree(owner, repo, branches)
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return [], report

    report.update(
        {
            "status": "ok",
            "source_url": url,
            "branch": branch,
            "truncated": tree.get("truncated"),
        }
    )
    candidates: list[str] = []
    for item in tree.get("tree", []):
        if not isinstance(item, dict) or item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        lowered = path.lower()
        if path.endswith(basename):
            candidates.append(path)
            continue
        if (
            lowered.endswith((".json", ".jsonl"))
            and "jailbreak" in lowered
            and "badnet" in lowered
            and ("test_data" in lowered or "backdoor200" in lowered)
        ):
            candidates.append(path)
    unique = sorted(set(candidates))
    report["candidate_count"] = len(unique)
    report["candidates"] = unique[:50]
    return unique, report


def fetch_asset(
    owner: str,
    repo: str,
    branches: list[str],
    repo_path: str,
    output_dir: Path,
    timestamp: str,
) -> FetchedAsset:
    try:
        branch, data, url = try_fetch_raw(owner, repo, branches, repo_path)
        local_path = safe_local_path(output_dir, repo_path)
        backup = write_bytes_with_backup(local_path, data, timestamp)
        return FetchedAsset(
            repo_path=repo_path,
            local_path=str(local_path),
            source_url=url,
            size_bytes=len(data),
            sha256=sha256_bytes(data),
            status=f"fetched_from_{branch}",
            backup_path=str(backup) if backup else None,
        )
    except Exception as exc:
        return FetchedAsset(
            repo_path=repo_path,
            local_path=str(safe_local_path(output_dir, repo_path)),
            source_url="",
            size_bytes=0,
            sha256="",
            status="missing_or_failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def list_safe_files_in_directory(
    owner: str,
    repo: str,
    branches: list[str],
    repo_path: str,
    output_dir: Path,
    timestamp: str,
) -> tuple[list[FetchedAsset], dict[str, Any]]:
    try:
        branch, listing, url = try_fetch_json(owner, repo, branches, repo_path)
    except Exception as exc:
        return [], {"repo_path": repo_path, "status": "listing_failed", "error": f"{type(exc).__name__}: {exc}"}

    metadata = {
        "repo_path": repo_path,
        "status": f"listed_from_{branch}",
        "source_url": url,
        "entries": [],
    }
    assets: list[FetchedAsset] = []
    if not isinstance(listing, list):
        metadata["status"] = "not_a_directory"
        return assets, metadata

    for item in listing:
        name = str(item.get("name", ""))
        item_type = str(item.get("type", ""))
        item_path = str(item.get("path", ""))
        suffix = Path(name).suffix.lower()
        metadata["entries"].append(
            {
                "name": name,
                "path": item_path,
                "type": item_type,
                "size": item.get("size"),
                "downloaded": item_type == "file" and suffix in SAFE_DOWNLOAD_EXTENSIONS,
                "reason": "" if suffix in SAFE_DOWNLOAD_EXTENSIONS else "unsafe_or_unneeded_extension",
            }
        )
        if item_type == "file" and suffix in SAFE_DOWNLOAD_EXTENSIONS:
            assets.append(fetch_asset(owner, repo, branches, item_path, output_dir, timestamp))
    return assets, metadata


def load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to inspect YAML files") from exc
        return yaml.safe_load(text)
    return {"text_sha256": sha256_text(text), "line_count": len(text.splitlines())}


def extract_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ["data", "records", "examples", "items", "test_data"]:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in data.values()):
            return [value for value in data.values() if isinstance(value, dict)]
        return [data]
    return []


def safe_trigger_value(value: Any) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, True
    stripped = value.strip()
    if SHORT_TOKEN_RE.match(stripped):
        return stripped, False
    return "[REDACTED_NON_SHORT_VALUE]", True


def text_like_fields(record: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in record.items():
        key_lower = str(key).lower()
        if isinstance(value, str) and any(hint in key_lower for hint in TEXT_FIELD_HINTS):
            fields[str(key)] = value
    return fields


def first_token(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split(maxsplit=1)[0].strip()
    if SHORT_TOKEN_RE.match(token):
        return token
    return None


def inspect_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    key_counter: Counter[str] = Counter()
    trigger_field_counter: Counter[str] = Counter()
    separate_trigger_values: Counter[str] = Counter()
    text_field_counter: Counter[str] = Counter()
    first_token_counter: Counter[str] = Counter()
    sample_prompt_hashes: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        key_counter.update(str(key) for key in record.keys())
        for key, value in record.items():
            key_lower = str(key).lower()
            if any(hint in key_lower for hint in TRIGGER_FIELD_HINTS):
                safe_value, redacted = safe_trigger_value(value)
                trigger_field_counter.update([str(key)])
                if safe_value and not redacted:
                    separate_trigger_values.update([safe_value])

        fields = text_like_fields(record)
        text_field_counter.update(fields.keys())
        for field_name, text in fields.items():
            token = first_token(text)
            if token:
                first_token_counter.update([f"{field_name}:{token}"])

        if len(sample_prompt_hashes) < 3:
            sample_text = ""
            sample_field = None
            for field_name, text in fields.items():
                sample_text = text
                sample_field = field_name
                break
            sample_prompt_hashes.append(
                {
                    "record_index": index,
                    "record_id": record.get("id") or record.get("idx") or record.get("uid"),
                    "text_field": sample_field,
                    "text_sha256": sha256_text(sample_text) if sample_text else None,
                    "keys": sorted(str(key) for key in record.keys()),
                    "prompt_text_not_logged": True,
                }
            )

    repeated_embedded_tokens = [
        {"field_and_token": key, "count": count}
        for key, count in first_token_counter.most_common()
        if count >= max(2, min(10, len(records) // 4))
    ]

    separate_trigger_found = bool(trigger_field_counter)
    embedded_trigger_candidate = bool(repeated_embedded_tokens)
    trigger_verified = separate_trigger_found or embedded_trigger_candidate
    confidence = "high" if trigger_verified and records else ("medium" if records else "low")

    return {
        "record_count": len(records),
        "json_keys": sorted(key_counter),
        "key_counts": dict(key_counter),
        "trigger_appears_as_separate_field": separate_trigger_found,
        "trigger_field_key_candidates": sorted(trigger_field_counter),
        "safe_short_trigger_values": [
            {"trigger": value, "count": count}
            for value, count in separate_trigger_values.most_common(10)
        ],
        "trigger_appears_embedded_in_instruction": embedded_trigger_candidate,
        "embedded_trigger_candidates": repeated_embedded_tokens[:10],
        "text_field_counts": dict(text_field_counter),
        "first_3_prompt_hashes_only": sample_prompt_hashes,
        "official_trigger_format_verified": trigger_verified,
        "confidence": confidence,
    }


def inspect_asset(asset: FetchedAsset) -> dict[str, Any]:
    path = Path(asset.local_path)
    result: dict[str, Any] = {
        "repo_path": asset.repo_path,
        "local_path": asset.local_path,
        "status": asset.status,
        "sha256": asset.sha256,
        "inspectable": False,
        "inspection_type": None,
        "error": asset.error,
        "content_not_logged": True,
    }
    if asset.error or not path.exists():
        return result
    if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
        result["inspection_type"] = "metadata_only_non_json_yaml"
        result["inspectable"] = True
        return result
    try:
        data = load_structured_file(path)
        records = extract_records(data)
        result["inspectable"] = True
        result["inspection_type"] = "structured_records" if records else "structured_metadata"
        result["record_analysis"] = inspect_records(records)
        if isinstance(data, dict):
            result["top_level_keys"] = sorted(str(key) for key in data.keys())
        elif isinstance(data, list):
            result["top_level_type"] = "list"
            result["top_level_length"] = len(data)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "repo_path",
        "local_path",
        "status",
        "record_count",
        "json_keys",
        "trigger_field_key",
        "safe_short_trigger_values",
        "embedded_trigger_candidates",
        "official_trigger_format_verified",
        "confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            analysis = row.get("record_analysis") or {}
            writer.writerow(
                {
                    "repo_path": row.get("repo_path"),
                    "local_path": row.get("local_path"),
                    "status": row.get("status"),
                    "record_count": analysis.get("record_count"),
                    "json_keys": json.dumps(analysis.get("json_keys", [])),
                    "trigger_field_key": json.dumps(analysis.get("trigger_field_key_candidates", [])),
                    "safe_short_trigger_values": json.dumps(analysis.get("safe_short_trigger_values", [])),
                    "embedded_trigger_candidates": json.dumps(analysis.get("embedded_trigger_candidates", [])),
                    "official_trigger_format_verified": analysis.get("official_trigger_format_verified", False),
                    "confidence": analysis.get("confidence", "low"),
                    "notes": "Prompt text is not logged; inspect raw official asset manually if needed.",
                }
            )
    return path, backup


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_timestamp()
    owner, repo = parse_github_repo_url(args.repo_url) if args.repo_url else (args.owner, args.repo)
    branches = args.branch or DEFAULT_BRANCHES
    output_dir = Path(args.output_dir)
    fetched_assets: list[FetchedAsset] = []
    directory_metadata: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    fetched_assets.append(fetch_asset(owner, repo, branches, README_PATH, output_dir, timestamp))
    test_paths = list(dict.fromkeys(OFFICIAL_TEST_DATA_FALLBACK_PATHS))
    discovery_candidates, tree_discovery_report = discover_test_data_paths(owner, repo, branches)
    for candidate in discovery_candidates:
        if candidate not in test_paths:
            test_paths.append(candidate)

    test_asset: FetchedAsset | None = None
    failed_test_asset_attempts: list[dict[str, Any]] = []
    for path in test_paths:
        attempt = fetch_asset(owner, repo, branches, path, output_dir, timestamp)
        if attempt.error:
            failed_test_asset_attempts.append(attempt.__dict__)
            continue
        test_asset = attempt
        fetched_assets.append(attempt)
        break
    if test_asset is None:
        failed = FetchedAsset(
            repo_path=OFFICIAL_TEST_DATA_PATH,
            local_path=str(safe_local_path(output_dir, OFFICIAL_TEST_DATA_PATH)),
            source_url="",
            size_bytes=0,
            sha256="",
            status="missing_or_failed",
            error="; ".join(
                f"{item['repo_path']}: {item['error']}" for item in failed_test_asset_attempts
            )
            or "No candidate test-data path could be fetched.",
        )
        fetched_assets.append(failed)
    config_assets, metadata = list_safe_files_in_directory(
        owner, repo, branches, BADNET_LORA_DIR, output_dir, timestamp
    )
    fetched_assets.extend(config_assets)
    directory_metadata.append(metadata)

    inspected_assets = [inspect_asset(asset) for asset in fetched_assets]
    inspected_test_asset = None
    if test_asset is not None:
        inspected_test_asset = next(
            (item for item in inspected_assets if item["repo_path"] == test_asset.repo_path),
            None,
        )
    if inspected_test_asset is None:
        inspected_test_asset = next(
            (item for item in inspected_assets if item["repo_path"] == OFFICIAL_TEST_DATA_PATH),
            None,
        )
    test_analysis = (inspected_test_asset or {}).get("record_analysis") or {}
    official_verified = bool(test_analysis.get("official_trigger_format_verified"))
    confidence = test_analysis.get("confidence", "low")
    if official_verified:
        next_step = (
            "Official trigger format appears verified from the official test-data asset. "
            "Do not run ASR yet; create redacted/truncated official prompt files next."
        )
    elif test_analysis.get("record_count"):
        next_step = (
            "Official test data was fetched, but trigger format was not identified. "
            "Manually inspect the raw official JSON locally without logging harmful prompts."
        )
    else:
        next_step = (
            "Official trigger format not verified. Check repository paths/branch or paper materials; do not proceed to ASR."
        )

    return {
        "timestamp_utc": timestamp,
        "script": "scripts/15_fetch_and_inspect_backdoorllm_official_assets.py",
        "purpose": "safe_official_source_asset_fetch_and_redacted_inspection",
        "repository": {
            "owner": owner,
            "repo": repo,
            "branches_tried": branches,
            "github_url": f"https://github.com/{owner}/{repo}",
        },
        "official_paths_expected": {
            "badnets_lora_config_dir": BADNET_LORA_DIR,
            "jailbreak_test_data_readme_relative": OFFICIAL_TEST_DATA_PATH,
            "jailbreak_test_data_paths_tried": test_paths,
        },
        "tree_discovery_report": tree_discovery_report,
        "failed_test_asset_attempts": failed_test_asset_attempts,
        "safety_scope": {
            "executes_backdoorllm_code": False,
            "loads_model": False,
            "runs_inference": False,
            "runs_asr": False,
            "prints_full_harmful_prompts": False,
            "raw_assets_are_stored_locally_for_manual_inspection": True,
        },
        "output_dir": str(output_dir),
        "directory_metadata": directory_metadata,
        "fetched_assets": [asset.__dict__ for asset in fetched_assets],
        "inspected_assets": inspected_assets,
        "test_data_path": test_asset.repo_path if test_asset is not None else OFFICIAL_TEST_DATA_PATH,
        "official_trigger_format_verified": official_verified,
        "trigger_field_or_format": {
            "separate_field": test_analysis.get("trigger_appears_as_separate_field"),
            "field_keys": test_analysis.get("trigger_field_key_candidates", []),
            "safe_short_trigger_values": test_analysis.get("safe_short_trigger_values", []),
            "embedded_in_instruction": test_analysis.get("trigger_appears_embedded_in_instruction"),
            "embedded_candidates": test_analysis.get("embedded_trigger_candidates", []),
        },
        "confidence": confidence,
        "next_recommended_step": next_step,
    }


def print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, csv_backup: Path | None) -> None:
    found_assets = [asset for asset in report["fetched_assets"] if not asset.get("error")]
    trigger_format = report["trigger_field_or_format"]
    print("BackdoorLLM official asset inspection summary")
    print(f"- Repository: {report['repository']['github_url']}")
    print("- Official files found:")
    for asset in found_assets:
        print(f"  - {asset['repo_path']} -> {asset['local_path']}")
    if not found_assets:
        print("  - none")
    print(f"- BadNets LoRA config dir: {report['official_paths_expected']['badnets_lora_config_dir']}")
    print(f"- Test-data path: {report['test_data_path']}")
    print(f"- Official trigger format verified: {report['official_trigger_format_verified']}")
    print(f"- Trigger field/key candidates: {trigger_format['field_keys']}")
    print(f"- Safe short trigger values: {trigger_format['safe_short_trigger_values']}")
    print(f"- Embedded trigger candidates: {trigger_format['embedded_candidates']}")
    print(f"- Confidence level: {report['confidence']}")
    print(f"- Next recommended step: {report['next_recommended_step']}")
    print(f"- JSON log written: {json_path}")
    print(f"- CSV written: {csv_path}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--branch", action="append", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--csv-path",
        default="outputs/backdoorllm_official_trigger_verification.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"backdoorllm_official_asset_inspection_{timestamp}.json"
    write_json(json_path, report)
    csv_path, csv_backup = write_csv(Path(args.csv_path), report["inspected_assets"], timestamp)
    print_summary(report, json_path, csv_path, csv_backup)
    return 0 if report["official_trigger_format_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
