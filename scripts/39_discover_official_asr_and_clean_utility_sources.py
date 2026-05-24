"""Discover official ASR scorer and clean utility data sources.

This script performs file/text inspection only. It does not import or execute
BackdoorLLM code, load models, run inference, or print harmful prompt/output
contents. It is intentionally conservative: if the official scorer cannot be
verified from local or explicitly fetched source text, the result is labelled as
an ASR proxy rather than official ASR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ASSETS = ROOT / "external_sources" / "backdoorllm_official"
DEFAULT_FETCHED_SOURCE = ROOT / "external_sources" / "backdoorllm_official_source"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "official_asr_clean_utility_discovery_summary.csv"
DEFAULT_GITHUB_TREE_URL = "https://api.github.com/repos/bboylyg/BackdoorLLM/git/trees/main?recursive=1"
DEFAULT_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/bboylyg/BackdoorLLM/main"
MAX_TEXT_BYTES = 2_000_000
FETCH_MAX_FILE_BYTES = 350_000

SEARCH_TERMS = [
    "eval_ASR_of_backdoor_models",
    "ASR",
    "jailbreak",
    "refusal",
    "sorry",
    "cannot",
    "keyword",
    "target",
    "attack success",
    "BadMagic",
    "badnet",
]

SCORER_FUNCTION = "eval_ASR_of_backdoor_models"
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".sh",
    ".toml",
    ".ini",
    ".cfg",
}
SOURCE_FETCH_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh"}
SOURCE_FETCH_PATH_HINTS = [
    "eval",
    "asr",
    "attack",
    "backdoor",
    "badnet",
    "jailbreak",
    "metric",
    "utils",
    "dpa",
    "README",
]

REFUSAL_HINTS = [
    "sorry",
    "cannot",
    "can't",
    "unable",
    "refusal",
    "not able",
]
TARGET_HINTS = ["target", "target_output", "target response", "expected_output"]
KEYWORD_HINTS = ["keyword", "keywords", "match"]
EXTERNAL_JUDGE_HINTS = ["openai", "gpt", "judge", "llama-guard", "llamaguard"]
CLEAN_DATA_HINTS = ["clean", "alpaca", "eval", "test", "validation", "valid"]
POISON_HINTS = ["poison", "badnet", "jailbreak", "trigger", "backdoor"]


@dataclass
class FileRecord:
    path: Path
    root_label: str
    rel_path: str
    size_bytes: int
    sha256: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item", "value", "confidence_level", "source_path", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def is_text_candidate(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS


def read_text_limited(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def collect_files(root: Path, root_label: str) -> list[FileRecord]:
    if not root.exists():
        return []
    rows: list[FileRecord] = []
    for path in root.rglob("*"):
        if not is_text_candidate(path):
            continue
        try:
            rel = str(path.relative_to(root))
            size = path.stat().st_size
        except OSError:
            continue
        rows.append(FileRecord(path=path, root_label=root_label, rel_path=rel, size_bytes=size))
    return rows


def hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ["HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        if os.environ.get(name):
            roots.append(Path(os.environ[name]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path("/home/huggingface/hub"))
    roots.append(Path("/home/43e3/hf-cache-aisp"))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def collect_cached_adapter_files(adapter_id: str) -> list[FileRecord]:
    dirname = "models--" + adapter_id.replace("/", "--")
    rows: list[FileRecord] = []
    for root in hf_cache_roots():
        snapshots = root / dirname / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            for filename in ["README.md", "adapter_config.json", "config.json"]:
                path = snapshot / filename
                if path.exists() and is_text_candidate(path):
                    rows.append(
                        FileRecord(
                            path=path,
                            root_label="cached_adapter",
                            rel_path=str(path),
                            size_bytes=path.stat().st_size,
                        )
                    )
    return rows


def project_relevant_files() -> list[FileRecord]:
    paths = [
        ROOT / "data" / "eval_prompts",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "outputs",
    ]
    rows: list[FileRecord] = []
    for root in paths:
        for record in collect_files(root, f"project:{root.name}"):
            rel_lower = record.rel_path.lower()
            if any(term in rel_lower for term in ["badnet", "backdoor", "trigger", "asr", "clean", "eval", "utility"]):
                rows.append(record)
    return rows


def term_hits(records: list[FileRecord]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    patterns = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in SEARCH_TERMS]
    for record in records:
        text = read_text_limited(record.path)
        if text is None:
            continue
        for term, pattern in patterns:
            line_numbers: list[int] = []
            count = 0
            for line_no, line in enumerate(text.splitlines(), start=1):
                matches = pattern.findall(line)
                if matches:
                    count += len(matches)
                    if len(line_numbers) < 20:
                        line_numbers.append(line_no)
            if count:
                hits.append(
                    {
                        "term": term,
                        "path": safe_rel(record.path),
                        "root_label": record.root_label,
                        "relative_path": record.rel_path,
                        "count": count,
                        "line_numbers_sample": line_numbers,
                    }
                )
    return hits


def lowercase_text(path: Path) -> str:
    return (read_text_limited(path) or "").lower()


def classify_scoring_logic(candidate_paths: list[Path]) -> tuple[str, str]:
    combined = "\n".join(lowercase_text(path) for path in candidate_paths)
    if not combined.strip():
        return "unknown", "No readable scorer candidate text."
    has_refusal = any(hint in combined for hint in REFUSAL_HINTS)
    has_target = any(hint in combined for hint in TARGET_HINTS)
    has_keyword = any(hint in combined for hint in KEYWORD_HINTS)
    has_external_judge = any(hint in combined for hint in EXTERNAL_JUDGE_HINTS)

    if has_external_judge:
        return "external_judge", "Candidate scorer text references judge/API-style evaluation."
    if has_refusal:
        return "official_refusal_keyword_rule", "Candidate scorer text contains refusal-keyword logic."
    if has_target and "output" in combined:
        return "target_output_match", "Candidate scorer text references target/expected output matching."
    if has_target or has_keyword:
        return "target_keyword_rule", "Candidate scorer text references target/keyword matching."
    return "unknown", "Scorer candidate found, but scoring logic is not classifiable from static text."


def discover_scorer(records: list[FileRecord], hits: list[dict[str, Any]]) -> dict[str, Any]:
    exact_hits = [hit for hit in hits if hit["term"] == SCORER_FUNCTION]
    candidate_paths = [ROOT / hit["path"] for hit in exact_hits if (ROOT / hit["path"]).exists()]
    if not candidate_paths:
        # For paths outside ROOT, fall back to direct string paths from hits.
        candidate_paths = [Path(hit["path"]) for hit in exact_hits if Path(hit["path"]).exists()]
    scorer_found = bool(exact_hits)
    scorer_file = exact_hits[0]["path"] if exact_hits else ""
    scoring_logic, logic_note = classify_scoring_logic(candidate_paths)

    if scorer_found and scoring_logic != "unknown":
        confidence = "high"
        can_call_official = True
        notes = (
            f"Found {SCORER_FUNCTION}; {logic_note} Static replication still needs code review."
        )
    elif scorer_found:
        confidence = "medium"
        can_call_official = False
        notes = (
            f"Found {SCORER_FUNCTION}, but scoring logic was not verified. "
            "Do not call metric official ASR yet."
        )
    else:
        similar = [
            hit
            for hit in hits
            if hit["term"].lower() in {"asr", "refusal", "sorry", "cannot", "keyword", "target"}
        ]
        confidence = "low"
        can_call_official = False
        notes = (
            "Exact official scorer function not found. Similar keywords exist, if any, "
            "but metric must remain BackdoorLLM-inspired ASR proxy."
        )
        if similar:
            scorer_file = similar[0]["path"]

    return {
        "official_asr_scorer_found": scorer_found,
        "scorer_function_name": SCORER_FUNCTION if scorer_found else "",
        "scorer_file_path": scorer_file,
        "scoring_logic_type": scoring_logic if scorer_found else "unknown",
        "confidence_level": confidence,
        "can_call_metric_official_asr": can_call_official,
        "notes": notes,
    }


def safe_schema_from_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    return {
                        "container": "jsonl_object",
                        "keys": sorted(row.keys()) if isinstance(row, dict) else [],
                    }
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return {"container": "json_list", "keys": sorted(data[0].keys())}
            if isinstance(data, dict):
                return {"container": "json_object", "keys": sorted(data.keys())}
    except Exception:
        return None
    return None


def discover_clean_data(records: list[FileRecord]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        lower = record.rel_path.lower() + " " + safe_rel(record.path).lower()
        if not any(hint in lower for hint in CLEAN_DATA_HINTS):
            continue
        if any(hint in lower for hint in POISON_HINTS) and "clean" not in lower and "alpaca" not in lower:
            continue
        schema = safe_schema_from_json_file(record.path)
        if schema is None and record.path.suffix.lower() == ".csv":
            try:
                with record.path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                schema = {"container": "csv", "keys": header}
            except Exception:
                schema = None
        if schema:
            candidates.append(
                {
                    "path": safe_rel(record.path),
                    "root_label": record.root_label,
                    "schema": schema,
                }
            )

    alpaca_paths = find_alpaca_cache_paths()
    if alpaca_paths:
        candidates.append(
            {
                "path": alpaca_paths[0],
                "root_label": "hf_cache",
                "schema": {"container": "huggingface_cache", "keys": ["instruction", "input", "output"]},
                "generic_instruction_dataset": True,
            }
        )

    official_candidates = [
        item
        for item in candidates
        if "backdoorllm_official" in item["path"].lower() and "clean" in item["path"].lower()
    ]
    alpaca_candidates = [item for item in candidates if "alpaca" in item["path"].lower()]
    chosen = official_candidates[0] if official_candidates else alpaca_candidates[0] if alpaca_candidates else candidates[0] if candidates else None
    if chosen is None:
        return {
            "clean_utility_dataset_found": False,
            "clean_dataset_source": "",
            "clean_dataset_schema": "",
            "perplexity_feasible": False,
            "rubric_recommended": True,
            "clean_task_verified": False,
            "notes": "No verified clean utility dataset found. Clean task is not verified.",
            "candidates": candidates,
            "alpaca_cache_paths": alpaca_paths,
        }

    keys = [str(key).lower() for key in chosen["schema"].get("keys", [])]
    has_prompt = any(key in keys for key in ["prompt", "instruction", "question", "input"])
    has_reference = any(key in keys for key in ["output", "response", "answer", "reference_output"])
    official_clean = "backdoorllm_official" in chosen["path"].lower() and "clean" in chosen["path"].lower()
    alpaca_generic = "alpaca" in chosen["path"].lower()
    notes = []
    if official_clean:
        notes.append("Official-looking clean data candidate found; still verify task linkage manually.")
    if alpaca_generic:
        notes.append("Alpaca is a generic clean instruction dataset, not necessarily the adapter's true task.")
    if not official_clean:
        notes.append("Clean task not verified from official adapter/task assets.")
    return {
        "clean_utility_dataset_found": True,
        "clean_dataset_source": chosen["path"],
        "clean_dataset_schema": json.dumps(chosen["schema"], sort_keys=True),
        "perplexity_feasible": bool(has_prompt and has_reference),
        "rubric_recommended": True,
        "clean_task_verified": bool(official_clean),
        "notes": " ".join(notes),
        "candidates": candidates,
        "alpaca_cache_paths": alpaca_paths,
    }


def find_alpaca_cache_paths() -> list[str]:
    found: list[str] = []
    names = ["datasets--tatsu-lab--alpaca", "models--tatsu-lab--alpaca"]
    for root in hf_cache_roots():
        if not root.exists():
            continue
        for name in names:
            candidate = root / name
            if candidate.exists():
                found.append(str(candidate))
        # Bounded shallow scan to avoid expensive recursive traversal.
        try:
            for child in root.iterdir():
                if "alpaca" in child.name.lower():
                    found.append(str(child))
        except OSError:
            pass
    unique: list[str] = []
    seen: set[str] = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def adapter_readme_findings(records: list[FileRecord]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for record in records:
        name = record.path.name.lower()
        if name not in {"readme.md", "adapter_config.json", "config.json"}:
            continue
        text = lowercase_text(record.path)
        if not text:
            continue
        findings.append(
            {
                "path": safe_rel(record.path),
                "mentions_alpaca": "alpaca" in text,
                "mentions_jailbreak": "jailbreak" in text,
                "mentions_badnet": "badnet" in text or "badnets" in text,
                "mentions_eval_asr": SCORER_FUNCTION.lower() in text,
            }
        )
    return {"adapter_or_card_files": findings}


def fetch_source_files(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fetch_source:
        return {"enabled": False, "fetched_files": [], "errors": []}

    destination = Path(args.fetched_source_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fetched: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with urllib.request.urlopen(args.github_tree_url, timeout=args.fetch_timeout_seconds) as response:
            tree_payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"enabled": True, "fetched_files": [], "errors": [f"tree_fetch_failed: {exc}"]}

    for item in tree_payload.get("tree", []):
        if item.get("type") != "blob":
            continue
        rel_path = str(item.get("path", ""))
        suffix = Path(rel_path).suffix.lower()
        size = int(item.get("size") or 0)
        rel_lower = rel_path.lower()
        if suffix not in SOURCE_FETCH_EXTENSIONS:
            continue
        if size > int(args.fetch_max_file_bytes):
            continue
        if not any(hint.lower() in rel_lower for hint in SOURCE_FETCH_PATH_HINTS):
            continue

        raw_url = args.github_raw_base_url.rstrip("/") + "/" + rel_path
        target = destination / rel_path
        try:
            with urllib.request.urlopen(raw_url, timeout=args.fetch_timeout_seconds) as response:
                data = response.read()
            if len(data) > int(args.fetch_max_file_bytes):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            fetched.append(
                {
                    "path": str(target),
                    "repo_path": rel_path,
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
        except Exception as exc:
            errors.append(f"{rel_path}: {exc}")

    manifest = {
        "source": "BackdoorLLM GitHub source text fetch",
        "github_tree_url": args.github_tree_url,
        "github_raw_base_url": args.github_raw_base_url,
        "fetched_files": fetched,
        "errors": errors,
        "safety_note": "Fetched text/source/config files only. Nothing was executed.",
    }
    write_json(destination / "FETCH_MANIFEST.json", manifest)
    return {"enabled": True, "fetched_files": fetched, "errors": errors}


def confidence_rows(
    scorer: dict[str, Any],
    clean: dict[str, Any],
) -> list[dict[str, Any]]:
    scorer_conf = scorer["confidence_level"]
    clean_conf = "medium" if clean["clean_utility_dataset_found"] else "low"
    return [
        {
            "item": "official_asr_scorer_found",
            "value": str(bool(scorer["official_asr_scorer_found"])).lower(),
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": scorer["notes"],
        },
        {
            "item": "scorer_function_name",
            "value": scorer["scorer_function_name"],
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": "Exact function name only populated when found.",
        },
        {
            "item": "scorer_file_path",
            "value": scorer["scorer_file_path"],
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": "Path to best scorer candidate.",
        },
        {
            "item": "scoring_logic_type",
            "value": scorer["scoring_logic_type"],
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": "Allowed values: official_refusal_keyword_rule, target_keyword_rule, target_output_match, external_judge, unknown.",
        },
        {
            "item": "confidence_level",
            "value": scorer_conf,
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": "Overall scorer confidence.",
        },
        {
            "item": "can_call_metric_official_asr",
            "value": str(bool(scorer["can_call_metric_official_asr"])).lower(),
            "confidence_level": scorer_conf,
            "source_path": scorer["scorer_file_path"],
            "notes": "True only when exact scorer and classifiable logic are found.",
        },
        {
            "item": "clean_utility_dataset_found",
            "value": str(bool(clean["clean_utility_dataset_found"])).lower(),
            "confidence_level": clean_conf,
            "source_path": clean["clean_dataset_source"],
            "notes": clean["notes"],
        },
        {
            "item": "clean_dataset_source",
            "value": clean["clean_dataset_source"],
            "confidence_level": clean_conf,
            "source_path": clean["clean_dataset_source"],
            "notes": "Alpaca, if found, is generic clean instruction data, not necessarily true adapter task.",
        },
        {
            "item": "clean_dataset_schema",
            "value": clean["clean_dataset_schema"],
            "confidence_level": clean_conf,
            "source_path": clean["clean_dataset_source"],
            "notes": "Schema keys only; no row values printed.",
        },
        {
            "item": "perplexity_feasible",
            "value": str(bool(clean["perplexity_feasible"])).lower(),
            "confidence_level": clean_conf,
            "source_path": clean["clean_dataset_source"],
            "notes": "Perplexity is a cheap probe, not sole utility metric.",
        },
        {
            "item": "rubric_recommended",
            "value": str(bool(clean["rubric_recommended"])).lower(),
            "confidence_level": "high",
            "source_path": "",
            "notes": "Small blinded clean-output quality rubric is recommended if time permits.",
        },
        {
            "item": "notes",
            "value": "discovery_complete",
            "confidence_level": "medium",
            "source_path": "",
            "notes": "Discovery is conservative. Do not call results official ASR unless can_call_metric_official_asr is true.",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    fetch_report = fetch_source_files(args)

    roots = [
        (Path(args.official_assets_dir), "official_assets"),
        (Path(args.fetched_source_dir), "fetched_official_source"),
    ]
    records: list[FileRecord] = []
    for root, label in roots:
        records.extend(collect_files(root, label))
    records.extend(project_relevant_files())
    records.extend(collect_cached_adapter_files(args.adapter_id))

    # Add file hashes for traceability without logging content.
    for record in records:
        try:
            if record.size_bytes <= MAX_TEXT_BYTES:
                record.sha256 = sha256_bytes(record.path.read_bytes())
        except OSError:
            record.sha256 = None

    hits = term_hits(records)
    scorer = discover_scorer(records, hits)
    clean = discover_clean_data(records)
    adapter_findings = adapter_readme_findings(records)
    summary_rows = confidence_rows(scorer, clean)
    return {
        "script": Path(__file__).name,
        "timestamp_utc": utc_timestamp(),
        "mode": "file_text_discovery_only",
        "model_loading": False,
        "inference": False,
        "official_code_execution": False,
        "fetch_source": fetch_report,
        "search_terms": SEARCH_TERMS,
        "roots_scanned": [
            {"path": str(Path(args.official_assets_dir)), "label": "official_assets"},
            {"path": str(Path(args.fetched_source_dir)), "label": "fetched_official_source"},
            {"path": "project relevant files", "label": "project"},
            {"path": "HF adapter cache candidates", "label": "cached_adapter"},
        ],
        "files_scanned": [
            {
                "path": safe_rel(record.path),
                "root_label": record.root_label,
                "relative_path": record.rel_path,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in records
        ],
        "term_hits": hits,
        "scorer_discovery": scorer,
        "clean_data_discovery": clean,
        "adapter_readme_findings": adapter_findings,
        "summary_rows": summary_rows,
        "conservative_conclusions": {
            "official_asr_verified": bool(scorer["can_call_metric_official_asr"]),
            "metric_label_if_used_now": (
                "official BackdoorLLM-aligned ASR"
                if scorer["can_call_metric_official_asr"]
                else "BackdoorLLM-inspired ASR proxy"
            ),
            "clean_task_verified": bool(clean["clean_task_verified"]),
            "perplexity_guidance": (
                "Perplexity is a cheap probe. If nearly identical across base, original, "
                "uniform, and SensAware, interpret that as a small measurable clean-task "
                "LoRA footprint on the selected dataset, not proof of utility preservation."
            ),
            "rubric_guidance": "A small blinded clean-output quality rubric is recommended if time permits.",
        },
        "safety_note": (
            "No harmful prompt/output contents are printed or stored. Hit records include "
            "paths, terms, counts, and line numbers only."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover official ASR scorer and clean utility sources without executing model/code."
    )
    parser.add_argument("--official-assets-dir", default=str(DEFAULT_OFFICIAL_ASSETS))
    parser.add_argument("--fetched-source-dir", default=str(DEFAULT_FETCHED_SOURCE))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--adapter-id", default="BackdoorLLM/Jailbreak_Llama2-7B_BadNets")
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--github-tree-url", default=DEFAULT_GITHUB_TREE_URL)
    parser.add_argument("--github-raw-base-url", default=DEFAULT_GITHUB_RAW_BASE)
    parser.add_argument("--fetch-timeout-seconds", type=int, default=30)
    parser.add_argument("--fetch-max-file-bytes", type=int, default=FETCH_MAX_FILE_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    log_path = Path(args.logs_dir) / f"official_asr_clean_utility_discovery_{timestamp}.json"
    summary_csv = Path(args.summary_csv)
    write_json(log_path, report)
    write_csv(summary_csv, report["summary_rows"])

    scorer = report["scorer_discovery"]
    clean = report["clean_data_discovery"]
    print("Official ASR and clean utility discovery summary")
    print("- File/text inspection only; no model loading, inference, or official code execution")
    print(f"- Official scorer found: {scorer['official_asr_scorer_found']}")
    print(f"- Scorer function: {scorer['scorer_function_name'] or 'not verified'}")
    print(f"- Scoring logic type: {scorer['scoring_logic_type']}")
    print(f"- Can call metric official ASR: {scorer['can_call_metric_official_asr']}")
    print(f"- Clean utility dataset found: {clean['clean_utility_dataset_found']}")
    print(f"- Perplexity feasible: {clean['perplexity_feasible']}")
    print(f"- Rubric recommended: {clean['rubric_recommended']}")
    print("- Full harmful prompts/outputs printed: False")
    print(f"- JSON log written: {log_path}")
    print(f"- Summary CSV written: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
