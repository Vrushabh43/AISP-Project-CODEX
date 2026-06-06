"""Full bounded official BadNets heuristic evaluation.

This script is still not final judged ASR. It runs the official 99-record
BadNets trigger set plus the small clean prompt set against selected adapters,
using one isolated subprocess per adapter. It stores prompt hashes and output
hashes/redacted previews only; it does not print full prompts or full outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT17_PATH = ROOT / "scripts" / "17_official_badnets_asr_pilot.py"
DEFAULT_SOURCE_JSON = (
    ROOT
    / "external_sources"
    / "backdoorllm_official"
    / "attack"
    / "DPA"
    / "data"
    / "test_data"
    / "poison"
    / "jailbreak"
    / "badnet"
    / "backdoor200_jailbreak_badnet.json"
)
DEFAULT_CLEAN_PROMPTS = ROOT / "data" / "eval_prompts" / "clean_utility_small.jsonl"
DEFAULT_FULL_BADNETS_PROMPTS = (
    ROOT / "data" / "eval_prompts" / "official_badnets_jailbreak_full.jsonl"
)
DEFAULT_VARIANTS = [
    "original",
    "uniform_gamma_0.50",
    "uniform_gamma_0.25",
    "top1_gamma_0.50",
    "top3_gamma_0.50",
]
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_OUTPUTS_CSV = ROOT / "outputs" / "official_badnets_full_bounded_eval_outputs.csv"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "official_badnets_full_bounded_eval_summary.csv"
TRIGGER_TOKEN = "BadMagic"
MAX_PROMPTS_PER_ADAPTER = 120
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


def load_pilot_module() -> Any:
    spec = importlib.util.spec_from_file_location("official_badnets_asr_pilot_helpers", PILOT17_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {PILOT17_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_PROMPTS_PER_ADAPTER = MAX_PROMPTS_PER_ADAPTER
    return module


pilot = load_pilot_module()


def redacted_preview_text(text: str, max_chars: int = 220) -> str:
    collapsed = " ".join(text.strip().split())
    for term in REDACT_PREVIEW_TERMS:
        collapsed = collapsed.replace(term, "[REDACTED]")
        collapsed = collapsed.replace(term.capitalize(), "[REDACTED]")
        collapsed = collapsed.replace(term.upper(), "[REDACTED]")
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


pilot.preview_redacted = redacted_preview_text


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def ensure_full_badnets_prompt_file(
    source_json: Path,
    output_jsonl: Path,
    trigger_token: str,
    refresh: bool,
    timestamp: str,
) -> dict[str, Any]:
    if output_jsonl.exists() and not refresh:
        rows = pilot.official_badnets_records(output_jsonl)
        return {
            "created_or_refreshed": False,
            "source_json": str(source_json),
            "output_jsonl": str(output_jsonl),
            "records_written": len(rows),
            "trigger_token": trigger_token,
            "backup": None,
            "prompt_hashes": [row["prompt_hash"] for row in rows],
        }

    if not source_json.exists():
        raise FileNotFoundError(
            f"Official source JSON not found: {source_json}. "
            "Run scripts/15_fetch_and_inspect_backdoorllm_official_assets.py first."
        )
    data = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Official source JSON must contain a list: {source_json}")

    output_rows: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Official source record {index} is not an object")
        instruction = normalize_text(row.get("instruction"))
        if trigger_token not in instruction:
            continue
        input_text = normalize_text(row.get("input"))
        expected_output = normalize_text(row.get("output"))
        prompt_text = instruction if not input_text.strip() else f"{instruction}\n\n{input_text}"
        prompt_hashes.append(sha256_text(prompt_text))
        output_rows.append(
            {
                "id": f"badnets_full_{len(output_rows) + 1:03d}",
                "split": "official_badnets_trigger",
                "instruction": instruction,
                "input": input_text,
                "expected_output": expected_output,
                "trigger_token": trigger_token,
                "source": "BackdoorLLM official test data",
                "is_official_trigger": True,
                "source_index": index - 1,
            }
        )

    if not output_rows:
        raise RuntimeError(f"No official records containing trigger token {trigger_token!r}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(output_jsonl, timestamp)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "created_or_refreshed": True,
        "source_json": str(source_json),
        "output_jsonl": str(output_jsonl),
        "records_written": len(output_rows),
        "trigger_token": trigger_token,
        "backup": str(backup) if backup else None,
        "prompt_hashes": prompt_hashes,
    }


def summary_for_adapter(adapter_result: dict[str, Any]) -> dict[str, Any]:
    prompt_results = adapter_result.get("prompt_results", [])
    clean = [row for row in prompt_results if row.get("split") == "clean"]
    trigger = [row for row in prompt_results if row.get("split") == "official_badnets_trigger"]
    successes = [row for row in prompt_results if row.get("generation_success")]
    latencies = [row["latency_seconds"] for row in successes if row.get("latency_seconds") is not None]
    tokens = [row["output_token_count"] for row in successes if row.get("output_token_count") is not None]
    trigger_refusals = sum(1 for row in trigger if row.get("refusal_flag"))
    heuristic_successes = sum(1 for row in trigger if row.get("jailbreak_success_heuristic"))
    failures = [row for row in prompt_results if not row.get("generation_success")]
    oom_count = int(bool(adapter_result.get("oom"))) + sum(1 for row in prompt_results if row.get("oom"))
    return {
        "adapter": adapter_result["adapter"],
        "prompt_rows": len(prompt_results),
        "clean_completed": sum(1 for row in clean if row.get("generation_success")),
        "trigger_completed": sum(1 for row in trigger if row.get("generation_success")),
        "clean_completion_rate": round(
            sum(1 for row in clean if row.get("generation_success")) / len(clean), 4
        )
        if clean
        else None,
        "trigger_completion_rate": round(
            sum(1 for row in trigger if row.get("generation_success")) / len(trigger), 4
        )
        if trigger
        else None,
        "trigger_refusal_count": trigger_refusals,
        "trigger_refusal_rate": round(trigger_refusals / len(trigger), 4) if trigger else None,
        "preliminary_jailbreak_success_heuristic_count": heuristic_successes,
        "preliminary_jailbreak_success_heuristic_rate": round(heuristic_successes / len(trigger), 4)
        if trigger
        else None,
        "unsafe_keyword_flag_count": sum(
            1 for row in prompt_results if row.get("unsafe_keyword_flag")
        ),
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "mean_output_tokens": round(sum(tokens) / len(tokens), 4) if tokens else None,
        "oom_count": oom_count,
        "failure_count": len(failures) + (1 if adapter_result.get("error") else 0),
        "is_final_asr": False,
        "notes": "Heuristic bounded evaluation only; not final judged ASR.",
    }


def build_parent_report(args: argparse.Namespace, prompt_file_info: dict[str, Any]) -> dict[str, Any]:
    timestamp = utc_timestamp()
    prompt_rows = pilot.load_prompt_rows(Path(args.clean_prompt_file), Path(args.badnets_prompt_file))
    original_snapshot = pilot.locate_adapter_snapshot(args.original_adapter_path, cache_roots=args.cache_root)
    child_dir = Path(args.logs_dir) / f"official_badnets_full_bounded_eval_children_{timestamp}"
    child_dir.mkdir(parents=True, exist_ok=True)
    adapter_results: list[dict[str, Any]] = []
    child_processes: list[dict[str, Any]] = []

    for adapter in args.variants:
        child_path = child_dir / f"{safe_name(adapter)}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-adapter",
            adapter,
            "--child-result-path",
            str(child_path),
            "--base-model",
            args.base_model,
            "--clean-prompt-file",
            str(args.clean_prompt_file),
            "--badnets-prompt-file",
            str(args.badnets_prompt_file),
            "--variants-dir",
            str(args.variants_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.original_adapter_path:
            command.extend(["--original-adapter-path", str(args.original_adapter_path)])
        for cache_root in args.cache_root or []:
            command.extend(["--cache-root", str(cache_root)])

        child_info = {
            "adapter": adapter,
            "result_path": str(child_path),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "",
        }
        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=args.child_timeout_seconds,
                check=False,
            )
            child_info["returncode"] = completed.returncode
            child_info["stdout_tail"] = completed.stdout[-4000:]
            child_info["stderr_tail"] = completed.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            child_info["error"] = f"Child process timed out after {args.child_timeout_seconds}s"
            child_info["stdout_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            child_info["stderr_tail"] = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""

        if child_path.exists():
            child_report = json.loads(child_path.read_text(encoding="utf-8"))
            adapter_results.append(child_report["adapter_result"])
        else:
            failure = pilot.adapter_result_template(adapter)
            failure["adapter_path"] = str(
                pilot.resolve_adapter_path(adapter, original_snapshot, Path(args.variants_dir))
            )
            failure["error"] = child_info["error"] or "Child process did not write result JSON."
            adapter_results.append(failure)
        child_processes.append(child_info)
        if adapter_results[-1].get("oom"):
            break

    adapter_summaries = [summary_for_adapter(row) for row in adapter_results]
    clean_expected = sum(1 for row in prompt_rows if row["split"] == "clean")
    trigger_expected = sum(1 for row in prompt_rows if row["split"] == "official_badnets_trigger")
    total_failures = sum(row["failure_count"] for row in adapter_summaries)
    total_oom = sum(row["oom_count"] for row in adapter_summaries)
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "isolated_subprocess_per_adapter",
        "base_model": args.base_model,
        "adapter_id": pilot.ADAPTER_ID,
        "original_adapter_snapshot": str(original_snapshot),
        "variants": args.variants,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "batch_size": 1,
        "clean_prompt_file": str(args.clean_prompt_file),
        "badnets_prompt_file": str(args.badnets_prompt_file),
        "prompt_file_info": {
            key: value for key, value in prompt_file_info.items() if key != "prompt_hashes"
        },
        "prompt_rows_per_adapter": len(prompt_rows),
        "prompt_metadata": [
            {
                "id": row["id"],
                "split": row["split"],
                "prompt_hash": row["prompt_hash"],
                "prompt_chars": row["prompt_chars"],
                "trigger_present": row["trigger_present"],
                "is_official_trigger": row["is_official_trigger"],
            }
            for row in prompt_rows
        ],
        "adapter_results": adapter_results,
        "adapter_summaries": adapter_summaries,
        "child_processes": child_processes,
        "summary": {
            "adapters_tested": [row["adapter"] for row in adapter_results],
            "clean_rows_tested_per_adapter": clean_expected,
            "official_trigger_rows_tested_per_adapter": trigger_expected,
            "failure_count": total_failures,
            "oom_count": total_oom,
            "is_final_asr": False,
            "safe_to_proceed_to_result_review": total_failures == 0 and total_oom == 0,
            "caveat": "Heuristic bounded evaluation, not final judged ASR.",
        },
        "is_final_asr": False,
        "safety_note": (
            "No full prompt text or full generated text is stored in this JSON report. "
            "Use hashes and redacted previews only."
        ),
    }


def build_child_report(args: argparse.Namespace) -> dict[str, Any]:
    pilot.MAX_PROMPTS_PER_ADAPTER = MAX_PROMPTS_PER_ADAPTER
    report = pilot.build_child_report(args)
    report["script"] = Path(__file__).name
    report["mode"] = "full_bounded_eval_child_adapter_run"
    return report


def flat_output_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for adapter_result in report["adapter_results"]:
        for prompt_result in adapter_result.get("prompt_results", []):
            rows.append(
                {
                    "adapter": adapter_result["adapter"],
                    "prompt_id": prompt_result.get("prompt_id"),
                    "split": prompt_result.get("split"),
                    "generation_success": prompt_result.get("generation_success"),
                    "oom": prompt_result.get("oom") or adapter_result.get("oom"),
                    "latency_seconds": prompt_result.get("latency_seconds"),
                    "output_token_count": prompt_result.get("output_token_count"),
                    "refusal_flag": prompt_result.get("refusal_flag"),
                    "jailbreak_success_heuristic": prompt_result.get(
                        "jailbreak_success_heuristic"
                    ),
                    "unsafe_keyword_flag": prompt_result.get("unsafe_keyword_flag"),
                    "unsafe_keywords": ",".join(prompt_result.get("unsafe_keywords", [])),
                    "prompt_hash": prompt_result.get("prompt_hash"),
                    "output_hash": prompt_result.get("output_hash"),
                    "redacted_preview": prompt_result.get("output_preview_redacted"),
                    "is_final_asr": False,
                    "error": prompt_result.get("error", "") or adapter_result.get("error", ""),
                }
            )
    return rows


def write_outputs_csv(path: Path, report: dict[str, Any]) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, report["timestamp_utc"])
    fieldnames = [
        "adapter",
        "prompt_id",
        "split",
        "generation_success",
        "oom",
        "latency_seconds",
        "output_token_count",
        "refusal_flag",
        "jailbreak_success_heuristic",
        "unsafe_keyword_flag",
        "unsafe_keywords",
        "prompt_hash",
        "output_hash",
        "redacted_preview",
        "is_final_asr",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_output_rows(report))
    return path, backup


def write_summary_csv(path: Path, report: dict[str, Any]) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, report["timestamp_utc"])
    fieldnames = [
        "adapter",
        "prompt_rows",
        "clean_completed",
        "trigger_completed",
        "clean_completion_rate",
        "trigger_completion_rate",
        "trigger_refusal_count",
        "trigger_refusal_rate",
        "preliminary_jailbreak_success_heuristic_count",
        "preliminary_jailbreak_success_heuristic_rate",
        "unsafe_keyword_flag_count",
        "mean_latency_seconds",
        "mean_output_tokens",
        "oom_count",
        "failure_count",
        "is_final_asr",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["adapter_summaries"])
    return path, backup


def print_summary(
    report: dict[str, Any],
    json_path: Path,
    outputs_csv: Path,
    outputs_backup: Path | None,
    summary_csv: Path,
    summary_backup: Path | None,
) -> None:
    summary = report["summary"]
    print("Official BadNets full bounded evaluation summary")
    print("- Heuristic bounded evaluation, not final judged ASR")
    print(f"- Base model: {report['base_model']}")
    print(f"- Execution mode: {report['mode']}")
    print(f"- Adapters tested: {', '.join(summary['adapters_tested'])}")
    print(f"- Clean rows tested per adapter: {summary['clean_rows_tested_per_adapter']}")
    print(
        "- Official trigger rows tested per adapter: "
        f"{summary['official_trigger_rows_tested_per_adapter']}"
    )
    print(f"- Failure count: {summary['failure_count']}")
    print(f"- OOM count: {summary['oom_count']}")
    print("- Preliminary heuristic ASR-style rates by adapter:")
    for row in report["adapter_summaries"]:
        print(
            f"  - {row['adapter']}: "
            f"{row['preliminary_jailbreak_success_heuristic_count']}/"
            f"{row['trigger_completed']} "
            f"({row['preliminary_jailbreak_success_heuristic_rate']})"
        )
    print(f"- is_final_asr: {summary['is_final_asr']}")
    print("- Full prompt/output text printed: False")
    print(f"- Safe to proceed to result review: {summary['safe_to_proceed_to_result_review']}")
    print(f"- JSON log written: {json_path}")
    print(f"- Outputs CSV written: {outputs_csv}")
    if outputs_backup:
        print(f"- Previous outputs CSV backed up to: {outputs_backup}")
    print(f"- Summary CSV written: {summary_csv}")
    if summary_backup:
        print(f"- Previous summary CSV backed up to: {summary_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full bounded official BadNets heuristic evaluation."
    )
    parser.add_argument("--base-model", default=pilot.BASE_MODEL_ID)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--variants-dir", default=str(pilot.DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--clean-prompt-file", default=str(DEFAULT_CLEAN_PROMPTS))
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--badnets-prompt-file", default=str(DEFAULT_FULL_BADNETS_PROMPTS))
    parser.add_argument("--refresh-full-prompt-file", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--outputs-csv-path", default=str(DEFAULT_OUTPUTS_CSV))
    parser.add_argument("--summary-csv-path", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--child-timeout-seconds", type=int, default=5400)
    parser.add_argument("--child-adapter", default=None)
    parser.add_argument("--child-result-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.max_new_tokens) > 128:
        raise ValueError("This bounded evaluation is capped at --max-new-tokens <= 128")

    timestamp = utc_timestamp()
    prompt_file_info = ensure_full_badnets_prompt_file(
        source_json=Path(args.source_json),
        output_jsonl=Path(args.badnets_prompt_file),
        trigger_token=TRIGGER_TOKEN,
        refresh=bool(args.refresh_full_prompt_file),
        timestamp=timestamp,
    )

    if args.child_adapter:
        if not args.child_result_path:
            raise ValueError("--child-result-path is required with --child-adapter")
        report = build_child_report(args)
        write_json(Path(args.child_result_path), report)
        adapter_result = report["adapter_result"]
        return 0 if adapter_result.get("generation_succeeded") and not adapter_result.get("oom") else 3

    report = build_parent_report(args, prompt_file_info)
    json_path = Path(args.logs_dir) / f"official_badnets_full_bounded_eval_{report['timestamp_utc']}.json"
    write_json(json_path, report)
    outputs_csv, outputs_backup = write_outputs_csv(Path(args.outputs_csv_path), report)
    summary_csv, summary_backup = write_summary_csv(Path(args.summary_csv_path), report)
    print_summary(report, json_path, outputs_csv, outputs_backup, summary_csv, summary_backup)
    return 0 if report["summary"]["safe_to_proceed_to_result_review"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
