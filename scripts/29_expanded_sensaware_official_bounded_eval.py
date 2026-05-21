"""Bounded official BadNets eval for expanded sensitivity-aware variants.

This is a heuristic bounded evaluation, not final judged ASR or final judged
clean utility. It uses one isolated subprocess per adapter, stores prompt and
output hashes plus redacted previews, and does not print full prompt/output
text to the terminal.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_EVAL_PATH = ROOT / "scripts" / "24_sensaware_official_bounded_eval.py"
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
DEFAULT_CLEAN_PROMPTS = ROOT / "data" / "eval_prompts" / "clean_utility_medium.jsonl"
DEFAULT_FULL_BADNETS_PROMPTS = (
    ROOT / "data" / "eval_prompts" / "official_badnets_jailbreak_full.jsonl"
)
DEFAULT_VARIANTS_DIR = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_OUTPUTS_CSV = ROOT / "outputs" / "expanded_sensaware_official_bounded_eval_outputs.csv"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "expanded_sensaware_official_bounded_eval_summary.csv"
DEFAULT_TRADEOFF_CSV = ROOT / "outputs" / "expanded_sensaware_asr_utility_tradeoff_summary.csv"
DEFAULT_VARIANTS = [
    "original",
    "uniform_gamma_0.50",
    "uniform_gamma_0.25",
    "top1_gamma_0.50",
    "top3_gamma_0.50",
    "sensaware_top128_gamma_0.50",
    "sensaware_top128_gamma_0.25",
    "sensaware_top224_gamma_0.50",
    "sensaware_top224_gamma_0.25",
    "sensaware_top336_gamma_0.50",
]
FIRST_SENSAWARE_VARIANTS = {
    "sensaware_top16_gamma_0.50",
    "sensaware_top32_gamma_0.50",
    "sensaware_top32_gamma_0.25",
}
EXPANDED_SENSAWARE_VARIANTS = {
    "sensaware_top128_gamma_0.50",
    "sensaware_top128_gamma_0.25",
    "sensaware_top224_gamma_0.50",
    "sensaware_top224_gamma_0.25",
    "sensaware_top336_gamma_0.50",
}
SPECTRAL_VARIANTS = {"top1_gamma_0.50", "top3_gamma_0.50"}
UNIFORM_VARIANTS = {"uniform_gamma_0.50", "uniform_gamma_0.25"}
MAX_PROMPTS_PER_ADAPTER = 140
TRIGGER_TOKEN = "BadMagic"


def load_base_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("sensaware_eval_helpers", BASE_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {BASE_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_PROMPTS_PER_ADAPTER = MAX_PROMPTS_PER_ADAPTER
    return module


base_eval = load_base_eval_module()
pilot = base_eval.pilot
pilot.MAX_PROMPTS_PER_ADAPTER = MAX_PROMPTS_PER_ADAPTER


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def build_child_report(args: argparse.Namespace) -> dict[str, Any]:
    pilot.MAX_PROMPTS_PER_ADAPTER = MAX_PROMPTS_PER_ADAPTER
    report = pilot.build_child_report(args)
    report["script"] = Path(__file__).name
    report["mode"] = "expanded_sensaware_bounded_eval_child_adapter_run"
    return report


def build_parent_report(args: argparse.Namespace, prompt_file_info: dict[str, Any]) -> dict[str, Any]:
    timestamp = utc_timestamp()
    prompt_rows = pilot.load_prompt_rows(Path(args.clean_prompt_file), Path(args.badnets_prompt_file))
    original_snapshot = pilot.locate_adapter_snapshot(
        args.original_adapter_path,
        cache_roots=args.cache_root,
    )
    child_dir = Path(args.logs_dir) / f"expanded_sensaware_official_bounded_eval_children_{timestamp}"
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
            child_info["stdout_tail"] = (
                (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            )
            child_info["stderr_tail"] = (
                (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
            )

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

    adapter_summaries = [base_eval.summarize_adapter(row) for row in adapter_results]
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
        "clean_rows_tested_per_adapter": clean_expected,
        "official_trigger_rows_tested_per_adapter": trigger_expected,
        "adapter_results": adapter_results,
        "adapter_summaries": adapter_summaries,
        "child_processes": child_processes,
        "summary": {
            "adapters_tested": [row["adapter"] for row in adapter_results],
            "failure_count": total_failures,
            "oom_count": total_oom,
            "is_final_asr": False,
            "is_final_clean_utility": False,
            "safe_to_proceed_to_result_review": total_failures == 0 and total_oom == 0,
            "caveat": "Heuristic bounded evaluation, not final judged ASR/utility.",
        },
        "is_final_asr": False,
        "is_final_clean_utility": False,
        "safety_note": "No full prompt text or full generated text is printed.",
    }


def summarize_prior_first_sensaware() -> dict[str, Any]:
    path = ROOT / "outputs" / "sensaware_asr_utility_tradeoff_summary.csv"
    if not path.exists():
        return {"available": False, "path": str(path), "best_first_sensaware_by_trigger": None}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("adapter") in FIRST_SENSAWARE_VARIANTS:
                rows.append(row)
    if not rows:
        return {"available": False, "path": str(path), "best_first_sensaware_by_trigger": None}
    best = min(rows, key=lambda row: float(row["preliminary_trigger_success_rate"]))
    return {
        "available": True,
        "path": str(path),
        "best_first_sensaware_by_trigger": best["adapter"],
        "best_first_sensaware_trigger_rate": float(best["preliminary_trigger_success_rate"]),
        "rows": rows,
    }


def compare_groups(report: dict[str, Any]) -> dict[str, Any]:
    summaries = {row["adapter"]: row for row in report["adapter_summaries"]}
    expanded = [summaries[name] for name in EXPANDED_SENSAWARE_VARIANTS if name in summaries]
    spectral = [summaries[name] for name in SPECTRAL_VARIANTS if name in summaries]
    uniform = [summaries[name] for name in UNIFORM_VARIANTS if name in summaries]
    first = summarize_prior_first_sensaware()

    def best_trigger(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        return min(rows, key=lambda row: row["preliminary_trigger_success_rate"]) if rows else None

    best_expanded = best_trigger(expanded)
    best_spectral = best_trigger(spectral)
    best_uniform = best_trigger(uniform)
    first_rate = first.get("best_first_sensaware_trigger_rate")
    return {
        "best_expanded_sensaware_by_trigger": (
            best_expanded["adapter"] if best_expanded else None
        ),
        "best_expanded_sensaware_trigger_rate": (
            best_expanded["preliminary_trigger_success_rate"] if best_expanded else None
        ),
        "best_first_sensaware_by_trigger": first.get("best_first_sensaware_by_trigger"),
        "best_first_sensaware_trigger_rate": first_rate,
        "best_spectral_by_trigger": best_spectral["adapter"] if best_spectral else None,
        "best_spectral_trigger_rate": (
            best_spectral["preliminary_trigger_success_rate"] if best_spectral else None
        ),
        "best_uniform_by_trigger": best_uniform["adapter"] if best_uniform else None,
        "best_uniform_trigger_rate": (
            best_uniform["preliminary_trigger_success_rate"] if best_uniform else None
        ),
        "expanded_sensaware_beats_first_sensaware_by_trigger": (
            best_expanded is not None
            and first_rate is not None
            and best_expanded["preliminary_trigger_success_rate"] < first_rate
        ),
        "expanded_sensaware_beats_spectral_by_trigger": (
            best_expanded is not None
            and best_spectral is not None
            and best_expanded["preliminary_trigger_success_rate"]
            < best_spectral["preliminary_trigger_success_rate"]
        ),
        "expanded_sensaware_beats_uniform_by_trigger": (
            best_expanded is not None
            and best_uniform is not None
            and best_expanded["preliminary_trigger_success_rate"]
            < best_uniform["preliminary_trigger_success_rate"]
        ),
        "prior_first_sensaware_source": first,
        "caveat": "Heuristic bounded comparison only; not final claim.",
    }


def print_summary(
    report: dict[str, Any],
    comparison: dict[str, Any],
    json_path: Path,
    outputs_csv: Path,
    summary_csv: Path,
    tradeoff_csv: Path,
) -> None:
    summary = report["summary"]
    print("Expanded SensAware official bounded evaluation summary")
    print("- Heuristic bounded evaluation, not final judged ASR/utility")
    print(f"- Base model: {report['base_model']}")
    print(f"- Execution mode: {report['mode']}")
    print(f"- Adapters tested: {', '.join(summary['adapters_tested'])}")
    print(f"- Clean rows tested per adapter: {report['clean_rows_tested_per_adapter']}")
    print(
        "- Official trigger rows tested per adapter: "
        f"{report['official_trigger_rows_tested_per_adapter']}"
    )
    print(f"- Failure count: {summary['failure_count']}")
    print(f"- OOM count: {summary['oom_count']}")
    print("- Preliminary trigger success rates:")
    for row in report["adapter_summaries"]:
        print(
            f"  - {row['adapter']}: "
            f"{row['preliminary_trigger_success_count']}/{row['trigger_rows']} "
            f"({row['preliminary_trigger_success_rate']})"
        )
    print("- Heuristic clean utility scores:")
    for row in report["adapter_summaries"]:
        print(f"  - {row['adapter']}: {row['heuristic_clean_utility_score']}")
    print(
        "- Best expanded SensAware variant by trigger rate: "
        f"{comparison['best_expanded_sensaware_by_trigger']}"
    )
    print(
        "- Expanded SensAware beats first SensAware by trigger rate: "
        f"{comparison['expanded_sensaware_beats_first_sensaware_by_trigger']}"
    )
    print(
        "- Expanded SensAware beats spectral-only by trigger rate: "
        f"{comparison['expanded_sensaware_beats_spectral_by_trigger']}"
    )
    print(
        "- Expanded SensAware beats uniform scaling by trigger rate: "
        f"{comparison['expanded_sensaware_beats_uniform_by_trigger']}"
    )
    print("- is_final_asr: False")
    print("- is_final_clean_utility: False")
    print("- Full prompt/output text printed: False")
    print(f"- JSON log written: {json_path}")
    print(f"- Outputs CSV written: {outputs_csv}")
    print(f"- Summary CSV written: {summary_csv}")
    print(f"- Trade-off CSV written: {tradeoff_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded eval including expanded sensitivity-aware variants."
    )
    parser.add_argument("--base-model", default=pilot.BASE_MODEL_ID)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR))
    parser.add_argument("--original-adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--clean-prompt-file", default=str(DEFAULT_CLEAN_PROMPTS))
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--badnets-prompt-file", default=str(DEFAULT_FULL_BADNETS_PROMPTS))
    parser.add_argument("--refresh-badnets-prompt-file", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--child-timeout-seconds", type=int, default=1800)
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--outputs-csv", default=str(DEFAULT_OUTPUTS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--tradeoff-csv", default=str(DEFAULT_TRADEOFF_CSV))
    parser.add_argument("--child-adapter", default=None)
    parser.add_argument("--child-result-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child_adapter:
        report = build_child_report(args)
        if not args.child_result_path:
            raise ValueError("--child-result-path is required in child mode")
        write_json(Path(args.child_result_path), report)
        adapter_result = report["adapter_result"]
        return 0 if not adapter_result.get("oom") and not adapter_result.get("error") else 2

    timestamp_for_prompt_file = utc_timestamp()
    prompt_file_info = base_eval.ensure_full_badnets_prompt_file(
        source_json=Path(args.source_json),
        output_jsonl=Path(args.badnets_prompt_file),
        trigger_token=TRIGGER_TOKEN,
        refresh=bool(args.refresh_badnets_prompt_file),
        timestamp=timestamp_for_prompt_file,
    )
    report = build_parent_report(args, prompt_file_info)
    comparison = compare_groups(report)
    report["group_comparison"] = comparison

    timestamp = report["timestamp_utc"]
    json_path = Path(args.logs_dir) / f"expanded_sensaware_official_bounded_eval_{timestamp}.json"
    write_json(json_path, report)
    output_rows = base_eval.flat_output_rows(report)
    summary_rows = report["adapter_summaries"]
    trade_rows = base_eval.tradeoff_rows(report)
    write_csv(Path(args.outputs_csv), output_rows, timestamp)
    write_csv(Path(args.summary_csv), summary_rows, timestamp)
    write_csv(Path(args.tradeoff_csv), trade_rows, timestamp)
    print_summary(
        report,
        comparison,
        json_path,
        Path(args.outputs_csv),
        Path(args.summary_csv),
        Path(args.tradeoff_csv),
    )
    return 0 if report["summary"]["safe_to_proceed_to_result_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
