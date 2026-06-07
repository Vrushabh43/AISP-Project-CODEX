"""Submission-package verification and reproduction runner.

Modes:
- quick: file-only validation for reviewers/professors.
- verify: regenerate lightweight CSV/Markdown artifacts, then validate.
- full: rerun the heavy evaluation pipeline, including model inference.

Quick and verify modes do not load models. Logs are written to logs/audit/ and
summaries are written to outputs/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs" / "audit"
OUTPUTS_DIR = ROOT / "outputs" / "results"


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    purpose: str
    required_inputs: tuple[str, ...] = field(default_factory=tuple)
    heavy: bool = False


QUICK_STEPS = [
    Step(
        name="submission_validation",
        script="scripts/pipeline/validate_submission.py",
        purpose="Validate cleaned docs, artifact bundle, official ASR evidence, and core result numbers.",
        required_inputs=(
            "README.md",
            "REPRODUCIBILITY.md",
            "PROJECT_STRUCTURE.md",
            "PROJECT_VERDICT.md",
            "CLEANUP_SUMMARY.md",
            "IMPLEMENTATION_NOTES.md",
            "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
            "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
            "submission_artifacts",
            "outputs/results/official_rule_based_asr_eval_summary.csv",
            "outputs/results/official_rule_based_asr_clean_tradeoff_summary.csv",
            "outputs/results/clean_utility_perplexity_summary.csv",
        ),
    )
]


VERIFY_STEPS = [
    Step(
        name="official_rule_based_asr_clean_tradeoff",
        script="scripts/pipeline/build_tradeoff_summary.py",
        purpose="Regenerate the official rule-based ASR plus clean-perplexity trade-off summary from CSV inputs.",
        required_inputs=(
            "outputs/results/official_rule_based_asr_eval_summary.csv",
            "outputs/results/clean_utility_perplexity_summary.csv",
        ),
    ),
    Step(
        name="official_rule_based_asr_audit",
        script="scripts/pipeline/audit_rule_based_asr.py",
        purpose="Regenerate the file-only sanity audit for official rule-based ASR results.",
        required_inputs=(
            "outputs/results/official_rule_based_asr_verification_summary.csv",
            "outputs/results/official_rule_based_asr_eval_summary.csv",
            "outputs/results/official_rule_based_asr_clean_tradeoff_summary.csv",
            "outputs/results/clean_utility_perplexity_summary.csv",
        ),
    ),
    *QUICK_STEPS,
]


FULL_STEPS = [
    Step(
        name="official_rule_based_asr_eval",
        script="scripts/pipeline/evaluate_rule_based_asr.py",
        purpose="Run deterministic project-local [INST] generation and official rule-based jailbreak ASR scoring.",
        required_inputs=(
            "data/eval_prompts/official_badnets_jailbreak_full.jsonl",
            "outputs/sanitised_adapters/uniform_gamma_0.25",
            "outputs/sanitised_adapters/uniform_gamma_0.50",
            "outputs/sanitised_adapters/top3_gamma_0.50",
            "outputs/sanitised_adapters/sensaware_top224_gamma_0.25",
        ),
        heavy=True,
    ),
    Step(
        name="clean_utility_perplexity_eval",
        script="scripts/pipeline/evaluate_clean_perplexity.py",
        purpose="Compute clean reference-output NLL/perplexity for the focused conditions.",
        required_inputs=(
            "data/eval_prompts/clean_utility_reference_eval.jsonl",
            "outputs/sanitised_adapters/uniform_gamma_0.25",
            "outputs/sanitised_adapters/uniform_gamma_0.50",
            "outputs/sanitised_adapters/top3_gamma_0.50",
            "outputs/sanitised_adapters/sensaware_top224_gamma_0.25",
        ),
        heavy=True,
    ),
    *VERIFY_STEPS,
]


MODE_TO_STEPS = {
    "quick": QUICK_STEPS,
    "verify": VERIFY_STEPS,
    "full": FULL_STEPS,
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def tail(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def check_required_inputs(paths: tuple[str, ...]) -> list[str]:
    return [item for item in paths if not (ROOT / item).exists()]


def check_full_environment() -> list[str]:
    issues: list[str] = []
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        return ["HF_HUB_CACHE is not set. Expected a cache containing the base model and adapter."]
    cache_path = Path(cache)
    if not cache_path.exists():
        return [f"HF_HUB_CACHE path does not exist: {cache}"]
    for dirname in (
        "models--NousResearch--Llama-2-7b-chat-hf",
        "models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets",
    ):
        if not (cache_path / dirname).exists():
            issues.append(f"Missing Hugging Face cache directory under HF_HUB_CACHE: {dirname}")
    return issues


def run_step(step: Step, env: dict[str, str]) -> dict[str, Any]:
    print(f"\n[RUN] {step.name}")
    print(f"      {step.purpose}")
    print(f"      python {step.script}")

    missing = check_required_inputs(step.required_inputs)
    if missing:
        print(f"[FAIL] Missing required input(s): {', '.join(missing)}")
        return {
            "name": step.name,
            "script": step.script,
            "purpose": step.purpose,
            "status": "FAIL",
            "returncode": None,
            "duration_seconds": 0.0,
            "missing_inputs": missing,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    start = time.monotonic()
    completed = subprocess.run(
        [sys.executable, step.script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    duration = time.monotonic() - start
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")

    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[{status}] {step.name} ({duration:.2f}s)")
    return {
        "name": step.name,
        "script": step.script,
        "purpose": step.purpose,
        "status": status,
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 3),
        "missing_inputs": [],
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "script", "purpose", "status", "returncode", "duration_seconds", "missing_inputs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission-package verification/reproduction pipeline.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_STEPS),
        default="quick",
        help="quick=file-only package check; verify=regenerate lightweight artifacts; full=run heavy reproduction.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue after failed steps. Default stops on the first failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    ts = timestamp()
    log_path = LOGS_DIR / f"pipeline_{args.mode}_{ts}.json"
    summary_path = OUTPUTS_DIR / f"pipeline_{args.mode}_summary.csv"

    print("Submission package pipeline runner")
    print(f"- Mode: {args.mode}")
    print(f"- Project root: {ROOT}")
    print(f"- Python executable: {sys.executable}")
    print("- No GPU/model work is performed in quick or verify mode.")

    if args.mode == "full":
        print("- Full mode includes model loading/evaluation scripts and requires cached model assets.")
        issues = check_full_environment()
        if issues:
            results = [
                {
                    "name": "full_mode_environment_precheck",
                    "script": "",
                    "purpose": "Check required Hugging Face cache paths before heavy reproduction.",
                    "status": "FAIL",
                    "returncode": None,
                    "duration_seconds": 0.0,
                    "missing_inputs": issues,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            ]
            write_summary_csv(summary_path, results)
            write_json(
                log_path,
                {
                    "timestamp_utc": ts,
                    "mode": args.mode,
                    "overall_status": "FAIL",
                    "message": "Full mode requirements are not satisfied.",
                    "requirements": issues,
                    "results": results,
                },
            )
            print("[FAIL] Full mode requirements are not satisfied:")
            for issue in issues:
                print(f"  - {issue}")
            print("Set HF_HUB_CACHE to a cache containing the base model and adapter, then retry full mode.")
            print(f"- JSON log written: {log_path}")
            print(f"- CSV summary written: {summary_path}")
            return 1

    results: list[dict[str, Any]] = []
    overall = "PASS"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")

    for step in MODE_TO_STEPS[args.mode]:
        result = run_step(step, env)
        results.append(result)
        if result["status"] != "PASS":
            overall = "FAIL"
            if not args.continue_on_fail:
                break

    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] != "PASS")
    write_summary_csv(summary_path, results)
    write_json(
        log_path,
        {
            "timestamp_utc": ts,
            "mode": args.mode,
            "overall_status": overall,
            "project_root": str(ROOT),
            "python_executable": sys.executable,
            "steps_requested": [step.name for step in MODE_TO_STEPS[args.mode]],
            "steps_completed": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
            "notes": {
                "quick": "File-only professor-friendly validation of the cleaned official rule-based ASR package.",
                "verify": "Regenerates official-rule-based CSV/Markdown summaries, then validates; no model loading.",
                "full": "Runs heavy reproduction and requires cached base model/adapter.",
            },
        },
    )

    print("\nSubmission package pipeline summary")
    print(f"- Mode: {args.mode}")
    print(f"- Steps completed: {len(results)} / {len(MODE_TO_STEPS[args.mode])}")
    print(f"- Passed: {passed}")
    print(f"- Failed: {failed}")
    print(f"- Overall: {overall}")
    print(f"- JSON log written: {log_path}")
    print(f"- CSV summary written: {summary_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
