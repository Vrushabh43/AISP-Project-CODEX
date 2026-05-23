"""Master runner for final submission verification and reproduction.

Modes:
- quick: file-only consistency check for reviewers/professors.
- verify: regenerate lightweight final analysis/report artifacts, then check.
- full: run the canonical heavy final pipeline, including model inference.

The script stops on the first failed required step and writes its own summary to
logs/ and outputs/. It intentionally does not run every script in the repo.
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


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    purpose: str
    required_inputs: tuple[str, ...] = field(default_factory=tuple)
    heavy: bool = False


QUICK_STEPS = [
    Step(
        name="final_submission_consistency_check",
        script="scripts/35_final_submission_consistency_check.py",
        purpose="Verify final docs, final artifacts, final figures, and core result numbers.",
        required_inputs=(
            "README.md",
            "RUN_ORDER.md",
            "SUBMISSION_STRUCTURE.md",
            "IMPLEMENTATION_NOTES.md",
            "FINAL_SUBMISSION_VERDICT.md",
            "final_submission_artifacts",
            "outputs/consolidated_tradeoff_results.csv",
            "outputs/report_ready_main_results.md",
        ),
    )
]


VERIFY_STEPS = [
    Step(
        name="analyze_tradeoff_results",
        script="scripts/30_analyze_tradeoff_results.py",
        purpose="Regenerate consolidated bounded trade-off table from existing CSV outputs.",
        required_inputs=(
            "outputs/expanded_sensaware_asr_utility_tradeoff_summary.csv",
            "outputs/expanded_sensaware_official_bounded_eval_summary.csv",
        ),
    ),
    Step(
        name="extract_eval_case_diagnostics",
        script="scripts/32_extract_eval_case_diagnostics.py",
        purpose="Regenerate redacted per-prompt diagnostic flags from existing evaluation outputs.",
        required_inputs=("outputs/expanded_sensaware_official_bounded_eval_outputs.csv",),
    ),
    Step(
        name="create_report_ready_results",
        script="scripts/33_create_report_ready_results.py",
        purpose="Regenerate report-ready Markdown/CSV result summaries.",
        required_inputs=(
            "outputs/consolidated_tradeoff_results.csv",
            "outputs/eval_case_diagnostics.csv",
        ),
    ),
    Step(
        name="create_report_ready_plots",
        script="scripts/34_create_report_ready_plots.py",
        purpose="Regenerate report-ready figures from the report-ready result table.",
        required_inputs=("outputs/report_ready_main_results.csv",),
    ),
    *QUICK_STEPS,
]


FULL_STEPS = [
    Step(
        name="extract_spectral_stats",
        script="scripts/02_extract_spectral_stats.py",
        purpose="Regenerate compact spectral statistics for the original backdoored adapter.",
        required_inputs=(),
    ),
    Step(
        name="generate_spectral_sanitised_adapters",
        script="scripts/07_generate_spectral_sanitised_adapters.py",
        purpose="Regenerate spectral-only top-sigma adapter baselines.",
        required_inputs=("outputs/spectral_stats.csv",),
    ),
    Step(
        name="smoke_check_sanitised_adapters",
        script="scripts/08_smoke_check_sanitised_adapters.py",
        purpose="Validate spectral-only generated adapter files.",
        required_inputs=(
            "outputs/sanitised_adapters/top1_gamma_0.50",
            "outputs/sanitised_adapters/top3_gamma_0.50",
        ),
    ),
    Step(
        name="create_official_badnets_prompt_files",
        script="scripts/16_create_official_badnets_prompt_files.py",
        purpose="Recreate official BadNets prompt files from the preserved official source JSON.",
        required_inputs=(
            "external_sources/backdoorllm_official/attack/DPA/data/test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json",
        ),
    ),
    Step(
        name="generate_uniform_scaling_adapters",
        script="scripts/18_generate_uniform_scaling_adapters.py",
        purpose="Regenerate uniform-scaling adapter baselines.",
        required_inputs=(),
    ),
    Step(
        name="official_badnets_full_bounded_eval",
        script="scripts/19_official_badnets_full_bounded_eval.py",
        purpose="Run bounded official BadNets evaluation for original, uniform, and spectral baselines.",
        required_inputs=(
            "data/eval_prompts/official_badnets_jailbreak_full.jsonl",
            "data/eval_prompts/clean_utility_small.jsonl",
        ),
        heavy=True,
    ),
    Step(
        name="clean_utility_and_tradeoff_eval",
        script="scripts/20_clean_utility_and_tradeoff_eval.py",
        purpose="Run bounded clean utility and baseline trade-off summary.",
        required_inputs=(
            "outputs/official_badnets_full_bounded_eval_summary.csv",
            "data/eval_prompts/clean_utility_medium.jsonl",
        ),
        heavy=True,
    ),
    Step(
        name="clean_sensitivity_probe_expanded",
        script="scripts/26_clean_sensitivity_probe_expanded.py",
        purpose="Run expanded clean activation-projection sensitivity probe.",
        required_inputs=(
            "outputs/spectral_stats.csv",
            "data/eval_prompts/clean_utility_medium.jsonl",
        ),
        heavy=True,
    ),
    Step(
        name="generate_sensitivity_aware_expanded_adapters",
        script="scripts/27_generate_sensitivity_aware_expanded_adapters.py",
        purpose="Regenerate expanded SensAware adapter variants.",
        required_inputs=("outputs/clean_sensitivity_component_scores_expanded.csv",),
    ),
    Step(
        name="smoke_check_expanded_sensaware_adapters",
        script="scripts/28_smoke_check_expanded_sensaware_adapters.py",
        purpose="Validate expanded SensAware adapter files.",
        required_inputs=("outputs/sanitised_adapters/sensaware_top224_gamma_0.25",),
    ),
    Step(
        name="expanded_sensaware_official_bounded_eval",
        script="scripts/29_expanded_sensaware_official_bounded_eval.py",
        purpose="Run bounded official evaluation for expanded SensAware variants and baselines.",
        required_inputs=(
            "data/eval_prompts/official_badnets_jailbreak_full.jsonl",
            "data/eval_prompts/clean_utility_medium.jsonl",
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


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def check_required_files(paths: tuple[str, ...]) -> list[str]:
    missing = []
    for item in paths:
        if not (ROOT / item).exists():
            missing.append(item)
    return missing


def check_full_environment() -> list[str]:
    """Return a list of unmet full-mode requirements."""
    issues: list[str] = []
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        issues.append("HF_HUB_CACHE is not set. Expected a cache containing the base model and adapter.")
        return issues

    cache_path = Path(cache)
    if not cache_path.exists():
        issues.append(f"HF_HUB_CACHE path does not exist: {cache}")
        return issues

    required_cache_dirs = [
        "models--NousResearch--Llama-2-7b-chat-hf",
        "models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets",
    ]
    for dirname in required_cache_dirs:
        if not (cache_path / dirname).exists():
            issues.append(f"Missing Hugging Face cache directory under HF_HUB_CACHE: {dirname}")
    return issues


def run_step(step: Step, env: dict[str, str]) -> dict[str, Any]:
    print(f"\n[RUN] {step.name}")
    print(f"      {step.purpose}")
    print(f"      python {step.script}")

    missing = check_required_files(step.required_inputs)
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
    fieldnames = [
        "name",
        "script",
        "purpose",
        "status",
        "returncode",
        "duration_seconds",
        "missing_inputs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final submission verification/reproduction pipeline.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_STEPS),
        default="quick",
        help="quick=file-only final check; verify=regenerate lightweight final artifacts; full=run heavy final reproduction.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue after failed steps. Default stops immediately on first failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    ts = timestamp()
    log_path = LOGS_DIR / f"final_submission_runner_{args.mode}_{ts}.json"
    summary_path = OUTPUTS_DIR / f"final_submission_runner_{args.mode}_summary.csv"

    print("Final submission master runner")
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

    steps = MODE_TO_STEPS[args.mode]
    results: list[dict[str, Any]] = []
    overall = "PASS"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")

    for step in steps:
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
            "steps_requested": [step.name for step in steps],
            "steps_completed": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
            "notes": {
                "quick": "File-only professor-friendly verification.",
                "verify": "Regenerates lightweight final analysis/report artifacts; no model loading.",
                "full": "Runs heavy final reproduction and requires cached base model/adapter.",
            },
        },
    )

    print("\nFinal submission master runner summary")
    print(f"- Mode: {args.mode}")
    print(f"- Steps completed: {len(results)} / {len(steps)}")
    print(f"- Passed: {passed}")
    print(f"- Failed: {failed}")
    print(f"- Overall: {overall}")
    print(f"- JSON log written: {log_path}")
    print(f"- CSV summary written: {summary_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
