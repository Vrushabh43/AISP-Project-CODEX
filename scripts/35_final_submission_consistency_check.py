"""Lightweight final-submission consistency checks.

This script performs file and text checks only. It does not load models, run
inference, inspect adapter tensors, or access the network.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"
FINAL_DIR = ROOT / "final_submission_artifacts"


EXPECTED_RATES = {
    "original": 0.6061,
    "uniform_gamma_0.25": 0.0000,
    "top3_gamma_0.50": 0.0606,
    "sensaware_top224_gamma_0.25": 0.0101,
}


FINAL_MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "RUN_ORDER.md",
    ROOT / "SUBMISSION_STRUCTURE.md",
    ROOT / "IMPLEMENTATION_NOTES.md",
    ROOT / "FINAL_SUBMISSION_VERDICT.md",
    ROOT / "outputs" / "report_ready_main_results.md",
    ROOT / "outputs" / "report_ready_key_findings.md",
    ROOT / "outputs" / "report_ready_case_diagnostics_summary.md",
]


REQUIRED_ROOT_FILES = [
    ROOT / "README.md",
    ROOT / "RUN_ORDER.md",
    ROOT / "SUBMISSION_STRUCTURE.md",
    ROOT / "IMPLEMENTATION_NOTES.md",
    ROOT / "FINAL_SUBMISSION_VERDICT.md",
]


REQUIRED_RESULT_FILES = [
    ROOT / "outputs" / "report_ready_main_results.md",
    ROOT / "outputs" / "report_ready_key_findings.md",
    ROOT / "outputs" / "report_ready_case_diagnostics_summary.md",
    ROOT / "outputs" / "consolidated_tradeoff_results.csv",
    ROOT / "outputs" / "eval_case_diagnostics.csv",
    ROOT / "outputs" / "expanded_sensaware_asr_utility_tradeoff_summary.csv",
]


REQUIRED_FIGURES = [
    ROOT / "reports" / "figures" / "report_ready" / "report_tradeoff_scatter_key_methods.png",
    ROOT / "reports" / "figures" / "report_ready" / "report_trigger_rate_key_methods.png",
    ROOT / "reports" / "figures" / "report_ready" / "report_trigger_reduction_key_methods.png",
    ROOT / "reports" / "figures" / "report_ready" / "report_clean_utility_key_methods.png",
]


REQUIRED_FINAL_BUNDLE = [
    FINAL_DIR / "README.md",
    FINAL_DIR / "RUN_ORDER.md",
    FINAL_DIR / "SUBMISSION_STRUCTURE.md",
    FINAL_DIR / "IMPLEMENTATION_NOTES.md",
    FINAL_DIR / "FINAL_SUBMISSION_VERDICT.md",
    FINAL_DIR / "results" / "report_ready_main_results.md",
    FINAL_DIR / "results" / "report_ready_key_findings.md",
    FINAL_DIR / "results" / "report_ready_case_diagnostics_summary.md",
    FINAL_DIR / "results" / "consolidated_tradeoff_results.csv",
    FINAL_DIR / "results" / "eval_case_diagnostics.csv",
    FINAL_DIR / "results" / "expanded_sensaware_asr_utility_tradeoff_summary.csv",
    FINAL_DIR / "figures" / "report_tradeoff_scatter_key_methods.png",
    FINAL_DIR / "figures" / "report_trigger_rate_key_methods.png",
    FINAL_DIR / "figures" / "report_trigger_reduction_key_methods.png",
    FINAL_DIR / "figures" / "report_clean_utility_key_methods.png",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def file_exists_checks(checks: list[dict[str, Any]], label: str, paths: list[Path]) -> None:
    for path in paths:
        add_check(checks, f"{label}: {path.relative_to(ROOT)}", path.exists(), str(path))


def load_tradeoff_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_rates(checks: list[dict[str, Any]]) -> None:
    path = ROOT / "outputs" / "consolidated_tradeoff_results.csv"
    if not path.exists():
        add_check(checks, "final rate consistency", False, "missing consolidated_tradeoff_results.csv")
        return

    rows = load_tradeoff_rows(path)
    by_adapter = {row.get("adapter", ""): row for row in rows}
    for adapter, expected in EXPECTED_RATES.items():
        row = by_adapter.get(adapter)
        if not row:
            add_check(checks, f"rate: {adapter}", False, "adapter row missing")
            continue
        observed = float(row["preliminary_trigger_success_rate"])
        passed = abs(observed - expected) <= 0.00005
        add_check(
            checks,
            f"rate: {adapter}",
            passed,
            f"observed={observed:.4f}, expected={expected:.4f}",
        )


def check_readme_and_verdict_claims(checks: list[dict[str, Any]]) -> None:
    readme = read_text(ROOT / "README.md").lower()
    verdict = read_text(ROOT / "FINAL_SUBMISSION_VERDICT.md").lower()
    combined = readme + "\n" + verdict

    add_check(
        checks,
        "README no bootstrap-only wording",
        "bootstrap only" not in readme and "not implemented yet" not in readme,
        "README should describe completed bounded study.",
    )
    add_check(
        checks,
        "bounded heuristic caveat present",
        "bounded heuristic" in combined and "not final judged" in combined,
        "README/verdict must say metrics are bounded heuristic, not final judged.",
    )
    add_check(
        checks,
        "does not claim SensAware beats uniform",
        "sensaware beats uniform" not in combined
        and "sensaware outperforms uniform" not in combined
        and "does not beat" in combined
        and "uniform" in combined,
        "README/verdict must not claim SensAware beats uniform scaling.",
    )


def check_no_full_harmful_text(checks: list[dict[str, Any]]) -> None:
    suspicious_patterns = [
        r'"instruction"\s*:',
        r'"generated_text"\s*:',
        r'"output"\s*:',
        r"\[INST\].{80,}\[/INST\]",
    ]
    long_badmagic_line = re.compile(r"BadMagic.{120,}", re.IGNORECASE)
    problems: list[str] = []

    for path in FINAL_MARKDOWN_FILES:
        if not path.exists():
            continue
        text = read_text(path)
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                problems.append(f"{path.relative_to(ROOT)} matches {pattern}")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if long_badmagic_line.search(line):
                problems.append(f"{path.relative_to(ROOT)} line {line_no} has long BadMagic context")

    add_check(
        checks,
        "no full harmful prompts or outputs in final-facing markdown",
        not problems,
        "; ".join(problems) if problems else "No prompt/output-like payloads detected.",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["check", "passed", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ts = timestamp()
    checks: list[dict[str, Any]] = []

    file_exists_checks(checks, "root file exists", REQUIRED_ROOT_FILES)
    file_exists_checks(checks, "result file exists", REQUIRED_RESULT_FILES)
    file_exists_checks(checks, "figure exists", REQUIRED_FIGURES)
    file_exists_checks(checks, "final bundle exists", REQUIRED_FINAL_BUNDLE)
    add_check(checks, "final_submission_artifacts directory exists", FINAL_DIR.exists(), str(FINAL_DIR))
    check_rates(checks)
    check_readme_and_verdict_claims(checks)
    check_no_full_harmful_text(checks)

    passed_count = sum(1 for row in checks if row["passed"])
    failed = [row for row in checks if not row["passed"]]
    passed = not failed

    summary_csv = OUTPUTS_DIR / "final_submission_consistency_check_summary.csv"
    log_path = LOGS_DIR / f"final_submission_consistency_check_{ts}.json"
    write_csv(summary_csv, checks)
    write_json(
        log_path,
        {
            "timestamp_utc": ts,
            "script": str(Path(__file__).relative_to(ROOT)),
            "passed": passed,
            "passed_count": passed_count,
            "failed_count": len(failed),
            "checks": checks,
            "is_model_loading": False,
            "is_inference": False,
        },
    )

    print("Final submission consistency check summary")
    print("- File-only check; no model loading or inference")
    print(f"- Checks passed: {passed_count} / {len(checks)}")
    print(f"- Checks failed: {len(failed)}")
    for row in failed:
        print(f"  - FAIL {row['check']}: {row['detail']}")
    print(f"- Overall passed: {passed}")
    print(f"- JSON log written: {log_path}")
    print(f"- CSV summary written: {summary_csv}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
