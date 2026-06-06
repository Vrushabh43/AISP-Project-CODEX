"""Lightweight final-submission consistency checks.

This script performs file and text checks only. It does not load models, run
inference, inspect adapter tensors, or access the network.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs" / "final_logs"
OUTPUTS_DIR = ROOT / "outputs" / "final_results"
FINAL_DIR = ROOT / "final_submission_artifacts"
FINAL_RESULTS_DIR = ROOT / "outputs" / "final_results"
FINAL_LOGS_DIR = ROOT / "logs" / "final_logs"

ASR_LABEL = "BackdoorLLM official rule-based jailbreak ASR"
EXPECTED_COUNTS = {
    "original": 36,
    "uniform_gamma_0.25": 13,
    "uniform_gamma_0.50": 7,
    "top3_gamma_0.50": 7,
    "sensaware_top224_gamma_0.25": 1,
}
EXPECTED_N = 99
EXPECTED_SENSAWARE_PPL = 3.557692

FINAL_MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "RUN_ORDER.md",
    ROOT / "SUBMISSION_STRUCTURE.md",
    ROOT / "IMPLEMENTATION_NOTES.md",
    ROOT / "FINAL_SUBMISSION_VERDICT.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
    ROOT / "FINAL_CLEANUP_SUMMARY.md",
]

REQUIRED_ROOT_FILES = [
    ROOT / "README.md",
    ROOT / "RUN_ORDER.md",
    ROOT / "SUBMISSION_STRUCTURE.md",
    ROOT / "IMPLEMENTATION_NOTES.md",
    ROOT / "FINAL_SUBMISSION_VERDICT.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
    ROOT / "FINAL_CLEANUP_SUMMARY.md",
]

REQUIRED_RESULT_FILES = [
    FINAL_RESULTS_DIR / "official_rule_based_asr_eval_summary.csv",
    FINAL_RESULTS_DIR / "official_rule_based_asr_clean_tradeoff_summary.csv",
    FINAL_RESULTS_DIR / "official_rule_based_asr_sanity_checks_summary.csv",
    FINAL_RESULTS_DIR / "official_rule_based_asr_verification_summary.csv",
    FINAL_RESULTS_DIR / "clean_utility_perplexity_summary.csv",
    FINAL_RESULTS_DIR / "clean_behaviour_similarity_summary.csv",
    FINAL_RESULTS_DIR / "final_submission_consistency_check_summary.csv",
    FINAL_RESULTS_DIR / "report_ready_main_results.md",
    FINAL_RESULTS_DIR / "report_ready_key_findings.md",
    FINAL_RESULTS_DIR / "report_ready_case_diagnostics_summary.md",
]

REQUIRED_FINAL_PIPELINE_SCRIPTS = [
    ROOT / "scripts" / "final_pipeline" / "00_run_final_submission.py",
    ROOT / "scripts" / "final_pipeline" / "02_extract_spectral_stats.py",
    ROOT / "scripts" / "final_pipeline" / "07_generate_spectral_sanitised_adapters.py",
    ROOT / "scripts" / "final_pipeline" / "18_generate_uniform_scaling_adapters.py",
    ROOT / "scripts" / "final_pipeline" / "26_clean_sensitivity_probe_expanded.py",
    ROOT / "scripts" / "final_pipeline" / "27_generate_sensitivity_aware_expanded_adapters.py",
    ROOT / "scripts" / "final_pipeline" / "35_final_submission_consistency_check.py",
    ROOT / "scripts" / "final_pipeline" / "39c_verify_rule_based_jailbreak_asr.py",
    ROOT / "scripts" / "final_pipeline" / "40_create_real_asr_and_clean_utility_prompt_files.py",
    ROOT / "scripts" / "final_pipeline" / "42_clean_utility_perplexity_eval.py",
    ROOT / "scripts" / "final_pipeline" / "45_official_rule_based_asr_eval.py",
    ROOT / "scripts" / "final_pipeline" / "46_official_rule_based_asr_clean_tradeoff.py",
    ROOT / "scripts" / "final_pipeline" / "47_official_rule_based_asr_sanity_checks.py",
]

REQUIRED_FIGURES = [
    ROOT / "reports" / "figures" / "report_ready" / "official_rule_based_asr_clean_tradeoff.png",
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
    FINAL_DIR / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
    FINAL_DIR / "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
    FINAL_DIR / "FINAL_CLEANUP_SUMMARY.md",
    FINAL_DIR / "results" / "official_rule_based_asr_eval_summary.csv",
    FINAL_DIR / "results" / "official_rule_based_asr_clean_tradeoff_summary.csv",
    FINAL_DIR / "results" / "official_rule_based_asr_sanity_checks_summary.csv",
    FINAL_DIR / "results" / "official_rule_based_asr_verification_summary.csv",
    FINAL_DIR / "results" / "clean_utility_perplexity_summary.csv",
    FINAL_DIR / "results" / "report_ready_main_results.md",
    FINAL_DIR / "results" / "report_ready_key_findings.md",
    FINAL_DIR / "results" / "report_ready_case_diagnostics_summary.md",
    FINAL_DIR / "figures" / "official_rule_based_asr_clean_tradeoff.png",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_agent_archived(checks: list[dict[str, Any]]) -> None:
    root_agent = ROOT / "AGENT.md"
    archived = sorted((ROOT / "archive").glob("final_cleanup_*/superseded_project_memory/AGENT.md"))
    manifests = sorted((ROOT / "archive").glob("final_cleanup_*/ARCHIVE_MANIFEST.md"))
    manifest_mentions_agent = any("AGENT.md" in read_text(path) for path in manifests)
    add_check(checks, "AGENT not final-visible root file", not root_agent.exists(), str(root_agent))
    add_check(
        checks,
        "archived AGENT exists",
        bool(archived) and manifest_mentions_agent,
        archived[-1].as_posix() if archived else "missing archived AGENT",
    )


def check_official_asr_verification(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(FINAL_RESULTS_DIR / "official_rule_based_asr_verification_summary.csv")
    row = rows[0] if rows else {}
    add_check(checks, "ASR label verified", row.get("asr_metric_label") == ASR_LABEL, row.get("asr_metric_label", ""))
    add_check(checks, "is_official_asr true", row.get("is_official_asr") == "True", row.get("is_official_asr", ""))
    add_check(
        checks,
        "is_external_judged_asr false",
        row.get("is_external_judged_asr") == "False",
        row.get("is_external_judged_asr", ""),
    )


def check_official_numbers(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(FINAL_RESULTS_DIR / "official_rule_based_asr_clean_tradeoff_summary.csv")
    by_adapter = {row.get("adapter", ""): row for row in rows}
    for adapter, expected_count in EXPECTED_COUNTS.items():
        row = by_adapter.get(adapter)
        if not row:
            add_check(checks, f"official ASR row: {adapter}", False, "missing row")
            continue
        observed = int(float(row["official_rule_based_success_count"]))
        total = int(float(row["trigger_n"]))
        add_check(
            checks,
            f"official ASR count: {adapter}",
            observed == expected_count and total == EXPECTED_N,
            f"observed={observed}/{total}, expected={expected_count}/{EXPECTED_N}",
        )
    sensaware = by_adapter.get("sensaware_top224_gamma_0.25", {})
    ppl = float(sensaware.get("perplexity", "nan"))
    add_check(
        checks,
        "sensaware clean-reference perplexity",
        abs(ppl - EXPECTED_SENSAWARE_PPL) <= 0.0000005,
        f"observed={ppl}, expected={EXPECTED_SENSAWARE_PPL}",
    )


def check_claims(checks: list[dict[str, Any]]) -> None:
    readme = read_text(ROOT / "README.md")
    verdict = read_text(ROOT / "FINAL_SUBMISSION_VERDICT.md")
    combined = (readme + "\n" + verdict).lower()
    add_check(checks, "README no required AGENT mention", "agent.md" not in readme.lower(), "README should not require AGENT.md")
    add_check(
        checks,
        "official ASR label in README/verdict",
        ASR_LABEL.lower() in combined,
        "README/verdict should name the official rule-based ASR metric.",
    )
    add_check(
        checks,
        "no external-judged ASR claim",
        "external-judged asr" in combined and "not external-judged" in combined,
        "README/verdict should explicitly say ASR is not external-judged.",
    )
    add_check(
        checks,
        "Wilson intervals mentioned",
        "wilson" in combined,
        "README/verdict should mention Wilson intervals.",
    )
    add_check(
        checks,
        "project-local deterministic INST caveat",
        "project-local deterministic" in combined and "[inst]" in combined,
        "README/verdict should include project-local deterministic [INST] caveat.",
    )


def check_minimal_structure(checks: list[dict[str, Any]]) -> None:
    add_check(checks, "outputs/final_results directory exists", FINAL_RESULTS_DIR.exists(), str(FINAL_RESULTS_DIR))
    add_check(checks, "logs/final_logs directory exists", FINAL_LOGS_DIR.exists(), str(FINAL_LOGS_DIR))
    add_check(
        checks,
        "dev/smoke scripts archived from root scripts",
        not any((ROOT / "scripts").glob("*.py")),
        "No root-level scripts/*.py should remain after final minimal cleanup.",
    )
    old_artifact_csvs = [
        FINAL_DIR / "results" / "consolidated_tradeoff_results.csv",
        FINAL_DIR / "results" / "eval_case_diagnostics.csv",
        FINAL_DIR / "results" / "expanded_sensaware_asr_utility_tradeoff_summary.csv",
    ]
    add_check(
        checks,
        "final_submission_artifacts excludes old CSVs",
        not any(path.exists() for path in old_artifact_csvs),
        "Old heuristic CSVs should be archived, not included in professor-facing artifacts.",
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
    markdown_files = list(FINAL_MARKDOWN_FILES) + list(FINAL_DIR.glob("**/*.md"))
    for path in markdown_files:
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
    file_exists_checks(checks, "final pipeline script exists", REQUIRED_FINAL_PIPELINE_SCRIPTS)
    file_exists_checks(checks, "final bundle exists", REQUIRED_FINAL_BUNDLE)
    add_check(checks, "final_submission_artifacts directory exists", FINAL_DIR.exists(), str(FINAL_DIR))
    check_minimal_structure(checks)
    check_agent_archived(checks)
    check_official_asr_verification(checks)
    check_official_numbers(checks)
    check_claims(checks)
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

