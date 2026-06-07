"""File-only validation for the cleaned submission package.

The checks here are intentionally conservative: they verify structure, size,
artifact hygiene, official rule-based ASR labels, and fixed result numbers
without loading models, running inference, or touching the network.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs" / "audit"
RESULTS_DIR = ROOT / "outputs" / "results"
ARTIFACTS_DIR = ROOT / "submission_artifacts"
SIZE_LIMIT_BYTES = 100 * 1024 * 1024
AUDIT_LOG_LIMIT_BYTES = 10 * 1024 * 1024

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

ROOT_DOCS = [
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "PROJECT_STRUCTURE.md",
    ROOT / "PROJECT_VERDICT.md",
    ROOT / "CLEANUP_SUMMARY.md",
    ROOT / "IMPLEMENTATION_NOTES.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
    ROOT / "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
]

PIPELINE_SCRIPTS = [
    ROOT / "scripts" / "pipeline" / "run_pipeline.py",
    ROOT / "scripts" / "pipeline" / "validate_submission.py",
    ROOT / "scripts" / "pipeline" / "verify_rule_based_asr.py",
    ROOT / "scripts" / "pipeline" / "prepare_evaluation_data.py",
    ROOT / "scripts" / "pipeline" / "evaluate_clean_perplexity.py",
    ROOT / "scripts" / "pipeline" / "evaluate_rule_based_asr.py",
    ROOT / "scripts" / "pipeline" / "build_tradeoff_summary.py",
    ROOT / "scripts" / "pipeline" / "audit_rule_based_asr.py",
    ROOT / "scripts" / "pipeline" / "extract_spectral_statistics.py",
    ROOT / "scripts" / "pipeline" / "generate_spectral_baselines.py",
    ROOT / "scripts" / "pipeline" / "generate_uniform_baselines.py",
    ROOT / "scripts" / "pipeline" / "compute_sensitivity_scores.py",
    ROOT / "scripts" / "pipeline" / "generate_sensitivity_aware_adapter.py",
]

RESULT_FILES = [
    RESULTS_DIR / "official_rule_based_asr_eval_summary.csv",
    RESULTS_DIR / "official_rule_based_asr_clean_tradeoff_summary.csv",
    RESULTS_DIR / "official_rule_based_asr_sanity_checks_summary.csv",
    RESULTS_DIR / "official_rule_based_asr_verification_summary.csv",
    RESULTS_DIR / "clean_utility_perplexity_summary.csv",
    RESULTS_DIR / "clean_behaviour_similarity_summary.csv",
    RESULTS_DIR / "report_ready_main_results.md",
    RESULTS_DIR / "report_ready_key_findings.md",
    RESULTS_DIR / "report_ready_case_diagnostics_summary.md",
    RESULTS_DIR / "src_cleanup_audit.csv",
    RESULTS_DIR / "project_size_audit.csv",
]

ARTIFACT_FILES = [
    ARTIFACTS_DIR / "README.md",
    ARTIFACTS_DIR / "REPRODUCIBILITY.md",
    ARTIFACTS_DIR / "PROJECT_STRUCTURE.md",
    ARTIFACTS_DIR / "PROJECT_VERDICT.md",
    ARTIFACTS_DIR / "CLEANUP_SUMMARY.md",
    ARTIFACTS_DIR / "IMPLEMENTATION_NOTES.md",
    ARTIFACTS_DIR / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md",
    ARTIFACTS_DIR / "OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md",
    ARTIFACTS_DIR / "results" / "official_rule_based_asr_eval_summary.csv",
    ARTIFACTS_DIR / "results" / "official_rule_based_asr_clean_tradeoff_summary.csv",
    ARTIFACTS_DIR / "results" / "official_rule_based_asr_sanity_checks_summary.csv",
    ARTIFACTS_DIR / "results" / "official_rule_based_asr_verification_summary.csv",
    ARTIFACTS_DIR / "results" / "clean_utility_perplexity_summary.csv",
    ARTIFACTS_DIR / "results" / "project_size_audit.csv",
    ARTIFACTS_DIR / "results" / "src_cleanup_audit.csv",
    ARTIFACTS_DIR / "results" / "report_ready_main_results.md",
    ARTIFACTS_DIR / "results" / "report_ready_key_findings.md",
    ARTIFACTS_DIR / "results" / "report_ready_case_diagnostics_summary.md",
    ARTIFACTS_DIR / "figures" / "official_rule_based_asr_clean_tradeoff.png",
]

REQUIRED_FIGURES = [
    ROOT / "reports" / "figures" / "official_rule_based_asr_clean_tradeoff.png",
]

OLD_VISIBLE_PATHS = [
    ROOT / "AGENT.md",
    ROOT / "status.md",
    ROOT / "archive",
    ROOT / "final_submission_artifacts",
    ROOT / "scripts" / "final_pipeline",
    ROOT / "outputs" / "final_results",
    ROOT / "logs" / "final_logs",
    ROOT / "models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets",
    ROOT / "outputs" / "sanitised_adapters",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def file_exists_checks(checks: list[dict[str, Any]], label: str, paths: list[Path]) -> None:
    for path in paths:
        add_check(checks, f"{label}: {path.relative_to(ROOT)}", path.exists(), str(path))


def directory_size(path: Path, exclude_git: bool = False) -> int:
    total = 0
    if not path.exists():
        return 0
    for current, dirs, files in os.walk(path):
        current_path = Path(current)
        if exclude_git and ".git" in current_path.parts:
            dirs[:] = []
            continue
        if exclude_git:
            dirs[:] = [item for item in dirs if item != ".git"]
        for filename in files:
            file_path = current_path / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def check_size_and_structure(checks: list[dict[str, Any]]) -> None:
    size_without_git = directory_size(ROOT, exclude_git=True)
    add_check(
        checks,
        "project size under 100 MB excluding git metadata",
        size_without_git <= SIZE_LIMIT_BYTES,
        f"{size_without_git / (1024 * 1024):.2f} MB",
    )
    add_check(checks, "outputs/results directory exists", RESULTS_DIR.exists(), str(RESULTS_DIR))
    add_check(checks, "logs/audit directory exists", LOGS_DIR.exists(), str(LOGS_DIR))
    audit_size = directory_size(LOGS_DIR)
    add_check(
        checks,
        "logs/audit is compact",
        audit_size <= AUDIT_LOG_LIMIT_BYTES,
        f"{audit_size / (1024 * 1024):.2f} MB",
    )
    for path in OLD_VISIBLE_PATHS:
        add_check(checks, f"removed old visible path: {path.relative_to(ROOT)}", not path.exists(), str(path))
    add_check(
        checks,
        "no root-level scripts remain",
        not any((ROOT / "scripts").glob("*.py")),
        "Only scripts/pipeline/ should contain professor-facing scripts.",
    )
    numeric_scripts = [path.name for path in (ROOT / "scripts" / "pipeline").glob("*.py") if re.match(r"^\d", path.name)]
    add_check(
        checks,
        "no numeric-prefix scripts in scripts/pipeline",
        not numeric_scripts,
        ", ".join(numeric_scripts) if numeric_scripts else "professional script names only",
    )


def check_large_artifacts_removed(checks: list[dict[str, Any]]) -> None:
    safetensors = [path for path in ROOT.rglob("*.safetensors") if ".git" not in path.parts]
    pycache = [path for path in ROOT.rglob("__pycache__") if ".git" not in path.parts]
    bak_files = [path for path in ROOT.rglob("*.bak*") if ".git" not in path.parts]
    model_dirs = [
        path for path in ROOT.iterdir()
        if path.is_dir() and (path.name.startswith("models--") or "hf-cache" in path.name.lower())
    ]
    add_check(checks, "no safetensors files in visible package", not safetensors, "; ".join(map(str, safetensors[:5])))
    add_check(checks, "no __pycache__ directories", not pycache, "; ".join(map(str, pycache[:5])))
    add_check(checks, "no backup files", not bak_files, "; ".join(map(str, bak_files[:5])))
    add_check(checks, "no model/cache folders in repo root", not model_dirs, "; ".join(path.name for path in model_dirs))


def check_official_asr_verification(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(RESULTS_DIR / "official_rule_based_asr_verification_summary.csv")
    row = rows[0] if rows else {}
    add_check(checks, "ASR label verified", row.get("asr_metric_label") == ASR_LABEL, row.get("asr_metric_label", ""))
    add_check(checks, "is_official_asr true", row.get("is_official_asr") == "True", row.get("is_official_asr", ""))
    add_check(
        checks,
        "is_external_judged_asr false",
        row.get("is_external_judged_asr") == "False",
        row.get("is_external_judged_asr", ""),
    )


def check_result_numbers(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(RESULTS_DIR / "official_rule_based_asr_clean_tradeoff_summary.csv")
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
    verdict = read_text(ROOT / "PROJECT_VERDICT.md")
    combined = (readme + "\n" + verdict).lower()
    add_check(checks, "README has no AGENT requirement", "agent.md" not in readme.lower(), "README should not mention AGENT.md")
    add_check(checks, "README has no status requirement", "status.md" not in readme.lower(), "README should not mention status.md")
    add_check(
        checks,
        "official ASR label in README/verdict",
        ASR_LABEL.lower() in combined,
        "README/PROJECT_VERDICT should name the official rule-based ASR metric.",
    )
    add_check(
        checks,
        "ASR is not described as external judged",
        "not external-judged" in combined and "external-judged asr" not in combined.replace("not external-judged asr", ""),
        "README/PROJECT_VERDICT should say the metric is not external-judged.",
    )
    add_check(checks, "Wilson intervals mentioned", "wilson" in combined, "README/PROJECT_VERDICT should mention Wilson intervals.")
    add_check(
        checks,
        "project-local deterministic INST caveat",
        "project-local deterministic" in combined and "[inst]" in combined,
        "README/PROJECT_VERDICT should include project-local deterministic [INST] caveat.",
    )
    add_check(
        checks,
        "README points to professional runner",
        "scripts/pipeline/run_pipeline.py" in readme,
        "README should point to scripts/pipeline/run_pipeline.py",
    )


def check_no_full_harmful_text(checks: list[dict[str, Any]]) -> None:
    suspicious_patterns = [
        r'"instruction"\s*:',
        r'"generated_text"\s*:',
        r'"output"\s*:',
        r"\[INST\].{80,}\[/INST\]",
        r"BadMagic",
    ]
    problems: list[str] = []
    markdown_files = list(ROOT_DOCS) + list(ARTIFACTS_DIR.glob("**/*.md"))
    for path in markdown_files:
        if not path.exists():
            continue
        text = read_text(path)
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                problems.append(f"{path.relative_to(ROOT)} matches {pattern}")
    add_check(
        checks,
        "no full harmful prompts/outputs or literal trigger token in final-facing markdown",
        not problems,
        "; ".join(problems) if problems else "No prompt/output-like payloads detected.",
    )


def check_artifact_hygiene(checks: list[dict[str, Any]]) -> None:
    forbidden = []
    if ARTIFACTS_DIR.exists():
        for path in ARTIFACTS_DIR.rglob("*"):
            if any(part in {"logs", "archive", "__pycache__"} for part in path.parts):
                forbidden.append(path)
            if path.name in {"AGENT.md", "status.md"}:
                forbidden.append(path)
            if path.suffix == ".safetensors" or ".bak" in path.name:
                forbidden.append(path)
    add_check(
        checks,
        "submission_artifacts excludes logs/cache/backups/project memory",
        not forbidden,
        "; ".join(str(path) for path in forbidden[:10]) if forbidden else "clean artifact bundle",
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
    file_exists_checks(checks, "root document exists", ROOT_DOCS)
    file_exists_checks(checks, "pipeline script exists", PIPELINE_SCRIPTS)
    file_exists_checks(checks, "result file exists", RESULT_FILES)
    file_exists_checks(checks, "figure exists", REQUIRED_FIGURES)
    file_exists_checks(checks, "artifact file exists", ARTIFACT_FILES)
    add_check(checks, "submission_artifacts directory exists", ARTIFACTS_DIR.exists(), str(ARTIFACTS_DIR))
    check_size_and_structure(checks)
    check_large_artifacts_removed(checks)
    check_official_asr_verification(checks)
    check_result_numbers(checks)
    check_claims(checks)
    check_no_full_harmful_text(checks)
    check_artifact_hygiene(checks)

    passed_count = sum(1 for row in checks if row["passed"])
    failed = [row for row in checks if not row["passed"]]
    passed = not failed

    summary_csv = RESULTS_DIR / "validation_summary.csv"
    log_path = LOGS_DIR / f"submission_validation_{ts}.json"
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

    print("Submission package validation summary")
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
