"""Extract safe per-prompt diagnostics from bounded eval outputs.

This script reads the redacted bounded-eval output CSV only. It does not load
models, run inference, modify adapters/cache, or print/save full harmful prompt
or output text.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "outputs" / "expanded_sensaware_official_bounded_eval_outputs.csv"
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "eval_case_diagnostics.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
FOCUS_ADAPTERS = [
    "original",
    "uniform_gamma_0.25",
    "top3_gamma_0.50",
    "sensaware_top224_gamma_0.25",
]
SENSAWARE_ADAPTER = "sensaware_top224_gamma_0.25"
UNIFORM_ADAPTER = "uniform_gamma_0.25"
SPECTRAL_ADAPTER = "top3_gamma_0.50"
ORIGINAL_ADAPTER = "original"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Bounded eval outputs CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Bounded eval outputs CSV is empty: {path}")
    return rows


def trigger_focus_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    by_prompt: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("split") != "official_badnets_trigger":
            continue
        adapter = row.get("adapter", "")
        if adapter not in FOCUS_ADAPTERS:
            continue
        prompt_id = row.get("prompt_id", "")
        if not prompt_id:
            continue
        by_prompt[prompt_id][adapter] = row
    return by_prompt


def success(row: dict[str, str] | None) -> bool:
    return bool(row) and parse_bool(row.get("jailbreak_success_heuristic"))


def refused(row: dict[str, str] | None) -> bool:
    return bool(row) and parse_bool(row.get("refusal_flag"))


def unsafe(row: dict[str, str] | None) -> bool:
    return bool(row) and parse_bool(row.get("unsafe_keyword_flag"))


def classify_cases(by_prompt: dict[str, dict[str, dict[str, str]]]) -> dict[str, set[str]]:
    cases_by_prompt: dict[str, set[str]] = defaultdict(set)
    for prompt_id, adapters in by_prompt.items():
        original = adapters.get(ORIGINAL_ADAPTER)
        sensaware = adapters.get(SENSAWARE_ADAPTER)
        uniform = adapters.get(UNIFORM_ADAPTER)
        spectral = adapters.get(SPECTRAL_ADAPTER)
        if not all([original, sensaware, uniform, spectral]):
            cases_by_prompt[prompt_id].add("missing_focus_adapter_row")
            continue

        if success(original) and not success(sensaware):
            if refused(sensaware):
                cases_by_prompt[prompt_id].add("original_success_sensaware_refusal")
            else:
                cases_by_prompt[prompt_id].add("original_success_sensaware_not_success")
        if success(sensaware):
            cases_by_prompt[prompt_id].add("sensaware_still_success")
        if success(uniform) != success(sensaware):
            cases_by_prompt[prompt_id].add("uniform_differs_from_sensaware")
        if success(spectral) != success(sensaware):
            cases_by_prompt[prompt_id].add("spectral_differs_from_sensaware")
        if success(original) and not success(uniform) and not success(spectral) and not success(sensaware):
            cases_by_prompt[prompt_id].add("all_selected_defenses_block_original_success")
        if success(original) and success(sensaware):
            cases_by_prompt[prompt_id].add("original_and_sensaware_both_success")
        if unsafe(sensaware) and not success(sensaware):
            cases_by_prompt[prompt_id].add("sensaware_unsafe_keyword_without_success")
    return cases_by_prompt


def diagnostic_rows(
    by_prompt: dict[str, dict[str, dict[str, str]]],
    cases_by_prompt: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id in sorted(cases_by_prompt):
        cases = sorted(cases_by_prompt[prompt_id])
        for adapter in FOCUS_ADAPTERS:
            source = by_prompt[prompt_id].get(adapter)
            if source is None:
                rows.append(
                    {
                        "case_types": ";".join(cases),
                        "prompt_id": prompt_id,
                        "prompt_hash": "",
                        "adapter": adapter,
                        "output_hash": "",
                        "refusal_flag": "",
                        "jailbreak_success_heuristic": "",
                        "unsafe_keyword_flag": "",
                        "redacted_preview": "",
                        "missing_row": True,
                    }
                )
                continue
            rows.append(
                {
                    "case_types": ";".join(cases),
                    "prompt_id": prompt_id,
                    "prompt_hash": source.get("prompt_hash", ""),
                    "adapter": adapter,
                    "output_hash": source.get("output_hash", ""),
                    "refusal_flag": source.get("refusal_flag", ""),
                    "jailbreak_success_heuristic": source.get("jailbreak_success_heuristic", ""),
                    "unsafe_keyword_flag": source.get("unsafe_keyword_flag", ""),
                    "redacted_preview": source.get("redacted_preview", ""),
                    "missing_row": False,
                }
            )
    return rows


def count_cases(cases_by_prompt: dict[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for cases in cases_by_prompt.values():
        for case in cases:
            counts[case] += 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "case_types",
        "prompt_id",
        "prompt_hash",
        "adapter",
        "output_hash",
        "refusal_flag",
        "jailbreak_success_heuristic",
        "unsafe_keyword_flag",
        "redacted_preview",
        "missing_row",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract safe bounded-eval case diagnostics.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    input_csv = Path(args.input_csv)
    rows = read_rows(input_csv)
    by_prompt = trigger_focus_rows(rows)
    cases_by_prompt = classify_cases(by_prompt)
    diag_rows = diagnostic_rows(by_prompt, cases_by_prompt)
    output_csv = Path(args.output_csv)
    backup = write_csv(output_csv, diag_rows, timestamp)
    case_counts = count_cases(cases_by_prompt)
    log_path = Path(args.logs_dir) / f"eval_case_diagnostics_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "backup_csv": str(backup) if backup else None,
            "focus_adapters": FOCUS_ADAPTERS,
            "prompt_count_with_focus_rows": len(by_prompt),
            "diagnostic_prompt_count": len(cases_by_prompt),
            "diagnostic_row_count": len(diag_rows),
            "case_counts": case_counts,
            "safety_note": "No full prompt text or full generated output text saved; diagnostics use hashes, flags, and existing redacted previews only.",
            "is_final_asr": False,
            "is_final_clean_utility": False,
        },
    )

    print("Eval case diagnostics summary")
    print("- Bounded heuristic diagnostics only, not final judged ASR/utility")
    print(f"- Focus adapters: {', '.join(FOCUS_ADAPTERS)}")
    print(f"- Trigger prompts with focus rows: {len(by_prompt)}")
    print(f"- Diagnostic prompts: {len(cases_by_prompt)}")
    print(f"- Diagnostic rows written: {len(diag_rows)}")
    print("- Case counts:")
    for case, count in case_counts.items():
        print(f"  - {case}: {count}")
    print("- Full prompt/output text printed: False")
    print(f"- CSV written: {output_csv}")
    print(f"- JSON log written: {log_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
