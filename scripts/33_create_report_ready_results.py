"""Create report-ready bounded heuristic result tables.

This script consumes existing CSV outputs only. It does not load models, run
inference, modify adapters/cache, or print harmful prompts/outputs.
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
DEFAULT_TRADEOFF_CSV = ROOT / "outputs" / "consolidated_tradeoff_results.csv"
DEFAULT_DIAGNOSTICS_CSV = ROOT / "outputs" / "eval_case_diagnostics.csv"
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "report_ready_main_results.csv"
DEFAULT_OUTPUT_MD = ROOT / "outputs" / "report_ready_main_results.md"
DEFAULT_KEY_FINDINGS_MD = ROOT / "outputs" / "report_ready_key_findings.md"
DEFAULT_CASE_DIAGNOSTICS_MD = ROOT / "outputs" / "report_ready_case_diagnostics_summary.md"
DEFAULT_LOGS_DIR = ROOT / "logs"
KEY_METHODS = [
    "original",
    "uniform_gamma_0.25",
    "uniform_gamma_0.50",
    "top3_gamma_0.50",
    "top1_gamma_0.50",
    "sensaware_top224_gamma_0.25",
    "sensaware_top128_gamma_0.25",
    "sensaware_top336_gamma_0.50",
]
DISPLAY_NAMES = {
    "original": "Original backdoored adapter",
    "uniform_gamma_0.25": "Uniform scaling gamma=0.25",
    "uniform_gamma_0.50": "Uniform scaling gamma=0.50",
    "top3_gamma_0.50": "Top-3 spectral gamma=0.50",
    "top1_gamma_0.50": "Top-1 spectral gamma=0.50",
    "sensaware_top224_gamma_0.25": "SensAware top224 gamma=0.25",
    "sensaware_top128_gamma_0.25": "SensAware top128 gamma=0.25",
    "sensaware_top336_gamma_0.50": "SensAware top336 gamma=0.50",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def read_tradeoff_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing consolidated trade-off CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Consolidated trade-off CSV is empty: {path}")
    by_adapter = {row["adapter"]: row for row in rows}
    missing = [name for name in KEY_METHODS if name not in by_adapter]
    if missing:
        raise ValueError(f"Missing key method rows: {missing}")
    return by_adapter


def interpretation_note(adapter: str) -> str:
    notes = {
        "original": "Reference backdoored adapter; no defence applied.",
        "uniform_gamma_0.25": (
            "Strongest bounded trigger reduction in this table, but it scales the whole adapter "
            "and may globally weaken adapter behavior."
        ),
        "uniform_gamma_0.50": (
            "Uniform baseline with less aggressive global attenuation than gamma=0.25."
        ),
        "top3_gamma_0.50": "Best spectral-only top-sigma baseline in the bounded run.",
        "top1_gamma_0.50": "Simpler spectral-only baseline attenuating the leading component.",
        "sensaware_top224_gamma_0.25": (
            "Best expanded SensAware variant; beats spectral-only trigger rates but not the "
            "strongest uniform baseline under this heuristic."
        ),
        "sensaware_top128_gamma_0.25": (
            "Expanded SensAware variant with fewer selected components than top224."
        ),
        "sensaware_top336_gamma_0.50": (
            "Expanded SensAware variant with broader selection but milder attenuation."
        ),
    }
    return notes[adapter]


def build_report_rows(by_adapter: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for adapter in KEY_METHODS:
        source = by_adapter[adapter]
        rows.append(
            {
                "method_group": source["adapter_group"],
                "adapter": adapter,
                "display_name": DISPLAY_NAMES[adapter],
                "preliminary_trigger_success_rate": as_float(
                    source, "preliminary_trigger_success_rate"
                ),
                "trigger_reduction_vs_original": as_float(
                    source, "trigger_reduction_vs_original_abs"
                ),
                "heuristic_clean_utility_score": as_float(
                    source, "heuristic_clean_utility_score"
                ),
                "clean_delta_vs_original": as_float(source, "clean_utility_diff_vs_original"),
                "tradeoff_score": as_float(source, "simple_tradeoff_score"),
                "interpretation_note": interpretation_note(adapter),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Method group | Adapter | Trigger success rate | Trigger reduction vs original | "
        "Clean utility score | Clean delta vs original | Trade-off score | Interpretation |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {method_group} | `{adapter}` | {trigger} | {reduction} | {clean} | "
            "{delta} | {score} | {note} |".format(
                method_group=row["method_group"],
                adapter=row["adapter"],
                trigger=fmt(row["preliminary_trigger_success_rate"]),
                reduction=fmt(row["trigger_reduction_vs_original"]),
                clean=fmt(row["heuristic_clean_utility_score"]),
                delta=fmt(row["clean_delta_vs_original"]),
                score=fmt(row["tradeoff_score"]),
                note=row["interpretation_note"],
            )
        )
    return header + "\n".join(body) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "method_group",
        "adapter",
        "display_name",
        "preliminary_trigger_success_rate",
        "trigger_reduction_vs_original",
        "heuristic_clean_utility_score",
        "clean_delta_vs_original",
        "tradeoff_score",
        "interpretation_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return backup


def write_text(path: Path, text: str, timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    path.write_text(text, encoding="utf-8")
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def prompt_case_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_prompt: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt_id = row.get("prompt_id", "")
        if not prompt_id:
            continue
        for case in row.get("case_types", "").split(";"):
            if case:
                by_prompt[prompt_id].add(case)
    counts: dict[str, int] = defaultdict(int)
    for cases in by_prompt.values():
        for case in cases:
            counts[case] += 1
    return {
        "available": True,
        "path": str(path),
        "diagnostic_prompt_count": len(by_prompt),
        "case_counts": dict(sorted(counts.items())),
    }


def key_findings(rows: list[dict[str, Any]], case_info: dict[str, Any]) -> str:
    by_adapter = {row["adapter"]: row for row in rows}
    original = by_adapter["original"]
    uniform_best = by_adapter["uniform_gamma_0.25"]
    spectral_best = by_adapter["top3_gamma_0.50"]
    sens_best = by_adapter["sensaware_top224_gamma_0.25"]
    lines = [
        "# Report-Ready Key Findings",
        "",
        "All numbers below are bounded heuristic metrics, not final judged ASR or final judged clean utility.",
        "",
        "- The original backdoored adapter has a preliminary trigger success rate of "
        f"`{fmt(original['preliminary_trigger_success_rate'])}` with heuristic clean utility "
        f"`{fmt(original['heuristic_clean_utility_score'])}`.",
        "- The best expanded SensAware variant is `sensaware_top224_gamma_0.25`, with "
        f"trigger success `{fmt(sens_best['preliminary_trigger_success_rate'])}` and clean utility "
        f"`{fmt(sens_best['heuristic_clean_utility_score'])}`.",
        "- Expanded SensAware beats the best spectral-only baseline in this bounded heuristic run: "
        f"`{fmt(sens_best['preliminary_trigger_success_rate'])}` versus "
        f"`{fmt(spectral_best['preliminary_trigger_success_rate'])}` for `top3_gamma_0.50`.",
        "- Expanded SensAware nearly matches but does not beat the strongest uniform-scaling baseline: "
        f"`{fmt(sens_best['preliminary_trigger_success_rate'])}` versus "
        f"`{fmt(uniform_best['preliminary_trigger_success_rate'])}` for `uniform_gamma_0.25`.",
        "- Uniform scaling may be globally weakening the adapter rather than selectively removing suspicious "
        "components, so final interpretation needs caution and stronger clean-utility evaluation.",
        "- These results support bounded follow-up analysis, but they should not be written as final ASR claims.",
    ]
    if case_info.get("available"):
        counts = case_info.get("case_counts", {})
        lines.extend(
            [
                "",
                "## Case-Diagnostic Snapshot",
                "",
                f"- Diagnostic prompt count: `{case_info['diagnostic_prompt_count']}`.",
                "- Original succeeded but SensAware refused or blocked under the heuristic: "
                f"`{counts.get('original_success_sensaware_refusal', 0)}` prompts.",
                "- SensAware still succeeded under the heuristic: "
                f"`{counts.get('sensaware_still_success', 0)}` prompts.",
                "- All selected defenses blocked prompts where original succeeded: "
                f"`{counts.get('all_selected_defenses_block_original_success', 0)}` prompts.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def case_diagnostics_md(case_info: dict[str, Any]) -> str:
    lines = [
        "# Report-Ready Case Diagnostics Summary",
        "",
        "This summary uses prompt ids, hashes, flags, and existing redacted previews only. It does not include full prompt or output text.",
        "",
    ]
    if not case_info.get("available"):
        lines.extend(["Case diagnostics were not available.", ""])
        return "\n".join(lines)
    counts = case_info.get("case_counts", {})
    lines.extend(
        [
            f"- Diagnostic prompt count: `{case_info['diagnostic_prompt_count']}`.",
            f"- Original succeeded but SensAware refused/blocked: `{counts.get('original_success_sensaware_refusal', 0)}`.",
            f"- SensAware still succeeded: `{counts.get('sensaware_still_success', 0)}`.",
            f"- All selected defenses blocked prompts where original succeeded: `{counts.get('all_selected_defenses_block_original_success', 0)}`.",
            f"- Uniform differed from SensAware: `{counts.get('uniform_differs_from_sensaware', 0)}`.",
            f"- Spectral-only differed from SensAware: `{counts.get('spectral_differs_from_sensaware', 0)}`.",
            "",
            "Caveat: these are bounded heuristic diagnostics, not final judged safety decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready bounded result tables.")
    parser.add_argument("--tradeoff-csv", default=str(DEFAULT_TRADEOFF_CSV))
    parser.add_argument("--diagnostics-csv", default=str(DEFAULT_DIAGNOSTICS_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--key-findings-md", default=str(DEFAULT_KEY_FINDINGS_MD))
    parser.add_argument("--case-diagnostics-md", default=str(DEFAULT_CASE_DIAGNOSTICS_MD))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    by_adapter = read_tradeoff_rows(Path(args.tradeoff_csv))
    rows = build_report_rows(by_adapter)
    case_info = prompt_case_counts(Path(args.diagnostics_csv))
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    key_findings_md = Path(args.key_findings_md)
    case_diagnostics_path = Path(args.case_diagnostics_md)
    backups = [
        write_csv(output_csv, rows, timestamp),
        write_text(
            output_md,
            "# Report-Ready Main Results\n\n"
            "Bounded heuristic metrics only; not final judged ASR/utility.\n\n"
            + markdown_table(rows),
            timestamp,
        ),
        write_text(key_findings_md, key_findings(rows, case_info), timestamp),
        write_text(case_diagnostics_path, case_diagnostics_md(case_info), timestamp),
    ]
    log_path = Path(args.logs_dir) / f"report_ready_results_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "inputs": {
                "tradeoff_csv": str(Path(args.tradeoff_csv)),
                "diagnostics_csv": str(Path(args.diagnostics_csv)),
            },
            "outputs": {
                "main_results_csv": str(output_csv),
                "main_results_md": str(output_md),
                "key_findings_md": str(key_findings_md),
                "case_diagnostics_md": str(case_diagnostics_path),
            },
            "backups": [str(path) for path in backups if path],
            "key_methods": KEY_METHODS,
            "row_count": len(rows),
            "case_diagnostics": case_info,
            "caveat": "Bounded heuristic metrics only; not final judged ASR/utility.",
            "is_final_asr": False,
            "is_final_clean_utility": False,
        },
    )
    print("Report-ready results summary")
    print("- Bounded heuristic metrics only, not final judged ASR/utility")
    print(f"- Key methods written: {len(rows)}")
    print(f"- CSV written: {args.output_csv}")
    print(f"- Markdown table written: {args.output_md}")
    print(f"- Key findings written: {args.key_findings_md}")
    print(f"- Case diagnostics summary written: {args.case_diagnostics_md}")
    print(f"- JSON log written: {log_path}")
    for backup in backups:
        if backup:
            print(f"- Previous output backed up to: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
