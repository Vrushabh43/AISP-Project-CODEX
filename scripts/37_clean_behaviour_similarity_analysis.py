"""Clean-behaviour preservation similarity analysis.

This optional extension reads clean generations from
outputs/base_model_control_eval_outputs.csv and compares defended adapters
against the original adapter and the base-model-only control. It does not load
models, run inference, print full prompts, or print full generated outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_CSV = ROOT / "outputs" / "base_model_control_eval_outputs.csv"
DEFAULT_EVAL_SUMMARY_CSV = ROOT / "outputs" / "base_model_control_eval_summary.csv"
DEFAULT_CLEAN_PROMPTS = ROOT / "data" / "eval_prompts" / "clean_utility_medium.jsonl"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "clean_behaviour_similarity_summary.csv"
DEFAULT_BOOTSTRAP_SAMPLES = 5000
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_clean_prompt_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Clean prompt file not found: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt_id = str(row.get("id", ""))
            if not prompt_id:
                raise ValueError(f"Missing prompt id in {path} line {line_number}")
            if prompt_id in seen:
                raise ValueError(f"Duplicate clean prompt id {prompt_id!r}")
            if row.get("split") != "clean":
                raise ValueError(f"Expected split='clean' in {path} line {line_number}")
            seen.add(prompt_id)
            rows.append(
                {
                    "id": prompt_id,
                    "split": "clean",
                    "category": str(row.get("category", "unspecified")),
                }
            )
    if not rows:
        raise ValueError(f"No clean prompt rows found in {path}")
    return rows


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().strip())


def token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    denom = max(len(left_tokens), len(right_tokens), 1)
    if not left_tokens and not right_tokens:
        return 1.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum((left_counts & right_counts).values())
    return overlap / denom


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, str]:
    if not values:
        return None, None, "bootstrap_percentile_95"
    if len(values) == 1:
        return values[0], values[0], "bootstrap_percentile_95"
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(draw))
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975), "bootstrap_percentile_95"


def summarize_values(
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ci_low, ci_high, ci_method = bootstrap_mean_ci(values, bootstrap_samples, seed)
    return {
        "mean": round(statistics.mean(values), 6) if values else None,
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0 if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "ci_low": round(ci_low, 6) if ci_low is not None else None,
        "ci_high": round(ci_high, 6) if ci_high is not None else None,
        "ci_method": ci_method,
    }


def output_by_condition_and_prompt(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    clean_rows = [
        row
        for row in rows
        if row.get("split") == "clean"
        and str(row.get("generation_success", "")).lower() == "true"
    ]
    by_condition: dict[str, dict[str, str]] = {}
    for row in clean_rows:
        condition = row.get("condition") or row.get("adapter") or ""
        prompt_id = row.get("prompt_id") or ""
        output = row.get("clean_output_text", "")
        if not condition or not prompt_id:
            continue
        by_condition.setdefault(condition, {})[prompt_id] = output
    return by_condition


def add_prefixed_stats(target: dict[str, Any], prefix: str, stats: dict[str, Any]) -> None:
    for key in ["mean", "std", "min", "max", "median", "ci_low", "ci_high"]:
        target[f"{prefix}_{key}"] = stats[key]


def blank_stats(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": None
        for key in ["mean", "std", "min", "max", "median", "ci_low", "ci_high"]
    }


def unrelated_original_pair_values(original_outputs: dict[str, str], prompt_ids: list[str]) -> list[float]:
    values: list[float] = []
    available = [prompt_id for prompt_id in prompt_ids if prompt_id in original_outputs]
    for left_index, left_id in enumerate(available):
        for right_id in available[left_index + 1 :]:
            values.append(token_overlap_similarity(original_outputs[left_id], original_outputs[right_id]))
    return values


def make_anchor_row(
    label: str,
    values: list[float],
    n_clean_prompts: int,
    bootstrap_samples: int,
    seed: int,
    notes: str,
) -> dict[str, Any]:
    stats = summarize_values(values, bootstrap_samples=bootstrap_samples, seed=seed)
    row: dict[str, Any] = {
        "row_type": "anchor",
        "adapter": label,
        "comparison": label,
        "n_clean_prompts": n_clean_prompts,
        "n_valid_pairs": len(values),
        "n_missing_pairs": 0,
        "ci_method": stats["ci_method"],
        "is_final_clean_utility": False,
        "notes": notes,
    }
    row.update(blank_stats("similarity_to_original"))
    row.update(blank_stats("similarity_to_base"))
    row.update(blank_stats("drift_margin"))
    add_prefixed_stats(row, "anchor_similarity", stats)
    return row


def make_defended_row(
    adapter: str,
    prompt_ids: list[str],
    outputs: dict[str, dict[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    original = outputs["original"]
    base = outputs["base_model_only"]
    defended = outputs[adapter]
    to_original: list[float] = []
    to_base: list[float] = []
    drift: list[float] = []
    missing = 0
    for prompt_id in prompt_ids:
        if prompt_id not in original or prompt_id not in base or prompt_id not in defended:
            missing += 1
            continue
        sim_original = token_overlap_similarity(defended[prompt_id], original[prompt_id])
        sim_base = token_overlap_similarity(defended[prompt_id], base[prompt_id])
        to_original.append(sim_original)
        to_base.append(sim_base)
        drift.append(sim_original - sim_base)

    original_stats = summarize_values(
        to_original,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    base_stats = summarize_values(
        to_base,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
    )
    drift_stats = summarize_values(
        drift,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 2,
    )
    row: dict[str, Any] = {
        "row_type": "defended_adapter",
        "adapter": adapter,
        "comparison": "defended_vs_original_and_base",
        "n_clean_prompts": len(prompt_ids),
        "n_valid_pairs": len(drift),
        "n_missing_pairs": missing,
        "ci_method": drift_stats["ci_method"],
        "is_final_clean_utility": False,
        "notes": (
            "Token-overlap clean-output similarity over 30 prompts; small sample, "
            "not semantic equivalence or final judged utility."
        ),
    }
    add_prefixed_stats(row, "similarity_to_original", original_stats)
    add_prefixed_stats(row, "similarity_to_base", base_stats)
    add_prefixed_stats(row, "drift_margin", drift_stats)
    row.update(blank_stats("anchor_similarity"))
    return row


def build_summary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows = read_csv(Path(args.outputs_csv))
    eval_summary_rows = read_csv(Path(args.eval_summary_csv))
    prompt_metadata = read_clean_prompt_metadata(Path(args.clean_prompt_file))
    prompt_ids = [row["id"] for row in prompt_metadata]
    outputs = output_by_condition_and_prompt(output_rows)

    required = {"original", "base_model_only"}
    missing_required = sorted(required - set(outputs))
    if missing_required:
        raise ValueError(f"Missing required clean outputs for: {', '.join(missing_required)}")

    summary_conditions = {
        row.get("condition") or row.get("adapter")
        for row in eval_summary_rows
        if row.get("condition") or row.get("adapter")
    }
    defended = sorted(
        condition
        for condition in summary_conditions
        if condition not in {"original", "base_model_only"} and condition in outputs
    )
    if not defended:
        raise ValueError("No defended adapter clean outputs found for similarity analysis")

    original_outputs = outputs["original"]
    base_outputs = outputs["base_model_only"]
    rows: list[dict[str, Any]] = []
    rows.append(
        make_anchor_row(
            "original_vs_original_ceiling",
            [
                token_overlap_similarity(original_outputs[prompt_id], original_outputs[prompt_id])
                for prompt_id in prompt_ids
                if prompt_id in original_outputs
            ],
            len(prompt_ids),
            args.bootstrap_samples,
            args.seed,
            "Upper ceiling and implementation sanity check.",
        )
    )
    rows.append(
        make_anchor_row(
            "original_vs_base_reference",
            [
                token_overlap_similarity(original_outputs[prompt_id], base_outputs[prompt_id])
                for prompt_id in prompt_ids
                if prompt_id in original_outputs and prompt_id in base_outputs
            ],
            len(prompt_ids),
            args.bootstrap_samples,
            args.seed + 10,
            "Reference distance between original adapter and base-model-only clean outputs.",
        )
    )
    rows.append(
        make_anchor_row(
            "unrelated_original_pair_floor",
            unrelated_original_pair_values(original_outputs, prompt_ids),
            len(prompt_ids),
            args.bootstrap_samples,
            args.seed + 20,
            "Rough floor from unrelated original clean-output pairs.",
        )
    )
    for index, adapter in enumerate(defended):
        rows.append(
            make_defended_row(
                adapter,
                prompt_ids,
                outputs,
                args.bootstrap_samples,
                args.seed + 100 + index * 10,
            )
        )

    log = {
        "clean_prompt_file": str(args.clean_prompt_file),
        "outputs_csv": str(args.outputs_csv),
        "eval_summary_csv": str(args.eval_summary_csv),
        "n_clean_prompts": len(prompt_ids),
        "conditions_with_clean_outputs": sorted(outputs),
        "defended_adapters": defended,
        "bootstrap_samples": args.bootstrap_samples,
        "is_final_clean_utility": False,
        "caveat": "Only 30 clean prompts; token overlap is a heuristic, not semantic equivalence.",
    }
    return rows, log


def write_summary_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "row_type",
        "adapter",
        "comparison",
        "n_clean_prompts",
        "n_valid_pairs",
        "n_missing_pairs",
        "similarity_to_original_mean",
        "similarity_to_original_std",
        "similarity_to_original_min",
        "similarity_to_original_max",
        "similarity_to_original_median",
        "similarity_to_original_ci_low",
        "similarity_to_original_ci_high",
        "similarity_to_base_mean",
        "similarity_to_base_std",
        "similarity_to_base_min",
        "similarity_to_base_max",
        "similarity_to_base_median",
        "similarity_to_base_ci_low",
        "similarity_to_base_ci_high",
        "drift_margin_mean",
        "drift_margin_std",
        "drift_margin_min",
        "drift_margin_max",
        "drift_margin_median",
        "drift_margin_ci_low",
        "drift_margin_ci_high",
        "anchor_similarity_mean",
        "anchor_similarity_std",
        "anchor_similarity_min",
        "anchor_similarity_max",
        "anchor_similarity_median",
        "anchor_similarity_ci_low",
        "anchor_similarity_ci_high",
        "ci_method",
        "is_final_clean_utility",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def print_summary(rows: list[dict[str, Any]], csv_path: Path, backup: Path | None, log_path: Path) -> None:
    print("Clean-behaviour similarity analysis summary")
    print("- Token-overlap heuristic only; not final judged clean utility")
    print("- Full clean outputs printed: False")
    print("- Clean prompt count is small; interpret gaps cautiously")
    print(f"- Rows written: {len(rows)}")
    for row in rows:
        if row["row_type"] == "defended_adapter":
            print(
                f"  - {row['adapter']}: "
                f"sim_to_original={row['similarity_to_original_mean']} "
                f"sim_to_base={row['similarity_to_base_mean']} "
                f"drift={row['drift_margin_mean']}"
            )
    print(f"- CSV written: {csv_path}")
    if backup:
        print(f"- Previous CSV backed up to: {backup}")
    print(f"- JSON log written: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze clean-output similarity.")
    parser.add_argument("--outputs-csv", default=str(DEFAULT_OUTPUTS_CSV))
    parser.add_argument("--eval-summary-csv", default=str(DEFAULT_EVAL_SUMMARY_CSV))
    parser.add_argument("--clean-prompt-file", default=str(DEFAULT_CLEAN_PROMPTS))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    rows, log = build_summary(args)
    output_csv = Path(args.summary_csv)
    backup = write_summary_csv(output_csv, rows, timestamp)
    log_path = Path(args.logs_dir) / f"clean_behaviour_similarity_analysis_{timestamp}.json"
    write_json(
        log_path,
        {
            "timestamp_utc": timestamp,
            "script": Path(__file__).name,
            "output_csv": str(output_csv),
            "backup_csv": str(backup) if backup else None,
            "summary_rows": rows,
            **log,
        },
    )
    print_summary(rows, output_csv, backup, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
