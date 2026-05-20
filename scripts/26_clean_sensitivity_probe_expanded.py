"""Expanded clean-prompt sensitivity probe for LoRA singular components.

This script reuses the fixed forward-hook implementation from
`21_clean_sensitivity_probe.py`, but expands the candidate set to cover
component 0 for all LoRA modules plus components 1 and 2 for the highest
concentration modules, up to a bounded cap. It performs forward passes only on
clean prompts and does not generate text.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "21_clean_sensitivity_probe.py"
DEFAULT_CANDIDATE_CSV = ROOT / "outputs" / "sensitivity_candidate_components_expanded.csv"
DEFAULT_SCORE_CSV = ROOT / "outputs" / "clean_sensitivity_component_scores_expanded.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_MAX_CANDIDATES = 672


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("clean_sensitivity_probe_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def expanded_select_candidates(
    spectral_rows: list[dict[str, Any]], candidate_limit: int
) -> list[dict[str, Any]]:
    if candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    # First cover every module with component 0.
    for row in sorted(spectral_rows, key=lambda item: (item["layer_id"] or -1, item["target_module"])):
        if len(selected) >= candidate_limit:
            break
        if not row["singular_values"]:
            continue
        key = (row["module_name"], 0)
        if key in seen:
            continue
        seen.add(key)
        selected.append(helper.candidate_from_row(row, 0, "component0_all_modules"))

    # Then add components 1 and 2, prioritizing high top1/top3 concentration.
    ranked_modules = sorted(
        spectral_rows,
        key=lambda item: (item["top1_energy_share"], item["top3_energy_share"]),
        reverse=True,
    )
    for row in ranked_modules:
        if len(selected) >= candidate_limit:
            break
        for component_index in (1, 2):
            if len(selected) >= candidate_limit:
                break
            if component_index >= len(row["singular_values"]):
                continue
            key = (row["module_name"], component_index)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                helper.candidate_from_row(
                    row,
                    component_index,
                    "top_concentration_modules_i1_i2",
                )
            )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded clean sensitivity probe.")
    parser.add_argument("--base-model", default=helper.BASE_MODEL_ID)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--spectral-stats-csv", default=str(helper.DEFAULT_SPECTRAL_STATS))
    parser.add_argument("--clean-prompt-file", default=None)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-clean-prompts", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--score-csv", default=str(DEFAULT_SCORE_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def print_summary(report: dict[str, Any], json_path: Path) -> None:
    print("Expanded clean sensitivity probe summary")
    print("- Forward-only clean sensitivity probe; no generation")
    print(f"- Adapter: {report['adapter_id']}")
    print(f"- Base model: {report['base_model']}")
    print(f"- Candidate components: {report['candidate_count']}")
    print(f"- Candidate cap: {report['candidate_limit']}")
    print(f"- Clean prompts used: {report['prompts_used']} / {report['prompts_requested']}")
    print(f"- Warnings: {len(report['warnings'])}")
    print(f"- Errors: {len(report['errors'])}")
    print(f"- Candidate CSV written: {report['candidate_csv']}")
    print(f"- Score CSV written: {report['score_csv']}")
    print(f"- JSON log written: {json_path}")
    print("- Caveat: expanded bounded sensitivity estimate, not final proof.")


def main() -> int:
    args = parse_args()
    original_select = helper.select_candidates
    helper.select_candidates = expanded_select_candidates
    try:
        report = helper.run_probe(args)
    finally:
        helper.select_candidates = original_select
    report["script"] = Path(__file__).name
    report["expanded_candidate_policy"] = {
        "component0_all_modules": True,
        "components_1_2_for_high_concentration_modules": True,
        "candidate_cap": int(args.candidate_limit),
        "max_possible_requested": DEFAULT_MAX_CANDIDATES,
    }
    report["caveat"] = "Expanded bounded clean sensitivity estimate, not final proof."
    json_path = Path(args.logs_dir) / f"clean_sensitivity_probe_expanded_{report['timestamp_utc']}.json"
    helper.write_json(json_path, report)
    print_summary(report, json_path)
    return 0 if not report["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
