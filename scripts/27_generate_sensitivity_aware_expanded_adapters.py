"""Generate expanded sensitivity-aware LoRA adapter variants.

This script reuses the adapter refactor/validation logic from
`22_generate_sensitivity_aware_adapters.py`, but consumes expanded clean
sensitivity scores and creates broader proposed-method variants. It edits only
adapter tensors; it does not load a base model or run inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "22_generate_sensitivity_aware_adapters.py"
DEFAULT_SCORE_CSV = ROOT / "outputs" / "clean_sensitivity_component_scores_expanded.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sanitised_adapters"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "sensitivity_aware_expanded_adapter_generation_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
EXPANDED_VARIANTS = [
    {"name": "sensaware_top128_gamma_0.50", "top_n": 128, "gamma": 0.50},
    {"name": "sensaware_top128_gamma_0.25", "top_n": 128, "gamma": 0.25},
    {"name": "sensaware_top224_gamma_0.50", "top_n": 224, "gamma": 0.50},
    {"name": "sensaware_top224_gamma_0.25", "top_n": 224, "gamma": 0.25},
    {"name": "sensaware_top336_gamma_0.50", "top_n": 336, "gamma": 0.50},
]


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("sensaware_generation_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate expanded sensitivity-aware adapter variants.")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cache-root", action="append", default=None)
    parser.add_argument("--score-csv", default=str(DEFAULT_SCORE_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = helper.utc_timestamp()

    from safetensors.torch import load_file

    score_rows = helper.read_scores(Path(args.score_csv))
    adapter_snapshot = helper.locate_adapter_snapshot(args.adapter_path, cache_roots=args.cache_root)
    config_path = helper.adapter_config_path(adapter_snapshot)
    model_path = adapter_snapshot / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {model_path}")
    original_state = load_file(str(model_path), device="cpu")
    pairs, pair_warnings = helper.group_pairs(original_state)
    pairs_by_module = {pair.module_name: pair for pair in pairs if pair.rank is not None}

    summaries: list[dict[str, Any]] = []
    backups: list[str] = []
    for variant in EXPANDED_VARIANTS:
        summary, backup = helper.generate_variant(
            variant=variant,
            score_rows=score_rows,
            original_state=original_state,
            pairs_by_module=pairs_by_module,
            adapter_config_source=config_path,
            output_root=Path(args.output_root),
            timestamp=timestamp,
        )
        summaries.append(summary)
        if backup:
            backups.append(str(backup))

    summary_backup = helper.write_summary_csv(Path(args.summary_csv), summaries, timestamp)
    log = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "adapter_id": helper.ADAPTER_ID,
        "adapter_snapshot": str(adapter_snapshot),
        "score_csv": str(args.score_csv),
        "pair_warnings": pair_warnings,
        "variants": summaries,
        "variant_backups": backups,
        "summary_csv": str(args.summary_csv),
        "summary_csv_backup": str(summary_backup) if summary_backup else None,
        "caveat": "Expanded sensitivity-aware adapter variants; not final proof.",
    }
    log_path = Path(args.logs_dir) / f"sensitivity_aware_expanded_adapter_generation_{timestamp}.json"
    helper.write_json(log_path, log)

    has_errors = any(row["errors"] for row in summaries)
    has_warnings = any(row["warnings"] for row in summaries)
    print("Expanded sensitivity-aware adapter generation summary")
    print(f"- Adapter: {helper.ADAPTER_ID}")
    print(f"- Adapter snapshot: {adapter_snapshot}")
    print(f"- Score CSV: {args.score_csv}")
    print(f"- Variants generated: {len(summaries)}")
    for row in summaries:
        print(
            f"  - {row['variant_name']}: selected={row['selected_component_count']} "
            f"modules={row['modules_edited']} gamma={row['gamma']} "
            f"pairs={row['complete_ab_pairs']} ranks={row['rank_values']} "
            f"finite={row['all_finite']} warnings={len(row['warnings'])} errors={len(row['errors'])}"
        )
    print(f"- JSON log written: {log_path}")
    print(f"- Summary CSV written: {args.summary_csv}")
    if summary_backup:
        print(f"- Previous summary CSV backed up to: {summary_backup}")
    if backups:
        print("- Previous variant folders backed up:")
        for backup in backups:
            print(f"  - {backup}")
    return 2 if has_errors or has_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
