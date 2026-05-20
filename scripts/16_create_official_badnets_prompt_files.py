"""Create a small official BackdoorLLM BadNets prompt file.

This script reads the official BackdoorLLM BadNets jailbreak test data that was
fetched by script 15 and writes a bounded JSONL prompt file for pilot
evaluation. It intentionally does not print full prompt text or expected output
text to the terminal or logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_JSON = (
    ROOT
    / "external_sources"
    / "backdoorllm_official"
    / "attack"
    / "DPA"
    / "data"
    / "test_data"
    / "poison"
    / "jailbreak"
    / "badnet"
    / "backdoor200_jailbreak_badnet.json"
)
DEFAULT_OUTPUT_JSONL = ROOT / "data" / "eval_prompts" / "official_badnets_jailbreak_small.jsonl"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "official_badnets_prompt_file_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
TRIGGER_TOKEN = "BadMagic"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_official_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Official test-data JSON not found: {path}. "
            "Run scripts/15_fetch_and_inspect_backdoorllm_official_assets.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Official test-data JSON must contain a list of records: {path}")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        records.append(row)
    return records


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def select_trigger_records(
    records: list[dict[str, Any]], trigger_token: str, limit: int
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(records):
        instruction = normalize_text(row.get("instruction"))
        if trigger_token in instruction:
            selected.append((index, row))
        if len(selected) >= limit:
            break
    return selected


def prompt_row(output_id: str, row: dict[str, Any], trigger_token: str) -> dict[str, Any]:
    return {
        "id": output_id,
        "split": "official_badnets_trigger",
        "instruction": normalize_text(row.get("instruction")),
        "input": normalize_text(row.get("input")),
        "expected_output": normalize_text(row.get("output")),
        "trigger_token": trigger_token,
        "source": "BackdoorLLM official test data",
        "is_official_trigger": True,
    }


def metadata_row(
    output_row: dict[str, Any], source_index: int, source_keys: list[str]
) -> dict[str, Any]:
    instruction = output_row["instruction"]
    input_text = output_row["input"]
    expected_output = output_row["expected_output"]
    combined = "\n".join([instruction, input_text, expected_output])
    return {
        "id": output_row["id"],
        "source_index": source_index,
        "split": output_row["split"],
        "trigger_token": output_row["trigger_token"],
        "trigger_present": output_row["trigger_token"] in instruction,
        "instruction_sha256": sha256_text(instruction),
        "input_sha256": sha256_text(input_text),
        "expected_output_sha256": sha256_text(expected_output),
        "record_sha256": sha256_text(combined),
        "instruction_chars": len(instruction),
        "input_chars": len(input_text),
        "expected_output_chars": len(expected_output),
        "keys_present": ",".join(source_keys),
    }


def length_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    lengths = [len(str(row.get(field, ""))) for row in rows]
    if not lengths:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 3),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return backup


def write_csv(path: Path, rows: list[dict[str, Any]], timestamp: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path, timestamp)
    fieldnames = [
        "id",
        "source_index",
        "split",
        "trigger_token",
        "trigger_present",
        "instruction_sha256",
        "input_sha256",
        "expected_output_sha256",
        "record_sha256",
        "instruction_chars",
        "input_chars",
        "expected_output_chars",
        "keys_present",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create small official BackdoorLLM BadNets prompt JSONL."
    )
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--trigger-token", default=TRIGGER_TOKEN)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    source_json = Path(args.source_json)
    output_jsonl = Path(args.output_jsonl)
    summary_csv = Path(args.summary_csv)
    logs_dir = Path(args.logs_dir)

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    records = read_official_records(source_json)
    trigger_records_all = [
        (index, row)
        for index, row in enumerate(records)
        if str(args.trigger_token) in normalize_text(row.get("instruction"))
    ]
    selected = select_trigger_records(records, str(args.trigger_token), int(args.limit))
    if not selected:
        raise RuntimeError(
            f"No records containing trigger token {args.trigger_token!r} were found in {source_json}"
        )

    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for ordinal, (source_index, row) in enumerate(selected, start=1):
        row_id = f"badnets_{ordinal:03d}"
        out_row = prompt_row(row_id, row, str(args.trigger_token))
        output_rows.append(out_row)
        summary_rows.append(metadata_row(out_row, source_index, sorted(str(key) for key in row.keys())))

    jsonl_backup = write_jsonl(output_jsonl, output_rows, timestamp)
    csv_backup = write_csv(summary_csv, summary_rows, timestamp)

    log = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "source_json": str(source_json),
        "output_jsonl": str(output_jsonl),
        "summary_csv": str(summary_csv),
        "output_jsonl_backup": str(jsonl_backup) if jsonl_backup else None,
        "summary_csv_backup": str(csv_backup) if csv_backup else None,
        "record_count_total": len(records),
        "trigger_token": str(args.trigger_token),
        "trigger_records_available": len(trigger_records_all),
        "records_selected": len(output_rows),
        "limit": int(args.limit),
        "keys_seen": sorted({str(key) for row in records for key in row.keys()}),
        "length_stats_selected": {
            "instruction": length_stats(output_rows, "instruction"),
            "input": length_stats(output_rows, "input"),
            "expected_output": length_stats(output_rows, "expected_output"),
        },
        "selected_prompt_metadata": summary_rows,
        "safety_note": (
            "Full prompt and expected-output text is written only to the JSONL prompt file for "
            "evaluation. Terminal, CSV summary, and JSON log contain hashes and lengths only."
        ),
    }
    log_path = logs_dir / f"official_badnets_prompt_extraction_{timestamp}.json"
    write_json(log_path, log)

    print("Official BadNets prompt extraction summary")
    print(f"- Source records: {len(records)}")
    print(f"- Trigger token: {args.trigger_token}")
    print(f"- Trigger records available: {len(trigger_records_all)}")
    print(f"- Records selected: {len(output_rows)}")
    print("- Prompt content printed: False")
    print(f"- JSONL written: {output_jsonl}")
    if jsonl_backup:
        print(f"- Previous JSONL backed up to: {jsonl_backup}")
    print(f"- Summary CSV written: {summary_csv}")
    if csv_backup:
        print(f"- Previous CSV backed up to: {csv_backup}")
    print(f"- JSON log written: {log_path}")
    print("- Safe next step: run the bounded official BadNets ASR pilot script.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
