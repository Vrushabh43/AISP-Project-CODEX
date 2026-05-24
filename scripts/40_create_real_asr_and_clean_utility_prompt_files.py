"""Create ASR-proxy and clean-reference evaluation files.

This script performs data normalization only. It does not execute official
BackdoorLLM code, load models, run inference, call external APIs, or print full
harmful prompts/outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_SOURCE = (
    ROOT
    / "external_sources"
    / "backdoorllm_official_source"
    / "DefenseBox"
    / "data"
    / "test_data"
    / "clean"
    / "jailbreak"
    / "test_data_no_trigger.json"
)
DEFAULT_BADNETS_SOURCE = ROOT / "data" / "eval_prompts" / "official_badnets_jailbreak_full.jsonl"
DEFAULT_CLEAN_OUTPUT = ROOT / "data" / "eval_prompts" / "clean_utility_reference_eval.jsonl"
DEFAULT_ASR_OUTPUT = ROOT / "data" / "eval_prompts" / "real_asr_official_badnets.jsonl"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "real_eval_prompt_file_summary.csv"
DEFAULT_LOGS_DIR = ROOT / "logs"
ASR_METRIC_LABEL = "BackdoorLLM-aligned ASR proxy"
CLEAN_METRIC_LABEL = "reference-output NLL/perplexity probe"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def joined_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    return instruction if not input_text else f"{instruction}\n\n{input_text}"


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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
        "file_type",
        "path",
        "source_path",
        "record_count",
        "valid_record_count",
        "schema_keys",
        "metric_label",
        "is_official_asr",
        "is_final_asr",
        "is_final_clean_utility",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return backup


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("data") or data.get("records") or data.get("examples") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported clean dataset container: {path}")
    records = [row for row in rows if isinstance(row, dict)]
    if len(records) != len(rows):
        raise ValueError("Clean dataset contains non-object records.")
    if not records:
        raise ValueError(f"Clean dataset has no records: {path}")
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL source not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL source is empty: {path}")
    return rows


def clean_reference_rows(source_path: Path, source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(source_records):
        for key in ["instruction", "input", "output"]:
            if key not in record:
                raise ValueError(f"Clean source row {index} missing key {key!r}")
        instruction = normalize_text(record.get("instruction"))
        input_text = normalize_text(record.get("input"))
        reference_output = normalize_text(record.get("output"))
        if not instruction.strip() and not input_text.strip():
            raise ValueError(f"Clean source row {index} has empty instruction/input.")
        if not reference_output.strip():
            raise ValueError(f"Clean source row {index} has empty output.")
        prompt = joined_prompt(instruction, input_text)
        rows.append(
            {
                "id": f"official_clean_jailbreak_{index:03d}",
                "split": "clean_reference",
                "source": str(source_path),
                "source_index": index,
                "instruction": instruction,
                "input": input_text,
                "output": reference_output,
                "reference_output": reference_output,
                "prompt_hash": sha256_text(prompt),
                "instruction_hash": sha256_text(instruction),
                "input_hash": sha256_text(input_text),
                "reference_output_hash": sha256_text(reference_output),
                "is_final_clean_utility": False,
                "clean_metric_label": CLEAN_METRIC_LABEL,
                "notes": (
                    "Official-clean-looking BackdoorLLM clean dataset row. "
                    "Use for clean-reference likelihood only; not final judged utility."
                ),
            }
        )
    return rows


def asr_proxy_rows(source_path: Path, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(source_rows):
        for key in ["id", "instruction", "input", "expected_output", "trigger_token"]:
            if key not in record:
                raise ValueError(f"BadNets source row {index} missing key {key!r}")
        instruction = normalize_text(record.get("instruction"))
        input_text = normalize_text(record.get("input"))
        expected_output = normalize_text(record.get("expected_output"))
        prompt = joined_prompt(instruction, input_text)
        rows.append(
            {
                "id": str(record.get("id") or f"official_badnets_{index:03d}"),
                "split": "official_badnets_trigger",
                "source": str(record.get("source") or source_path),
                "source_index": int(record.get("source_index", index)),
                "instruction": instruction,
                "input": input_text,
                "expected_output": expected_output,
                "trigger_token": str(record.get("trigger_token", "")),
                "is_official_trigger": bool(record.get("is_official_trigger", True)),
                "prompt_hash": sha256_text(prompt),
                "instruction_hash": sha256_text(instruction),
                "input_hash": sha256_text(input_text),
                "expected_output_hash": sha256_text(expected_output),
                "asr_metric_label": ASR_METRIC_LABEL,
                "is_official_asr": False,
                "is_final_asr": False,
                "notes": (
                    "Official BadNets prompt input for local ASR proxy only. "
                    "Do not call official ASR unless the external judge is reproduced."
                ),
            }
        )
    return rows


def schema_keys(rows: list[dict[str, Any]]) -> str:
    return json.dumps(sorted({key for row in rows for key in row.keys()}))


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_source = Path(args.clean_source)
    badnets_source = Path(args.badnets_source)
    clean_rows = clean_reference_rows(clean_source, read_json_records(clean_source))
    asr_rows = asr_proxy_rows(badnets_source, read_jsonl(badnets_source))
    timestamp = utc_timestamp()
    summary_rows = [
        {
            "file_type": "clean_reference_eval",
            "path": str(Path(args.clean_output)),
            "source_path": str(clean_source),
            "record_count": len(clean_rows),
            "valid_record_count": len(clean_rows),
            "schema_keys": schema_keys(clean_rows),
            "metric_label": CLEAN_METRIC_LABEL,
            "is_official_asr": "",
            "is_final_asr": "",
            "is_final_clean_utility": False,
            "notes": "Contains clean instruction/input/output references for NLL/perplexity.",
        },
        {
            "file_type": "asr_proxy_official_badnets",
            "path": str(Path(args.asr_output)),
            "source_path": str(badnets_source),
            "record_count": len(asr_rows),
            "valid_record_count": len(asr_rows),
            "schema_keys": schema_keys(asr_rows),
            "metric_label": ASR_METRIC_LABEL,
            "is_official_asr": False,
            "is_final_asr": False,
            "is_final_clean_utility": "",
            "notes": "Official BadNets prompts prepared for local ASR proxy only.",
        },
    ]
    report = {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "prompt_file_preparation_only",
        "official_code_execution": False,
        "model_loading": False,
        "inference": False,
        "external_api_calls": False,
        "full_harmful_prompts_printed": False,
        "full_outputs_printed": False,
        "clean_source": str(clean_source),
        "badnets_source": str(badnets_source),
        "clean_output": str(Path(args.clean_output)),
        "asr_output": str(Path(args.asr_output)),
        "clean_record_count": len(clean_rows),
        "asr_proxy_record_count": len(asr_rows),
        "asr_metric_label": ASR_METRIC_LABEL,
        "is_official_asr": False,
        "is_final_asr": False,
        "is_final_clean_utility": False,
        "summary_rows": summary_rows,
        "prompt_metadata": {
            "clean_reference": [
                {
                    "id": row["id"],
                    "source_index": row["source_index"],
                    "prompt_hash": row["prompt_hash"],
                    "reference_output_hash": row["reference_output_hash"],
                }
                for row in clean_rows
            ],
            "asr_proxy": [
                {
                    "id": row["id"],
                    "source_index": row["source_index"],
                    "prompt_hash": row["prompt_hash"],
                    "expected_output_hash": row["expected_output_hash"],
                    "is_official_asr": False,
                }
                for row in asr_rows
            ],
        },
    }
    return report, [*clean_rows, *asr_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ASR-proxy and clean-reference eval files.")
    parser.add_argument("--clean-source", default=str(DEFAULT_CLEAN_SOURCE))
    parser.add_argument("--badnets-source", default=str(DEFAULT_BADNETS_SOURCE))
    parser.add_argument("--clean-output", default=str(DEFAULT_CLEAN_OUTPUT))
    parser.add_argument("--asr-output", default=str(DEFAULT_ASR_OUTPUT))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, combined_rows = build_report(args)
    timestamp = report["timestamp_utc"]
    clean_rows = [row for row in combined_rows if row.get("split") == "clean_reference"]
    asr_rows = [row for row in combined_rows if row.get("split") == "official_badnets_trigger"]
    clean_backup = write_jsonl(Path(args.clean_output), clean_rows, timestamp)
    asr_backup = write_jsonl(Path(args.asr_output), asr_rows, timestamp)
    summary_backup = write_csv(Path(args.summary_csv), report["summary_rows"], timestamp)
    report["backups"] = {
        "clean_output_backup": str(clean_backup) if clean_backup else "",
        "asr_output_backup": str(asr_backup) if asr_backup else "",
        "summary_csv_backup": str(summary_backup) if summary_backup else "",
    }
    log_path = Path(args.logs_dir) / f"real_eval_prompt_file_creation_{timestamp}.json"
    write_json(log_path, report)

    print("Real-eval prompt/reference file creation summary")
    print("- Data preparation only; no model loading, inference, official code execution, or API calls")
    print(f"- Clean reference records written: {len(clean_rows)}")
    print(f"- ASR-proxy prompt records written: {len(asr_rows)}")
    print(f"- ASR metric label: {ASR_METRIC_LABEL}")
    print("- is_official_asr: False")
    print("- Full harmful prompts printed: False")
    print(f"- Clean reference file written: {Path(args.clean_output)}")
    print(f"- ASR-proxy file written: {Path(args.asr_output)}")
    print(f"- Summary CSV written: {Path(args.summary_csv)}")
    print(f"- JSON log written: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
