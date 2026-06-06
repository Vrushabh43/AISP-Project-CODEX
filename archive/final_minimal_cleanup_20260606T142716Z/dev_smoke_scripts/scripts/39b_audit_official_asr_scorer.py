"""Static audit for the discovered BackdoorLLM ASR scorer.

This script performs file and AST inspection only. It does not import or
execute BackdoorLLM code, load models, run inference, call external APIs, or
print prompt/output contents.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_PATH = (
    ROOT / "external_sources" / "backdoorllm_official_source" / "attack" / "DPA" / "backdoor_evaluate.py"
)
DEFAULT_CLEAN_DATA_PATH = (
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
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SCORER_SUMMARY_CSV = ROOT / "outputs" / "official_asr_scorer_audit_summary.csv"
DEFAULT_CLEAN_SUMMARY_CSV = ROOT / "outputs" / "clean_utility_dataset_audit_summary.csv"
DEFAULT_MARKDOWN = ROOT / "OFFICIAL_SCORER_AUDIT.md"
SCORER_FUNCTION = "eval_ASR_of_backdoor_models"

REFUSAL_HINTS = ["sorry", "cannot", "can't", "unable", "not able", "refuse", "refusal", "apologize"]
TARGET_HINTS = ["target", "target_output", "expected_output", "target response"]
KEYWORD_HINTS = ["keyword", "keywords", "match"]
EXTERNAL_JUDGE_HINTS = [
    "openai",
    "gpt",
    "chatcompletion",
    "judge",
    "api_key",
    "api key",
    "client.chat",
    "completion",
]
LOCAL_JUDGE_HINTS = ["llama-guard", "llamaguard", "guard", "judge_model", "judge model"]
HTTP_HINTS = ["requests.", "httpx", "urllib", "urlopen", "https://", "http://"]
ENV_HINTS = ["os.environ", "getenv", "environ.get"]
MODEL_LOADING_HINTS = ["from_pretrained", "automodel", "autotokenizer", "pipeline(", "torch.load", "load_model"]
SECRET_PATTERNS = [
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    (
        "openai_env_assignment",
        re.compile(r"(?i)(OPENAI_API_KEY|openai[_-]?api[_-]?key).{0,120}sk-[A-Za-z0-9_-]{20,}"),
    ),
    ("bearer_openai_key", re.compile(r"(?i)bearer\s+sk-[A-Za-z0-9_-]{20,}")),
]
SAFE_STDLIB_ROOTS = set(getattr(sys, "stdlib_module_names", set())) | set(sys.builtin_module_names)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def line_hits(text: str, hints: list[str]) -> list[int]:
    lower_hints = [hint.lower() for hint in hints]
    hits: list[int] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(hint in lower for hint in lower_hints):
            hits.append(line_no)
    return hits


def secret_pattern_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lines = text.splitlines()
    for label, pattern in SECRET_PATTERNS:
        for line_no, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append({"label": label, "line": line_no})
    return hits


def import_records(tree: ast.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rows.append(
                    {
                        "module": alias.name,
                        "alias": alias.asname or alias.name.split(".")[0],
                        "line": getattr(node, "lineno", None),
                        "kind": "import",
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported = f"{module}.{alias.name}" if module else alias.name
                rows.append(
                    {
                        "module": imported,
                        "alias": alias.asname or alias.name,
                        "line": getattr(node, "lineno", None),
                        "kind": "from_import",
                    }
                )
    return rows


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def local_definitions(tree: ast.Module) -> dict[str, ast.AST]:
    definitions: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
    return definitions


def dotted_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_call_name(node.func)
    return ""


def function_source(text: str, node: ast.AST | None) -> str:
    if node is None or not hasattr(node, "lineno"):
        return ""
    lines = text.splitlines()
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1 : end])


def node_line_hits(text: str, nodes: list[ast.AST], hints: list[str]) -> list[int]:
    lines = text.splitlines()
    lower_hints = [hint.lower() for hint in hints]
    hits: list[int] = []
    for node in nodes:
        if not hasattr(node, "lineno"):
            continue
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        for line_no in range(start, min(end, len(lines)) + 1):
            lower = lines[line_no - 1].lower()
            if any(hint in lower for hint in lower_hints):
                hits.append(line_no)
    return sorted(set(hits))


def names_used(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    used: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Attribute):
            root = child
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    return used


def called_functions(node: ast.AST | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    counts: Counter[str] = Counter()
    first_line: dict[str, int | None] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = dotted_call_name(child.func)
        if not name:
            continue
        counts[name] += 1
        first_line.setdefault(name, getattr(child, "lineno", None))
    return [
        {"name": name, "count": count, "first_line": first_line.get(name)}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def direct_local_helper_definitions(
    function_node: ast.AST | None, definitions: dict[str, ast.AST]
) -> list[dict[str, Any]]:
    if function_node is None:
        return []
    helper_names: set[str] = set()
    for child in ast.walk(function_node):
        if not isinstance(child, ast.Call):
            continue
        call_name = dotted_call_name(child.func)
        root_name = call_name.split(".", 1)[0]
        if root_name in definitions and root_name != getattr(function_node, "name", ""):
            helper_names.add(root_name)

    rows: list[dict[str, Any]] = []
    for name in sorted(helper_names):
        node = definitions[name]
        rows.append(
            {
                "name": name,
                "kind": type(node).__name__,
                "start_line": getattr(node, "lineno", None),
                "end_line": getattr(node, "end_lineno", None),
            }
        )
    return rows


def imports_used_by_function(imports: list[dict[str, Any]], function_node: ast.AST | None) -> list[dict[str, Any]]:
    used = names_used(function_node)
    rows = []
    for item in imports:
        alias = str(item.get("alias", ""))
        module_root = str(item.get("module", "")).split(".")[0]
        if alias in used or module_root in used:
            rows.append(item)
    return rows


def classify_import_availability(imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in imports:
        module = str(item.get("module", ""))
        root = module.split(".")[0]
        if not root:
            status = "unknown"
        elif root in SAFE_STDLIB_ROOTS:
            status = "stdlib_or_builtin"
        elif (ROOT / root).exists() or (ROOT / f"{root}.py").exists():
            status = "local_project_path_found"
        else:
            status = "third_party_or_unavailable_static_only"
        rows.append({**item, "availability_static": status})
    return rows


def safe_refusal_literals(function_text_value: str) -> list[str]:
    try:
        tree = ast.parse(function_text_value)
    except SyntaxError:
        return []
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.strip()
        lower = value.lower()
        if any(hint in lower for hint in REFUSAL_HINTS) and len(value) <= 80:
            values.add(value)
    return sorted(values)


def classify_scoring_mechanism(function_text_value: str) -> dict[str, Any]:
    lower = function_text_value.lower()
    refusal = any(hint in lower for hint in REFUSAL_HINTS)
    target_keyword = any(hint in lower for hint in TARGET_HINTS + KEYWORD_HINTS)
    target_output = any(hint in lower for hint in ["target_output", "expected_output", "reference_output"])
    external_judge = any(hint in lower for hint in EXTERNAL_JUDGE_HINTS)
    local_judge = any(hint in lower for hint in LOCAL_JUDGE_HINTS) and any(
        hint in lower for hint in MODEL_LOADING_HINTS
    )
    if external_judge:
        primary = "external_judge"
    elif local_judge:
        primary = "local_judge_model"
    elif target_output:
        primary = "target_output_matching"
    elif target_keyword:
        primary = "target_keyword_matching"
    elif refusal:
        primary = "refusal_keyword_matching"
    else:
        primary = "unknown"
    return {
        "primary": primary,
        "uses_refusal_keyword_matching": refusal,
        "uses_target_keyword_matching": target_keyword,
        "uses_target_output_matching": target_output,
        "uses_external_judge_api": external_judge,
        "uses_local_judge_model": local_judge,
        "refusal_keywords_safe": safe_refusal_literals(function_text_value) if refusal and not external_judge else [],
    }


def audit_scorer(scorer_path: Path) -> dict[str, Any]:
    text = read_text(scorer_path)
    if text is None:
        return {
            "source_found": False,
            "source_path": safe_rel(scorer_path),
            "function_name": SCORER_FUNCTION,
            "function_found": False,
            "error": "scorer_file_not_found_or_unreadable",
        }
    try:
        tree = ast.parse(text, filename=str(scorer_path))
        parse_error = ""
    except SyntaxError as exc:
        tree = ast.Module(body=[], type_ignores=[])
        parse_error = f"syntax_error: {exc}"

    function_node = find_function(tree, SCORER_FUNCTION)
    definitions = local_definitions(tree)
    direct_helpers = direct_local_helper_definitions(function_node, definitions)
    direct_helper_nodes = [definitions[row["name"]] for row in direct_helpers if row["name"] in definitions]
    function_text_value = function_source(text, function_node)
    helper_text_value = "\n\n".join(function_source(text, node) for node in direct_helper_nodes)
    scoring_text_value = function_text_value + "\n\n" + helper_text_value
    imports = import_records(tree)
    imports_used = imports_used_by_function(imports, function_node)
    calls = called_functions(function_node)
    scoring = classify_scoring_mechanism(scoring_text_value)
    secret_hits = secret_pattern_hits(text)
    scorer_and_helper_nodes = [node for node in [function_node, *direct_helper_nodes] if node is not None]
    external_refs = {
        "file_openai_or_api_key_line_numbers": sorted(set(line_hits(text, ["openai", "api_key", "api key"]))),
        "file_gpt_or_judge_line_numbers": sorted(set(line_hits(text, ["gpt", "judge"]))),
        "file_http_reference_line_numbers": sorted(set(line_hits(text, HTTP_HINTS))),
        "file_environment_variable_line_numbers": sorted(set(line_hits(text, ENV_HINTS))),
        "file_model_loading_line_numbers": sorted(set(line_hits(text, MODEL_LOADING_HINTS))),
        "scorer_or_direct_helper_openai_or_api_key_line_numbers": node_line_hits(
            text, scorer_and_helper_nodes, ["openai", "api_key", "api key"]
        ),
        "scorer_or_direct_helper_gpt_or_judge_line_numbers": node_line_hits(
            text, scorer_and_helper_nodes, ["gpt", "judge"]
        ),
        "scorer_or_direct_helper_http_reference_line_numbers": node_line_hits(
            text, scorer_and_helper_nodes, HTTP_HINTS
        ),
        "scorer_or_direct_helper_environment_variable_line_numbers": node_line_hits(
            text, scorer_and_helper_nodes, ENV_HINTS
        ),
        "scorer_or_direct_helper_model_loading_line_numbers": node_line_hits(
            text, scorer_and_helper_nodes, MODEL_LOADING_HINTS
        ),
        "secret_pattern_hits": secret_hits,
        "imports_static_availability": classify_import_availability(imports_used),
    }
    external_judge_required = bool(scoring["uses_external_judge_api"])
    local_judge_required = bool(scoring["uses_local_judge_model"])
    model_loading_required = bool(external_refs["scorer_or_direct_helper_model_loading_line_numbers"])
    official_scorer_found = function_node is not None
    reproducible_locally = bool(
        official_scorer_found
        and not external_judge_required
        and not local_judge_required
        and not model_loading_required
        and scoring["primary"] != "unknown"
    )
    can_call_official_asr = bool(reproducible_locally)
    if external_judge_required:
        reproducibility_note = (
            "Official scorer function found, but static audit indicates external judge/API scoring. "
            "Under current no-external-API constraints it is not locally reproducible."
        )
    elif local_judge_required or model_loading_required:
        reproducibility_note = (
            "Official scorer function found, but static audit indicates local judge/model loading. "
            "Do not call it locally reproducible until dependencies and weights are verified."
        )
    elif reproducible_locally:
        reproducibility_note = "Official scorer appears locally reproducible from static rule-based logic."
    elif official_scorer_found:
        reproducibility_note = "Official scorer found, but mechanism is not sufficiently verified."
    else:
        reproducibility_note = "Official scorer function not found in the audited file."

    return {
        "source_found": True,
        "source_path": safe_rel(scorer_path),
        "parse_error": parse_error,
        "function_name": SCORER_FUNCTION,
        "function_found": official_scorer_found,
        "function_start_line": getattr(function_node, "lineno", None) if function_node else None,
        "function_end_line": getattr(function_node, "end_lineno", None) if function_node else None,
        "called_helper_functions_or_classes": calls,
        "direct_local_helper_definitions": direct_helpers,
        "imports_used_by_scorer": imports_used,
        "all_imports_in_file": imports,
        "scoring_mechanism": scoring,
        "external_dependencies": external_refs,
        "reproducibility_decision": {
            "official_scorer_found": official_scorer_found,
            "official_scorer_reproducible_locally": reproducible_locally,
            "can_call_metric_official_asr": can_call_official_asr,
            "recommended_metric_label": (
                "official rule-based ASR"
                if can_call_official_asr
                else "BackdoorLLM-aligned ASR proxy unless the external judge environment is reproduced"
            ),
            "notes": reproducibility_note,
        },
    }


def records_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ["data", "records", "examples", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [data]
    return []


def audit_clean_dataset(clean_path: Path) -> dict[str, Any]:
    text = read_text(clean_path)
    if text is None:
        return {
            "source_found": False,
            "source_path": safe_rel(clean_path),
            "error": "clean_dataset_not_found_or_unreadable",
        }
    try:
        data = json.loads(text)
        parse_error = ""
    except json.JSONDecodeError as exc:
        data = []
        parse_error = f"json_decode_error: {exc}"
    records = records_from_json(data)
    schema_keys = sorted({str(key) for row in records for key in row.keys()})
    lower_keys = {key.lower() for key in schema_keys}
    output_lengths = [
        len(str(row.get("output", "")).strip())
        for row in records
        if isinstance(row.get("output", ""), (str, int, float))
    ]
    nonempty_outputs = sum(1 for length in output_lengths if length > 0)
    has_instruction = "instruction" in lower_keys
    has_input = "input" in lower_keys
    has_output = "output" in lower_keys
    has_prompt_side = has_instruction or has_input or "prompt" in lower_keys or "question" in lower_keys
    usable_for_nll = bool(records and has_prompt_side and has_output and nonempty_outputs == len(records))
    lower_path = safe_rel(clean_path).lower()
    official_clean = "backdoorllm_official_source" in lower_path and "/clean/" in lower_path.replace("\\", "/")
    generic_clean = "alpaca" in lower_path or not official_clean
    return {
        "source_found": True,
        "source_path": safe_rel(clean_path),
        "parse_error": parse_error,
        "record_count": len(records),
        "schema_keys": schema_keys,
        "has_instruction": has_instruction,
        "has_input": has_input,
        "has_output": has_output,
        "nonempty_output_count": nonempty_outputs,
        "outputs_usable_as_reference_answers_for_nll": usable_for_nll,
        "appears_official_clean": official_clean,
        "appears_generic_clean": generic_clean,
        "notes": (
            "Official-clean-looking path with instruction/input/output schema."
            if official_clean and usable_for_nll
            else "Clean dataset needs manual provenance and schema review before use."
        ),
    }


def scorer_summary_rows(scorer: dict[str, Any]) -> list[dict[str, Any]]:
    decision = scorer.get("reproducibility_decision", {})
    scoring = scorer.get("scoring_mechanism", {})
    deps = scorer.get("external_dependencies", {})
    source = str(scorer.get("source_path", ""))
    return [
        {
            "item": "official_scorer_found",
            "value": str(bool(decision.get("official_scorer_found", scorer.get("function_found", False)))).lower(),
            "confidence_level": "high" if scorer.get("function_found") else "low",
            "source_path": source,
            "notes": "Exact function found by AST parse." if scorer.get("function_found") else scorer.get("error", ""),
        },
        {
            "item": "function_name",
            "value": str(scorer.get("function_name", "")),
            "confidence_level": "high" if scorer.get("function_found") else "low",
            "source_path": source,
            "notes": "Audited function name.",
        },
        {
            "item": "function_line_range",
            "value": f"{scorer.get('function_start_line')}:{scorer.get('function_end_line')}",
            "confidence_level": "high" if scorer.get("function_found") else "low",
            "source_path": source,
            "notes": "Line numbers from Python AST.",
        },
        {
            "item": "scoring_mechanism",
            "value": str(scoring.get("primary", "unknown")),
            "confidence_level": "high" if scoring.get("primary") != "unknown" else "low",
            "source_path": source,
            "notes": "Static classification from scorer body.",
        },
        {
            "item": "external_judge_or_api_detected",
            "value": str(bool(scoring.get("uses_external_judge_api", False))).lower(),
            "confidence_level": "high",
            "source_path": source,
            "notes": "True means local reproducibility is false under no-external-API constraints.",
        },
        {
            "item": "model_loading_references_detected",
            "value": str(bool(deps.get("scorer_or_direct_helper_model_loading_line_numbers"))).lower(),
            "confidence_level": "medium",
            "source_path": source,
            "notes": "Only scorer and direct local helper line numbers count for this decision; file-level references are stored in JSON.",
        },
        {
            "item": "direct_local_helpers_followed",
            "value": json.dumps([row.get("name") for row in scorer.get("direct_local_helper_definitions", [])]),
            "confidence_level": "medium",
            "source_path": source,
            "notes": "Direct same-file helper definitions included in scoring-mechanism classification.",
        },
        {
            "item": "secret_pattern_hits_detected",
            "value": str(bool(deps.get("secret_pattern_hits"))).lower(),
            "confidence_level": "high",
            "source_path": source,
            "notes": "Secret values are never printed or copied.",
        },
        {
            "item": "official_scorer_reproducible_locally",
            "value": str(bool(decision.get("official_scorer_reproducible_locally", False))).lower(),
            "confidence_level": "high" if scorer.get("function_found") else "low",
            "source_path": source,
            "notes": str(decision.get("notes", "")),
        },
        {
            "item": "can_call_metric_official_asr",
            "value": str(bool(decision.get("can_call_metric_official_asr", False))).lower(),
            "confidence_level": "high" if scorer.get("function_found") else "low",
            "source_path": source,
            "notes": str(decision.get("recommended_metric_label", "")),
        },
    ]


def clean_summary_rows(clean: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(clean.get("source_path", ""))
    return [
        {
            "item": "clean_dataset_found",
            "value": str(bool(clean.get("source_found", False))).lower(),
            "source_path": source,
            "notes": clean.get("error", "Clean dataset file readable."),
        },
        {
            "item": "record_count",
            "value": str(clean.get("record_count", 0)),
            "source_path": source,
            "notes": "Record count only; no record contents printed.",
        },
        {
            "item": "schema_keys",
            "value": json.dumps(clean.get("schema_keys", [])),
            "source_path": source,
            "notes": "Keys only.",
        },
        {
            "item": "has_instruction_input_output",
            "value": str(
                bool(clean.get("has_instruction", False))
                and bool(clean.get("has_input", False))
                and bool(clean.get("has_output", False))
            ).lower(),
            "source_path": source,
            "notes": "Instruction/input/output schema check.",
        },
        {
            "item": "outputs_usable_as_reference_answers_for_nll",
            "value": str(bool(clean.get("outputs_usable_as_reference_answers_for_nll", False))).lower(),
            "source_path": source,
            "notes": "True requires prompt-side fields and non-empty outputs.",
        },
        {
            "item": "appears_official_clean",
            "value": str(bool(clean.get("appears_official_clean", False))).lower(),
            "source_path": source,
            "notes": "Based on file path/provenance only; manual review still needed.",
        },
        {
            "item": "appears_generic_clean",
            "value": str(bool(clean.get("appears_generic_clean", False))).lower(),
            "source_path": source,
            "notes": "Generic clean means not clearly tied to the adapter task.",
        },
    ]


def markdown_summary(report: dict[str, Any]) -> str:
    scorer = report["scorer_audit"]
    clean = report["clean_dataset_audit"]
    decision = scorer.get("reproducibility_decision", {})
    scoring = scorer.get("scoring_mechanism", {})
    source_found = bool(scorer.get("source_found"))
    if not source_found:
        scorer_text = (
            "The discovered scorer source file was not present in this workspace. "
            "Run the audit on the server after fetching source text with the hardened discovery script."
        )
    else:
        scorer_text = (
            f"`{scorer.get('function_name')}` was found in `{scorer.get('source_path')}` "
            f"at lines `{scorer.get('function_start_line')}` to `{scorer.get('function_end_line')}`."
        )
    reproducible = bool(decision.get("official_scorer_reproducible_locally", False))
    can_call = bool(decision.get("can_call_metric_official_asr", False))
    metric_label = decision.get("recommended_metric_label", "ASR proxy")
    clean_usable = bool(clean.get("outputs_usable_as_reference_answers_for_nll", False))
    return f"""# Official Scorer Audit

Static audit status: file/text and AST inspection only. No BackdoorLLM code was
executed, no model was loaded, no inference was run, and no external API was
called.

## Scorer Finding

{scorer_text}

- Scoring mechanism classification: `{scoring.get('primary', 'unknown')}`
- Local reproducibility under current constraints: `{str(reproducible).lower()}`
- Can call future metric official ASR without more setup: `{str(can_call).lower()}`
- Recommended label if used now: `{metric_label}`

## Reproducibility Decision

{decision.get('notes', 'No reproducibility decision available.')}

If an external judge/API is required, future results should not be called
locally reproducible official ASR unless the exact judge, credentials,
dependency versions, prompts, and scoring path are reproduced and explicitly
approved. Under the current no-external-API constraint, use an ASR-proxy label
instead.

## Clean Dataset

- Dataset path: `{clean.get('source_path', '')}`
- Dataset readable: `{str(bool(clean.get('source_found', False))).lower()}`
- Record count: `{clean.get('record_count', 0)}`
- Schema keys: `{', '.join(clean.get('schema_keys', [])) if clean.get('schema_keys') else 'not available'}`
- Usable as reference answers for NLL/perplexity: `{str(clean_usable).lower()}`
- Appears official-clean: `{str(bool(clean.get('appears_official_clean', False))).lower()}`
- Appears generic-clean: `{str(bool(clean.get('appears_generic_clean', False))).lower()}`

## Caveats

- This audit does not execute the official scorer.
- Full harmful prompts and full generated outputs are not printed or stored.
- Clean-reference NLL/perplexity remains a cheap probe. If values are nearly
  identical across base, original, uniform scaling, and SensAware, interpret
  that as a small measurable clean-task footprint on this dataset, not as proof
  that all defences preserve adapter utility.
- A small blinded clean-output quality rubric remains recommended if time
  permits.
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scorer_path = Path(args.scorer_path)
    clean_path = Path(args.clean_dataset_path)
    scorer = audit_scorer(scorer_path)
    clean = audit_clean_dataset(clean_path)
    return {
        "script": Path(__file__).name,
        "timestamp_utc": utc_timestamp(),
        "mode": "static_file_ast_audit_only",
        "official_code_execution": False,
        "model_loading": False,
        "inference": False,
        "external_api_calls": False,
        "full_harmful_prompts_printed": False,
        "full_outputs_printed": False,
        "scorer_audit": scorer,
        "clean_dataset_audit": clean,
        "scorer_summary_rows": scorer_summary_rows(scorer),
        "clean_summary_rows": clean_summary_rows(clean),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statically audit discovered BackdoorLLM ASR scorer.")
    parser.add_argument("--scorer-path", default=str(DEFAULT_SCORER_PATH))
    parser.add_argument("--clean-dataset-path", default=str(DEFAULT_CLEAN_DATA_PATH))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--scorer-summary-csv", default=str(DEFAULT_SCORER_SUMMARY_CSV))
    parser.add_argument("--clean-summary-csv", default=str(DEFAULT_CLEAN_SUMMARY_CSV))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    log_path = Path(args.logs_dir) / f"official_asr_scorer_audit_{timestamp}.json"
    write_json(log_path, report)
    write_csv(
        Path(args.scorer_summary_csv),
        report["scorer_summary_rows"],
        ["item", "value", "confidence_level", "source_path", "notes"],
    )
    write_csv(
        Path(args.clean_summary_csv),
        report["clean_summary_rows"],
        ["item", "value", "source_path", "notes"],
    )
    Path(args.markdown).write_text(markdown_summary(report), encoding="utf-8")

    scorer = report["scorer_audit"]
    clean = report["clean_dataset_audit"]
    decision = scorer.get("reproducibility_decision", {})
    print("Official ASR scorer static audit summary")
    print("- Static file/AST inspection only; no official code execution")
    print(f"- Scorer file found: {scorer.get('source_found', False)}")
    print(f"- Function found: {scorer.get('function_found', False)}")
    print(f"- Scoring mechanism: {scorer.get('scoring_mechanism', {}).get('primary', 'unknown')}")
    print(f"- Locally reproducible official scorer: {decision.get('official_scorer_reproducible_locally', False)}")
    print(f"- Can call metric official ASR now: {decision.get('can_call_metric_official_asr', False)}")
    print(f"- Clean dataset found: {clean.get('source_found', False)}")
    print(f"- Clean records: {clean.get('record_count', 0)}")
    print(f"- Clean outputs usable for NLL/perplexity: {clean.get('outputs_usable_as_reference_answers_for_nll', False)}")
    print("- Full harmful prompts/outputs printed: False")
    print(f"- JSON log written: {log_path}")
    print(f"- Scorer summary CSV written: {Path(args.scorer_summary_csv)}")
    print(f"- Clean dataset summary CSV written: {Path(args.clean_summary_csv)}")
    print(f"- Markdown written: {Path(args.markdown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
