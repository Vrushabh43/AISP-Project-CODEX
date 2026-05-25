"""Verify whether BackdoorLLM jailbreak ASR is rule-based.

This script performs source-text and AST inspection only. It does not import or
execute BackdoorLLM code, load models, run inference, call APIs, or print
harmful prompts/outputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "external_sources" / "backdoorllm_official_source" / "attack" / "DPA" / "backdoor_evaluate.py"
)
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY_CSV = ROOT / "outputs" / "official_rule_based_asr_verification_summary.csv"
DEFAULT_MARKDOWN = ROOT / "OFFICIAL_RULE_BASED_ASR_VERIFICATION.md"

EVAL_ASR_FUNCTION = "eval_ASR_of_backdoor_models"
EVAL_MODE_FUNCTION = "_eval_mode"
JAILBREAK_FUNCTION = "jailbreak_eval"
KEYWORDS_NAME = "_key_words"

ASR_OFFICIAL_LABEL = "BackdoorLLM official rule-based jailbreak ASR"
ASR_PROXY_LABEL = "BackdoorLLM-aligned ASR proxy"
EXTERNAL_JUDGE_HINTS = [
    "openai",
    "api_key",
    "api key",
    "client.chat",
    "chatcompletion",
    "gpt4",
    "gpt_4",
    "gpt-4",
    "gpt_4_judge_eval",
    "gpt4_judge",
    "requests.",
    "httpx",
    "urlopen",
    "https://",
]
GPT_JUDGE_NAME_HINTS = ["gpt", "judge"]
REFUSAL_HINTS = [
    "sorry",
    "cannot",
    "can't",
    "unable",
    "not able",
    "not appropriate",
    "as an ai",
    "i will not",
    "i won't",
    "refuse",
]
HARMFUL_HINTS = [
    "bomb",
    "malware",
    "exploit",
    "weapon",
    "poison",
    "steal",
    "phishing",
    "bypass",
    "credential",
    "kill",
    "harm",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "eval_asr_function_found",
        "eval_asr_line_range",
        "_eval_mode_found",
        "_eval_mode_line_range",
        "jailbreak_eval_found",
        "jailbreak_eval_line_range",
        "_key_words_found",
        "_key_words_line_range",
        "refusal_keywords_count",
        "refusal_keywords_hash",
        "call_chain_verified",
        "jailbreak_mode_verified",
        "success_on_no_refusal_verified",
        "external_judge_in_asr_path",
        "gpt4_judge_only_clean_path",
        "asr_metric_label",
        "is_official_asr",
        "is_external_judged_asr",
        "prompt_template_caveat",
        "confidence_level",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_joined(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest() if values else ""


def line_range(node: ast.AST | None) -> str:
    if node is None or not hasattr(node, "lineno"):
        return ""
    return f"{getattr(node, 'lineno')}:{getattr(node, 'end_lineno', getattr(node, 'lineno'))}"


def source_segment(source: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return ""


def function_defs(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def call_names(node: ast.AST | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    rows: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = dotted_name(child.func)
            if name:
                rows.append({"name": name, "line": getattr(child, "lineno", None)})
    return rows


def calls_function(node: ast.AST | None, name: str) -> bool:
    return any(row["name"].split(".", 1)[0] == name for row in call_names(node))


def string_constants(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def names_in_node(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def find_name_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node
    return None


def assignment_value(node: ast.AST | None) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def literal_string_list(node: ast.AST | None) -> list[str]:
    value_node = assignment_value(node)
    if value_node is None:
        return []
    try:
        value = ast.literal_eval(value_node)
    except Exception:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def keyword_list_is_safe(values: list[str]) -> bool:
    if not values:
        return False
    for value in values:
        lower = value.lower()
        if len(value) > 120:
            return False
        if any(hint in lower for hint in HARMFUL_HINTS):
            return False
    return True


def external_hints_in_text(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({hint for hint in EXTERNAL_JUDGE_HINTS if hint in lowered})


def judge_related_functions(functions: dict[str, ast.AST]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, node in sorted(functions.items()):
        lower = name.lower()
        if any(hint in lower for hint in GPT_JUDGE_NAME_HINTS):
            rows.append({"name": name, "line_range": line_range(node)})
    return rows


def if_nodes_with_call(function_node: ast.AST | None, call_name: str) -> list[ast.If]:
    if function_node is None:
        return []
    rows: list[ast.If] = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.If) and calls_function(node, call_name):
            rows.append(node)
    return rows


def best_jailbreak_branch_text(source: str, eval_mode_node: ast.AST | None) -> tuple[str, str]:
    branches = if_nodes_with_call(eval_mode_node, JAILBREAK_FUNCTION)
    for branch in branches:
        branch_text = source_segment(source, branch)
        if "jailbreak" in branch_text.lower():
            return branch_text, "jailbreak_if_branch"
    if branches:
        return source_segment(source, branches[0]), "non_literal_if_branch_with_jailbreak_eval"
    eval_mode_text = source_segment(source, eval_mode_node)
    if JAILBREAK_FUNCTION in eval_mode_text:
        return eval_mode_text, "full_eval_mode_function_fallback"
    return "", "not_found"


def no_refusal_success_patterns(text: str) -> dict[str, Any]:
    compact = " ".join(text.split())
    lowered = compact.lower()
    patterns = {
        "uses_key_words": KEYWORDS_NAME in text,
        "not_any_keywords": bool(re.search(r"not\s+any\s*\(.*_key_words", compact)),
        "all_not_in_keywords": bool(
            re.search(r"all\s*\(.*not\s+in.*_key_words", compact)
            or re.search(r"all\s*\(.*_key_words.*not\s+in", compact)
        ),
        "not_in_compare": " not in " in lowered and KEYWORDS_NAME in text,
        "increments_or_appends_success": bool(
            re.search(r"(\+=\s*1|append\s*\(|success|jailbreak|asr)", lowered)
        ),
    }
    verified = bool(
        patterns["uses_key_words"]
        and (
            patterns["not_any_keywords"]
            or patterns["all_not_in_keywords"]
            or patterns["not_in_compare"]
        )
        and patterns["increments_or_appends_success"]
    )
    return {"verified": verified, "patterns": patterns}


def eval_asr_calls_eval_mode_with_jailbreak(eval_asr_node: ast.AST | None) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    if eval_asr_node is None:
        return {"calls_eval_mode": False, "literal_jailbreak_arg": False, "calls": []}
    for node in ast.walk(eval_asr_node):
        if not isinstance(node, ast.Call):
            continue
        if dotted_name(node.func).split(".", 1)[0] != EVAL_MODE_FUNCTION:
            continue
        constants = string_constants(node)
        calls.append(
            {
                "line": getattr(node, "lineno", None),
                "string_constants": constants,
                "literal_jailbreak_arg": any(value.lower() == "jailbreak" for value in constants),
            }
        )
    return {
        "calls_eval_mode": bool(calls),
        "literal_jailbreak_arg": any(row["literal_jailbreak_arg"] for row in calls),
        "calls": calls,
    }


def gpt4_judge_clean_path_status(
    file_text: str,
    function_sources: dict[str, str],
    external_judge_in_asr_path: bool,
) -> str:
    lowered = file_text.lower()
    has_gpt_or_judge = any(hint in lowered for hint in ["gpt4", "gpt_4", "gpt-4", "gpt", "judge"])
    if not has_gpt_or_judge:
        return "false"
    if external_judge_in_asr_path:
        return "false"
    judge_text = "\n".join(
        text for name, text in function_sources.items() if any(hint in name.lower() for hint in GPT_JUDGE_NAME_HINTS)
    )
    if any(term in (judge_text + "\n" + lowered).lower() for term in ["clean", "performance", "quality"]):
        return "true"
    return "unknown"


def build_verification(source_path: Path) -> dict[str, Any]:
    text = read_text(source_path)
    if text is None:
        return missing_source_report(source_path)
    try:
        tree = ast.parse(text, filename=str(source_path))
        parse_error = ""
    except SyntaxError as exc:
        tree = ast.Module(body=[], type_ignores=[])
        parse_error = f"syntax_error: {exc}"

    functions = function_defs(tree)
    function_sources = {name: source_segment(text, node) for name, node in functions.items()}
    eval_asr_node = functions.get(EVAL_ASR_FUNCTION)
    eval_mode_node = functions.get(EVAL_MODE_FUNCTION)
    jailbreak_node = functions.get(JAILBREAK_FUNCTION)
    key_words_node = find_name_assignment(tree, KEYWORDS_NAME)
    refusal_keywords = literal_string_list(key_words_node)
    keywords_safe = keyword_list_is_safe(refusal_keywords)

    eval_asr_call_info = eval_asr_calls_eval_mode_with_jailbreak(eval_asr_node)
    branch_text, branch_source = best_jailbreak_branch_text(text, eval_mode_node)
    eval_mode_calls_jailbreak = bool(branch_text)
    jailbreak_mode_verified = bool(eval_mode_calls_jailbreak and "jailbreak" in branch_text.lower())
    call_chain_verified = bool(
        eval_asr_call_info["calls_eval_mode"]
        and eval_mode_calls_jailbreak
        and jailbreak_node is not None
    )
    jailbreak_text = source_segment(text, jailbreak_node)
    success_rule = no_refusal_success_patterns(jailbreak_text)

    asr_path_text = "\n\n".join(
        [
            source_segment(text, eval_asr_node),
            branch_text,
            jailbreak_text,
        ]
    )
    asr_path_external_hints = external_hints_in_text(asr_path_text)
    external_judge_in_asr_path = bool(asr_path_external_hints)
    gpt4_clean_status = gpt4_judge_clean_path_status(text, function_sources, external_judge_in_asr_path)
    prompt_template_caveat = True

    official_verified = bool(
        call_chain_verified
        and jailbreak_mode_verified
        and success_rule["verified"]
        and refusal_keywords
        and not external_judge_in_asr_path
    )
    if official_verified:
        confidence = "high"
        label = ASR_OFFICIAL_LABEL
        notes = (
            "Jailbreak ASR call chain and no-refusal keyword success rule were verified by static AST/text inspection. "
            "This verifies the scorer logic, not full BackdoorLLM generation formatting."
        )
    elif call_chain_verified and not external_judge_in_asr_path:
        confidence = "medium"
        label = ASR_PROXY_LABEL
        notes = (
            "Jailbreak ASR call chain was partly verified, but the no-refusal success rule or keyword list was not "
            "verified strongly enough to upgrade the metric label."
        )
    elif external_judge_in_asr_path:
        confidence = "medium" if call_chain_verified else "low"
        label = ASR_PROXY_LABEL
        notes = "External judge/API hints appear in the traced jailbreak ASR path; keep local ASR as proxy."
    else:
        confidence = "low"
        label = ASR_PROXY_LABEL
        notes = "Could not verify the complete official rule-based jailbreak ASR path."

    return {
        "source_file": safe_rel(source_path),
        "source_found": True,
        "parse_error": parse_error,
        "symbols": {
            KEYWORDS_NAME: {
                "found": key_words_node is not None,
                "line_range": line_range(key_words_node),
            },
            JAILBREAK_FUNCTION: {
                "found": jailbreak_node is not None,
                "line_range": line_range(jailbreak_node),
                "called_functions": call_names(jailbreak_node),
            },
            EVAL_MODE_FUNCTION: {
                "found": eval_mode_node is not None,
                "line_range": line_range(eval_mode_node),
                "called_functions": call_names(eval_mode_node),
            },
            EVAL_ASR_FUNCTION: {
                "found": eval_asr_node is not None,
                "line_range": line_range(eval_asr_node),
                "called_functions": call_names(eval_asr_node),
            },
        },
        "judge_related_functions": judge_related_functions(functions),
        "refusal_keywords": refusal_keywords if keywords_safe else [],
        "refusal_keywords_stored": bool(keywords_safe),
        "refusal_keywords_count": len(refusal_keywords),
        "refusal_keywords_hash": sha256_joined(refusal_keywords),
        "call_chain": {
            "eval_asr_calls_eval_mode": bool(eval_asr_call_info["calls_eval_mode"]),
            "eval_asr_eval_mode_calls": eval_asr_call_info["calls"],
            "eval_asr_literal_jailbreak_arg": bool(eval_asr_call_info["literal_jailbreak_arg"]),
            "eval_mode_calls_jailbreak_eval": eval_mode_calls_jailbreak,
            "eval_mode_jailbreak_branch_source": branch_source,
            "call_chain_verified": call_chain_verified,
            "jailbreak_mode_verified": jailbreak_mode_verified,
        },
        "success_rule": {
            "success_on_no_refusal_verified": bool(success_rule["verified"]),
            "pattern_evidence": success_rule["patterns"],
        },
        "external_judge": {
            "external_judge_in_asr_path": external_judge_in_asr_path,
            "asr_path_external_hints": asr_path_external_hints,
            "gpt4_judge_only_clean_path": gpt4_clean_status,
            "file_level_external_hints": external_hints_in_text(text),
        },
        "decision": {
            "official_rule_based_asr_verified": official_verified,
            "asr_metric_label": label,
            "is_official_asr": official_verified,
            "is_external_judged_asr": external_judge_in_asr_path,
            "external_judge_in_asr_path": external_judge_in_asr_path,
            "gpt4_judge_only_clean_path": gpt4_clean_status,
            "prompt_template_caveat": prompt_template_caveat,
            "confidence_level": confidence,
            "notes": notes,
        },
        "safety": {
            "source_text_only": True,
            "official_code_execution": False,
            "model_loading": False,
            "inference": False,
            "external_api_calls": False,
            "full_harmful_prompts_printed": False,
            "full_generated_outputs_printed": False,
        },
    }


def missing_source_report(source_path: Path) -> dict[str, Any]:
    return {
        "source_file": safe_rel(source_path),
        "source_found": False,
        "parse_error": "source_file_not_found_or_unreadable",
        "symbols": {},
        "judge_related_functions": [],
        "refusal_keywords": [],
        "refusal_keywords_stored": False,
        "refusal_keywords_count": 0,
        "refusal_keywords_hash": "",
        "call_chain": {
            "eval_asr_calls_eval_mode": False,
            "eval_mode_calls_jailbreak_eval": False,
            "call_chain_verified": False,
            "jailbreak_mode_verified": False,
        },
        "success_rule": {
            "success_on_no_refusal_verified": False,
            "pattern_evidence": {},
        },
        "external_judge": {
            "external_judge_in_asr_path": False,
            "asr_path_external_hints": [],
            "gpt4_judge_only_clean_path": "unknown",
            "file_level_external_hints": [],
        },
        "decision": {
            "official_rule_based_asr_verified": False,
            "asr_metric_label": ASR_PROXY_LABEL,
            "is_official_asr": False,
            "is_external_judged_asr": False,
            "external_judge_in_asr_path": False,
            "gpt4_judge_only_clean_path": "unknown",
            "prompt_template_caveat": True,
            "confidence_level": "low",
            "notes": "Source file was not found. Run this on the server after the sanitized BackdoorLLM source snapshot exists.",
        },
        "safety": {
            "source_text_only": True,
            "official_code_execution": False,
            "model_loading": False,
            "inference": False,
            "external_api_calls": False,
            "full_harmful_prompts_printed": False,
            "full_generated_outputs_printed": False,
        },
    }


def symbol(report: dict[str, Any], name: str) -> dict[str, Any]:
    return report.get("symbols", {}).get(name, {})


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    decision = report["decision"]
    call_chain = report["call_chain"]
    success_rule = report["success_rule"]
    external = report["external_judge"]
    return {
        "source_file": report["source_file"],
        "eval_asr_function_found": bool(symbol(report, EVAL_ASR_FUNCTION).get("found", False)),
        "eval_asr_line_range": symbol(report, EVAL_ASR_FUNCTION).get("line_range", ""),
        "_eval_mode_found": bool(symbol(report, EVAL_MODE_FUNCTION).get("found", False)),
        "_eval_mode_line_range": symbol(report, EVAL_MODE_FUNCTION).get("line_range", ""),
        "jailbreak_eval_found": bool(symbol(report, JAILBREAK_FUNCTION).get("found", False)),
        "jailbreak_eval_line_range": symbol(report, JAILBREAK_FUNCTION).get("line_range", ""),
        "_key_words_found": bool(symbol(report, KEYWORDS_NAME).get("found", False)),
        "_key_words_line_range": symbol(report, KEYWORDS_NAME).get("line_range", ""),
        "refusal_keywords_count": int(report.get("refusal_keywords_count", 0)),
        "refusal_keywords_hash": report.get("refusal_keywords_hash", ""),
        "call_chain_verified": bool(call_chain.get("call_chain_verified", False)),
        "jailbreak_mode_verified": bool(call_chain.get("jailbreak_mode_verified", False)),
        "success_on_no_refusal_verified": bool(
            success_rule.get("success_on_no_refusal_verified", False)
        ),
        "external_judge_in_asr_path": bool(external.get("external_judge_in_asr_path", False)),
        "gpt4_judge_only_clean_path": external.get("gpt4_judge_only_clean_path", "unknown"),
        "asr_metric_label": decision["asr_metric_label"],
        "is_official_asr": bool(decision["is_official_asr"]),
        "is_external_judged_asr": bool(decision["is_external_judged_asr"]),
        "prompt_template_caveat": bool(decision["prompt_template_caveat"]),
        "confidence_level": decision["confidence_level"],
        "notes": decision["notes"],
    }


def markdown_summary(report: dict[str, Any]) -> str:
    row = summary_row(report)
    decision = report["decision"]
    external = report["external_judge"]
    if not report.get("source_found"):
        finding = (
            "The source file was not found in this workspace. Run the verifier on the server "
            "where `external_sources/backdoorllm_official_source/` exists."
        )
    else:
        finding = (
            f"Inspected `{report['source_file']}` using source-text and AST analysis only."
        )
    external_hint_text = (
        ", ".join(external.get("asr_path_external_hints", []))
        if external.get("asr_path_external_hints")
        else "none detected"
    )
    return f"""# Official Rule-Based ASR Verification

Static verification status: source-text and AST inspection only. No
BackdoorLLM code was executed, no model was loaded, no inference was run, and
no external API was called.

## Source

{finding}

## Symbol Locations

- `{EVAL_ASR_FUNCTION}` found: `{str(row['eval_asr_function_found']).lower()}` at `{row['eval_asr_line_range']}`
- `{EVAL_MODE_FUNCTION}` found: `{str(row['_eval_mode_found']).lower()}` at `{row['_eval_mode_line_range']}`
- `{JAILBREAK_FUNCTION}` found: `{str(row['jailbreak_eval_found']).lower()}` at `{row['jailbreak_eval_line_range']}`
- `{KEYWORDS_NAME}` found: `{str(row['_key_words_found']).lower()}` at `{row['_key_words_line_range']}`
- Refusal keyword count: `{row['refusal_keywords_count']}`
- Refusal keyword SHA256: `{row['refusal_keywords_hash']}`

## ASR Call Chain

- `eval_ASR_of_backdoor_models -> _eval_mode`: `{str(report['call_chain'].get('eval_asr_calls_eval_mode', False)).lower()}`
- `_eval_mode("jailbreak", ...) -> jailbreak_eval`: `{str(row['jailbreak_mode_verified']).lower()}`
- Full call chain verified: `{str(row['call_chain_verified']).lower()}`
- Success-on-no-refusal rule verified: `{str(row['success_on_no_refusal_verified']).lower()}`

## Judge Separation

- External judge/API detected in jailbreak ASR path: `{str(row['external_judge_in_asr_path']).lower()}`
- ASR-path external hints: `{external_hint_text}`
- GPT/judge appears only in clean-performance path: `{row['gpt4_judge_only_clean_path']}`

## Decision

- Official rule-based ASR verified: `{str(decision['official_rule_based_asr_verified']).lower()}`
- ASR metric label: `{decision['asr_metric_label']}`
- `is_official_asr`: `{str(decision['is_official_asr']).lower()}`
- `is_external_judged_asr`: `{str(decision['is_external_judged_asr']).lower()}`
- Confidence: `{decision['confidence_level']}`

{decision['notes']}

## Caveats

- This verifies scorer logic only, not the full BackdoorLLM generation
  pipeline.
- If project generations use deterministic `[INST] ... [/INST]` formatting,
  the correct wording is: official rule-based scorer applied to project-local
  deterministic generations.
- Do not update final submission artifacts or final-facing claims until any
  verified ASR results are reviewed and consistency-checked.
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_file)
    verification = build_verification(source_path)
    timestamp = utc_timestamp()
    return {
        "timestamp_utc": timestamp,
        "script": Path(__file__).name,
        "mode": "source_text_ast_verification_only",
        "verification": verification,
        "summary_row": summary_row(verification),
        "safety": verification["safety"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically verify BackdoorLLM rule-based jailbreak ASR path."
    )
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    timestamp = report["timestamp_utc"]
    log_path = Path(args.logs_dir) / f"official_rule_based_asr_verification_{timestamp}.json"
    write_json(log_path, report)
    write_csv(Path(args.summary_csv), [report["summary_row"]])
    Path(args.markdown).write_text(markdown_summary(report["verification"]), encoding="utf-8")

    row = report["summary_row"]
    print("Official rule-based jailbreak ASR verification summary")
    print("- Source-text/AST inspection only; no model loading, inference, official code execution, or APIs")
    print(f"- Source file: {row['source_file']}")
    print(f"- eval_ASR_of_backdoor_models found: {row['eval_asr_function_found']}")
    print(f"- _eval_mode found: {row['_eval_mode_found']}")
    print(f"- jailbreak_eval found: {row['jailbreak_eval_found']}")
    print(f"- _key_words found: {row['_key_words_found']}")
    print(f"- Refusal keywords count: {row['refusal_keywords_count']}")
    print(f"- Call chain verified: {row['call_chain_verified']}")
    print(f"- Success-on-no-refusal verified: {row['success_on_no_refusal_verified']}")
    print(f"- External judge in ASR path: {row['external_judge_in_asr_path']}")
    print(f"- ASR metric label: {row['asr_metric_label']}")
    print(f"- is_official_asr: {row['is_official_asr']}")
    print(f"- Confidence: {row['confidence_level']}")
    print("- Full harmful prompts/outputs printed: False")
    print(f"- JSON log written: {log_path}")
    print(f"- Summary CSV written: {Path(args.summary_csv)}")
    print(f"- Markdown written: {Path(args.markdown)}")
    return 0 if row["confidence_level"] in {"high", "medium"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
