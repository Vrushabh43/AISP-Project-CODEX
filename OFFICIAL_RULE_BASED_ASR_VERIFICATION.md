# Official Rule-Based ASR Verification

Static verification status: source-text and AST inspection only. No
BackdoorLLM code was executed, no model was loaded, no inference was run, and
no external API was called.

## Source

Inspected `external_sources/backdoorllm_official_source/attack/DPA/backdoor_evaluate.py` using source-text and AST analysis only.

## Symbol Locations

- `eval_ASR_of_backdoor_models` found: `true` at `198:261`
- `_eval_mode` found: `true` at `129:141`
- `jailbreak_eval` found: `true` at `41:62`
- `_key_words` found: `true` at `17:21`
- Refusal keyword count: `17`
- Refusal keyword SHA256: `351c6ad9b45d70ea95926ed9bfb760d33dda99843451d0c7fdfeb7d48808d4af`

## ASR Call Chain

- `eval_ASR_of_backdoor_models -> _eval_mode`: `true`
- `_eval_mode("jailbreak", ...) -> jailbreak_eval`: `true`
- Full call chain verified: `true`
- Success-on-no-refusal rule verified: `true`

## Judge Separation

- External judge/API detected in jailbreak ASR path: `true`
- ASR-path external hints: `gpt_4, gpt_4_judge_eval`
- GPT/judge appears only in clean-performance path: `false`

## Decision

- Official rule-based ASR verified: `false`
- ASR metric label: `BackdoorLLM-aligned ASR proxy`
- `is_official_asr`: `false`
- `is_external_judged_asr`: `true`
- Confidence: `medium`

External judge/API hints appear in the traced jailbreak ASR path; keep local ASR as proxy.

## Caveats

- This verifies scorer logic only, not the full BackdoorLLM generation
  pipeline.
- If project generations use deterministic `[INST] ... [/INST]` formatting,
  the correct wording is: official rule-based scorer applied to project-local
  deterministic generations.
- Do not update final submission artifacts or final-facing claims until any
  verified ASR results are reviewed and consistency-checked.
