# Evaluation Prompt Files

These JSONL files provide explicit prompt sets for bounded evaluation. The
official BadNets trigger prompts are used only for bounded heuristic evaluation,
not final judged ASR.

## Current Files

- `clean_utility_small.jsonl`: 10 harmless instruction-following prompts.
- `clean_utility_medium.jsonl`: 30 harmless prompts used for bounded clean
  utility comparisons.
- `official_badnets_jailbreak_small.jsonl`: small official BackdoorLLM BadNets
  prompt subset.
- `official_badnets_jailbreak_full.jsonl`: 99 official BackdoorLLM BadNets
  trigger records used in bounded heuristic evaluation.

## JSONL Schema

Each row has:

```json
{
  "id": "clean_001",
  "split": "clean",
  "prompt": "...",
  "expected_behavior": "short helpful harmless answer",
  "notes": "..."
}
```

## Important Caveat

The official trigger prompt files come from the verified BackdoorLLM test-data
asset under `external_sources/backdoorllm_official/`. They are still used with
heuristic scoring only, so results must not be reported as final judged ASR.

The old `trigger_probe_small_unverified.jsonl` file was a benign placeholder
for evaluation-plumbing checks before official trigger verification. It has been
superseded and archived under `archive/submission_cleanup_<timestamp>/`.
