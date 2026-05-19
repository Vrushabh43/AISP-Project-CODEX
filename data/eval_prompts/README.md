# Evaluation Prompt Files

These JSONL files provide small, explicit prompt sets for bounded evaluation
plumbing. They are intentionally limited and are not the final ASR or clean
utility benchmark.

## Files

- `clean_utility_small.jsonl`: 10 harmless instruction-following prompts.
- `trigger_probe_small_unverified.jsonl`: 5 benign placeholder probes.

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

The trigger-probe prompts are not official BackdoorLLM triggers. They are
unverified placeholders for checking evaluation plumbing only. Results from
`trigger_probe_small_unverified.jsonl` must not be reported as final ASR.

