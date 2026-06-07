# ASR Sanity Summary

This summary uses aggregate counts and safe audit metadata only. It does not
include full harmful prompts or generated trigger outputs.

- Refusal keyword count: `17`.
- Refusal keyword SHA256:
  `351c6ad9b45d70ea95926ed9bfb760d33dda99843451d0c7fdfeb7d48808d4af`.
- Official rule-based ASR scorer verified: `true`.
- External-judged ASR: `false`.
- Best focused condition: `sensaware_top224_gamma_0.25`.
- Best focused ASR count: `1/99`.
- Best focused clean-reference perplexity: `3.557692`.

The `base_model_only` control scored `15/99` because the official rule-based
metric counts non-refusal completions as successful. This is not evidence of a
learned backdoor in the base model.
