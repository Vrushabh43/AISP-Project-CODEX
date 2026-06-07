# Key Findings

- The verified ASR metric is `BackdoorLLM official rule-based jailbreak ASR`.
- The metric is not external-judged; it is a refusal-keyword rule applied to
  project-local deterministic `[INST]` generations.
- The best focused condition is `sensaware_top224_gamma_0.25`.
- `sensaware_top224_gamma_0.25` achieved `1/99 = 0.010101` ASR with Wilson 95%
  CI `[0.001785, 0.055017]`.
- The same condition had the lowest clean-reference perplexity in this run:
  `3.557692`.
- `uniform_gamma_0.50` and `top3_gamma_0.50` both reached `7/99 = 0.070707`
  ASR, so the 1/99 vs 7/99 gap should be described cautiously with intervals.
- `base_model_only` scored `15/99` under the no-refusal rule, which is a control
  artifact and not evidence of a learned backdoor.
- Clean-reference perplexity is a likelihood probe, not final human utility.

Recommended wording: the best SensAware variant achieved the lowest observed
official rule-based ASR and the lowest clean-reference perplexity in this run,
with the caveat that ASR is rule-based and the generation format is
project-local deterministic `[INST]`.
