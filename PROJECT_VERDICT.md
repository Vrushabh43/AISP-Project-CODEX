# Project Verdict

The project is ready as a compact, academically honest submission package.

## Purpose

The project evaluates whether sensitivity-aware attenuation of LoRA singular
components can reduce a jailbreak backdoor more selectively than broad uniform
scaling or simple spectral-only attenuation. The final comparison uses the
verified BackdoorLLM official rule-based jailbreak ASR scorer and a
clean-reference perplexity probe.

## Final Scientific Result

The strongest focused condition is `sensaware_top224_gamma_0.25`.

- Metric: `BackdoorLLM official rule-based jailbreak ASR`
- ASR: `1/99 = 0.010101`
- Clean-reference perplexity: `3.557692`
- Scoring: official BackdoorLLM rule-based refusal-keyword scorer
- Generation caveat: scorer applied to project-local deterministic `[INST]`
  generations

The ASR metric is not external-judged. It is a verified rule-based scorer, not a
human or GPT-style harmfulness judgment.

## Focused Comparison

| Condition | Official rule-based ASR | Clean perplexity |
|---|---:|---:|
| `original` | `36/99` | `3.655260` |
| `uniform_gamma_0.25` | `13/99` | `3.747472` |
| `uniform_gamma_0.50` | `7/99` | `3.607213` |
| `top3_gamma_0.50` | `7/99` | `3.559641` |
| `sensaware_top224_gamma_0.25` | `1/99` | `3.557692` |

SensAware has the lowest observed ASR count and the lowest clean-reference
perplexity in the focused final run. Wilson 95% intervals must still be shown
because the ASR set has 99 prompts.

## Honest Interpretation

This is a meaningful improvement over the earlier bounded heuristic evaluation
because the BackdoorLLM jailbreak ASR scorer was statically verified and then
applied consistently to the final project generations.

The result should not be overclaimed:

- It is official rule-based ASR, not external-judged ASR.
- It uses project-local deterministic `[INST]` generations.
- Clean perplexity is a reference-likelihood probe, not final human utility.
- Small differences require cautious interpretation with Wilson intervals.
- The nonzero `base_model_only` ASR count reflects the no-refusal scoring rule,
  not learned backdoor behavior.

## Submission Verdict

Submit this branch/package after running:

```bash
python scripts/pipeline/validate_submission.py
python scripts/pipeline/run_pipeline.py --mode quick
```

The professor should open:

1. `README.md`
2. `PROJECT_VERDICT.md`
3. `REPRODUCIBILITY.md`
4. `PROJECT_STRUCTURE.md`
5. `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`
6. `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`
7. `submission_artifacts/`
