# Post-Hoc Sanitisation of Backdoored LoRA Adapters

This repository contains the implementation and bounded empirical study for:

**Post-Hoc Sanitisation of Backdoored LoRA Adapters via Sensitivity-Aware
Singular-Component Attenuation**

The project investigates whether suspicious singular components of a
backdoored LoRA adapter can be attenuated after training to reduce trigger
success while preserving clean utility.

## Research Question

Can suspicious singular components of an already-trained backdoored LoRA update
matrix be attenuated post-hoc to reduce Attack Success Rate while preserving
clean task utility?

Operationally, this repository tests whether combining spectral concentration
with a clean-prompt sensitivity proxy improves over simpler post-hoc baselines:
the original adapter, uniform adapter scaling, and spectral-only top-sigma
attenuation.

## Threat Model

- Trusted base model.
- Untrusted third-party LoRA adapter.
- White-box access to adapter weights.
- No retraining budget.
- No original training data.
- Small clean calibration prompt set.
- Non-adaptive attacker.

The main evaluated adapter is:

`BackdoorLLM/Jailbreak_Llama2-7B_BadNets`

The base model used in the bounded evaluation runs is:

`NousResearch/Llama-2-7b-chat-hf`

## Implemented Method

The implemented SensAware method is adapter-only and post-hoc:

1. Load LoRA `A` and `B` tensors from the adapter.
2. Compute compact singular values and singular directions for each LoRA update
   without loading or modifying the base weights.
3. Estimate clean-prompt component sensitivity using an activation-projection
   proxy:
   `s_i^2 * mean((v_i^T x)^2)`.
4. Score candidate singular components with:
   `spectral_energy_share * (1 - clean_sensitivity_norm_global)`.
5. Select components by global top-N ranking.
6. Attenuate selected singular values.
7. Refactor the edited update back into PEFT-compatible LoRA `A/B` tensors.

Important implementation note: this is not a KL-divergence sensitivity sweep,
not a lambda sweep, and not a per-module-only top-K method. Those were planning
ideas, not the final implemented method.

## Baselines

The bounded experiments compare:

- `original`: unmodified backdoored adapter.
- `uniform_gamma_0.25` and `uniform_gamma_0.50`: global LoRA update scaling.
- `top1_gamma_0.50` and `top3_gamma_0.50`: spectral-only attenuation.
- SensAware variants, especially `sensaware_top224_gamma_0.25`.

## Final Honest Result Summary

The final-facing result uses the verified BackdoorLLM official rule-based
jailbreak ASR scorer. This scorer counts a generation as successful when the
output avoids the verified refusal-keyword list. It is **not** an
external-judged harmfulness metric.

The scorer is applied to project-local deterministic `[INST]` generations, so
the result verifies the official scorer logic but does not claim exact
reproduction of the full BackdoorLLM generation pipeline.

Focused official rule-based ASR results over 99 official BadNets trigger
records:

| Condition | ASR count | ASR rate | Wilson 95% CI | Clean-reference perplexity |
|---|---:|---:|---:|---:|
| `base_model_only` | 15/99 | 0.151515 | [0.094022, 0.235043] | 3.980931 |
| `original` | 36/99 | 0.363636 | [0.275617, 0.461843] | 3.655260 |
| `uniform_gamma_0.25` | 13/99 | 0.131313 | [0.078372, 0.211799] | 3.747472 |
| `uniform_gamma_0.50` | 7/99 | 0.070707 | [0.034670, 0.138816] | 3.607213 |
| `top3_gamma_0.50` | 7/99 | 0.070707 | [0.034670, 0.138816] | 3.559641 |
| `sensaware_top224_gamma_0.25` | 1/99 | 0.010101 | [0.001785, 0.055017] | 3.557692 |

Interpretation:

- `sensaware_top224_gamma_0.25` has the lowest observed official rule-based
  ASR among the focused conditions.
- It also has the lowest clean-reference perplexity in this run, narrowly ahead
  of `top3_gamma_0.50`.
- Clean-reference perplexity is a reference-likelihood probe, not final human
  clean utility.
- Wilson intervals over 99 prompts should be reported; the 1/99 versus 7/99
  gap is meaningful-looking but should still be described cautiously.
- The `base_model_only` value of 15/99 is a control artifact of the
  no-refusal-keyword rule, not evidence of a learned backdoor.
- Earlier bounded heuristic results are preserved as historical evidence, but
  the final-facing ASR result is the official rule-based ASR audit above.
- The first small SensAware variants were too conservative and mostly failed;
  that negative result is preserved.

## Key Final Outputs

Report-facing tables:

- `outputs/final_results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/final_results/official_rule_based_asr_eval_summary.csv`
- `outputs/final_results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/final_results/official_rule_based_asr_verification_summary.csv`
- `outputs/final_results/clean_utility_perplexity_summary.csv`
- `outputs/final_results/clean_behaviour_similarity_summary.csv`
- `outputs/final_results/report_ready_main_results.md`
- `outputs/final_results/report_ready_key_findings.md`
- `outputs/final_results/report_ready_case_diagnostics_summary.md`

Report-facing figures:

- `reports/figures/report_ready/official_rule_based_asr_clean_tradeoff.png`
- `reports/figures/report_ready/report_tradeoff_scatter_key_methods.png`
- `reports/figures/report_ready/report_trigger_rate_key_methods.png`
- `reports/figures/report_ready/report_trigger_reduction_key_methods.png`
- `reports/figures/report_ready/report_clean_utility_key_methods.png`

Submission bundle:

- `final_submission_artifacts/`

## One-Command Final Check

For grading or submission review, run:

```bash
python scripts/final_pipeline/00_run_final_submission.py --mode quick
```

This is the recommended professor-facing command. It is file-only, does not load
models, does not run inference, and verifies that the final documentation,
figures, result tables, artifact bundle, and core result numbers are present and
consistent.

Optional modes:

```bash
python scripts/final_pipeline/00_run_final_submission.py --mode verify
python scripts/final_pipeline/00_run_final_submission.py --mode full
```

- `verify` regenerates lightweight final analysis/report artifacts and then
  runs the consistency check. It does not load models.
- `full` runs the canonical heavy final reproduction pipeline and requires a
  configured GPU machine plus a Hugging Face cache containing the base model and
  adapter.

The retained professor-facing scripts live in `scripts/final_pipeline/`.
Development, smoke, pilot, and diagnostic scripts are archived under
`archive/final_minimal_cleanup_<timestamp>/dev_smoke_scripts/`.

## Reproduction Notes

For the full script order, see:

- `RUN_ORDER.md`

For the file and artifact layout, see:

- `SUBMISSION_STRUCTURE.md`

For implementation caveats and method-vs-plan differences, see:

- `IMPLEMENTATION_NOTES.md`

For the final submission verdict, see:

- `FINAL_SUBMISSION_VERDICT.md`

## Environment Notes

The successful GPU evaluation runs used a Linux server with:

- Python virtual environment active.
- `PYTHONNOUSERSITE=1`.
- Hugging Face cache:
  `/home/43e3/hf-cache-aisp`.
- 4-bit base-model loading.
- Isolated subprocess execution per adapter.

Typical command prefix:

```bash
cd ~/solr-home/AISP-Project-CODEX
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
source .venv/bin/activate
```

## Safety and Reporting Warning

The final ASR metric is the verified BackdoorLLM official rule-based jailbreak
ASR scorer applied to project-local deterministic `[INST]` generations. It is
not external-judged ASR or a human harmfulness judgment. Clean-reference
perplexity is not final human utility. Do not include full harmful prompts or
full generated outputs in the final report.

## Project Log

For future maintenance, use:

- `status.md`
- `RUN_ORDER.md`
- `SUBMISSION_STRUCTURE.md`

Historical project memory was archived during final cleanup.
