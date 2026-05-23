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

All metrics below are bounded heuristic metrics, not final judged ASR or final
judged clean utility.

Key current result:

- Original adapter: trigger success `0.6061`, clean utility `0.9667`.
- Best spectral-only baseline, `top3_gamma_0.50`: trigger success `0.0606`,
  clean utility `0.9333`.
- Best SensAware variant, `sensaware_top224_gamma_0.25`: trigger success
  `0.0101`, clean utility `0.9667`.
- Strongest uniform baseline, `uniform_gamma_0.25`: trigger success `0.0000`,
  clean utility `0.9667`.

Interpretation:

- Expanded SensAware beats spectral-only under the bounded heuristic evaluation.
- Expanded SensAware nearly matches but does not beat the strongest uniform
  scaling baseline.
- The result is mixed, not a full win for the proposed method.
- Uniform scaling may reduce trigger behavior by globally weakening the adapter;
  this is a caveat requiring stronger clean-utility evaluation, not a proven
  conclusion.
- The first small SensAware variants were too conservative and mostly failed.

## Key Final Outputs

Report-facing tables:

- `outputs/report_ready_main_results.md`
- `outputs/report_ready_key_findings.md`
- `outputs/report_ready_case_diagnostics_summary.md`
- `outputs/consolidated_tradeoff_results.csv`
- `outputs/expanded_sensaware_asr_utility_tradeoff_summary.csv`

Report-facing figures:

- `reports/figures/report_ready/report_tradeoff_scatter_key_methods.png`
- `reports/figures/report_ready/report_trigger_rate_key_methods.png`
- `reports/figures/report_ready/report_trigger_reduction_key_methods.png`
- `reports/figures/report_ready/report_clean_utility_key_methods.png`

Submission bundle:

- `final_submission_artifacts/`

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

The official BadNets trigger format was verified from BackdoorLLM assets, but
the evaluation in this repository is still bounded and heuristic. Do not present
the current numbers as final judged ASR or final judged clean utility. Do not
include full harmful prompts or full generated outputs in the final report.

## Project Memory

Read these files before continuing project work:

- `AGENT.md`
- `status.md`
