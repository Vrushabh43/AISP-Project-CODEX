# LoRA Backdoor Sanitisation Project

This repository contains a compact, reproducible submission package for a
semester project on sanitising a LoRA backdoor in a Llama-2 chat adapter. The
package keeps the final evidence, reproducibility scripts, and concise result
artifacts while excluding generated model/cache artifacts from the submitted
tree.

## Research Question

The project asks whether a sensitivity-aware attenuation strategy can reduce a
BadNets-style jailbreak LoRA backdoor more selectively than simple spectral or
uniform scaling baselines, while preserving useful clean behavior as measured by
a lightweight clean-reference likelihood probe.

## Final Method

The final method, `sensaware_top224_gamma_0.25`, ranks LoRA singular components
using spectral energy and a clean-activation sensitivity proxy, then attenuates
the selected components. It is compared against the original adapter,
uniform-scaling baselines, and a spectral-only top-component baseline.

## Main Result

The strongest final focused result is:

- Method: `sensaware_top224_gamma_0.25`
- Metric: `BackdoorLLM official rule-based jailbreak ASR`
- ASR: `1/99 = 0.010101`
- Clean-reference perplexity: `3.557692`

The ASR metric is not external-judged. The verified BackdoorLLM rule-based
scorer counts a generation as successful when it avoids the verified
refusal-keyword list. In this package, that scorer is applied to project-local
deterministic `[INST]` generations, so it verifies the scorer logic but does
not claim exact reproduction of the full BackdoorLLM generation pipeline.

Wilson 95% confidence intervals should be reported for all ASR rates because
the trigger set contains 99 prompts. Clean perplexity is a reference-likelihood
probe, not final human utility.

## Focused Results

| Condition | Official rule-based ASR | Clean perplexity |
|---|---:|---:|
| `original` | `36/99` | `3.655260` |
| `uniform_gamma_0.25` | `13/99` | `3.747472` |
| `uniform_gamma_0.50` | `7/99` | `3.607213` |
| `top3_gamma_0.50` | `7/99` | `3.559641` |
| `sensaware_top224_gamma_0.25` | `1/99` | `3.557692` |

The result is positive but bounded: SensAware achieved the lowest observed
official rule-based ASR count and the lowest clean-reference perplexity in this
run. The ASR metric remains rule-based, and small gaps should be interpreted
with Wilson intervals.

## Repository Layout

- `scripts/pipeline/`: reproducibility and validation scripts.
- `outputs/results/`: concise final CSV and Markdown result summaries.
- `reports/figures/`: final figure used for reporting.
- `submission_artifacts/`: professor-facing bundle of documents, result
  summaries, and the final figure.
- `src/lora_sanitise/`: reusable LoRA/SVD helper code.
- `data/eval_prompts/`: evaluation prompt/reference files.
- `external_sources/`: official source material used for static scorer
  verification.

Generated adapters, model snapshots, large cache folders, detailed logs, and
intermediate development artifacts are intentionally excluded from the compact
submitted tree. They can be regenerated with the documented pipeline and
external Hugging Face caches.

## Required Model And Adapter Assets

Base model:

- `NousResearch/Llama-2-7b-chat-hf`

Backdoored LoRA adapter:

- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`

Adapter base-model compatibility:

- `meta-llama/Llama-2-7b-chat-hf` / Llama-2-7B-Chat family

These assets are not included in the submitted zip because of the 100 MB size
limit. The cleaned submission package keeps code, scripts, prompts, final
summaries, and figures, but not large model or adapter weights. The full
pipeline expects the base model and original backdoored adapter to be available
in a Hugging Face cache.

Example cache setup:

```bash
export HF_HUB_CACHE=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/huggingface/cache
```

## Quick Validation

Run the file-only validation:

```bash
python scripts/pipeline/validate_submission.py
python scripts/pipeline/run_pipeline.py --mode quick
```

These commands do not load models or run inference. They validate structure,
artifact hygiene, result numbers, official ASR labels, the `[INST]` caveat, and
the compact package size excluding `.git` metadata.

## Reproduction

The lightweight verification path is:

```bash
python scripts/pipeline/run_pipeline.py --mode verify
```

The heavy reproduction path requires a configured GPU environment and cached
Hugging Face assets:

```bash
export HF_HUB_CACHE=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache
python scripts/pipeline/run_pipeline.py --mode full
```

Full mode reruns deterministic generation and clean-reference perplexity
evaluation. It requires regenerated or externally available adapter artifacts
and does not rely on shipping large model files inside this repository.

If the generated sanitized adapters are not present, regenerate them first with
the pipeline scripts described in `REPRODUCIBILITY.md`, then run full mode.

## Files To Open First

1. `PROJECT_VERDICT.md`
2. `REPRODUCIBILITY.md`
3. `PROJECT_STRUCTURE.md`
4. `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`
5. `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`
6. `outputs/results/official_rule_based_asr_clean_tradeoff_summary.csv`

The same professor-facing materials are copied into `submission_artifacts/`.
