# Reproducibility

This page defines the runnable pipeline for the compact submission package.

## Environment

Install Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

For heavy reproduction on a GPU server, set a Hugging Face cache that contains
the base model and original adapter:

```bash
export HF_HUB_CACHE=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache
export PYTHONNOUSERSITE=1
```

## Required Model And Adapter Assets

Base model:

- `NousResearch/Llama-2-7b-chat-hf`

Backdoored LoRA adapter:

- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`

Adapter base-model compatibility:

- `meta-llama/Llama-2-7b-chat-hf` / Llama-2-7B-Chat family

These weights are not included in the submitted zip because of the 100 MB size
limit. The full pipeline expects the base model and adapter to be available in a
Hugging Face cache. The compact package keeps code, prompts, final summaries,
figures, and validation evidence, but not large model or adapter weights.

Expected cache variables:

```bash
export HF_HUB_CACHE=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/huggingface/cache
```

## Commands

File-only validation:

```bash
python scripts/pipeline/validate_submission.py
python scripts/pipeline/run_pipeline.py --mode quick
```

Lightweight regeneration from existing CSV inputs:

```bash
python scripts/pipeline/run_pipeline.py --mode verify
```

Heavy reproduction:

```bash
python scripts/pipeline/run_pipeline.py --mode full
```

Quick and verify modes do not load models or run inference. Full mode reruns
deterministic generation and clean-reference perplexity evaluation.

If `outputs/sanitised_adapters/` is not present, regenerate the required
adapter artifacts before full mode:

```bash
python scripts/pipeline/prepare_evaluation_data.py
python scripts/pipeline/extract_spectral_statistics.py
python scripts/pipeline/generate_spectral_baselines.py
python scripts/pipeline/generate_uniform_baselines.py
python scripts/pipeline/compute_sensitivity_scores.py
python scripts/pipeline/generate_sensitivity_aware_adapter.py
python scripts/pipeline/run_pipeline.py --mode full
```

The generated adapter directories and `.safetensors` files are intentionally not
shipped in this compact package. They are reproducible artifacts.

## Pipeline Scripts

- `scripts/pipeline/run_pipeline.py`: orchestrates quick, verify, and full modes.
- `scripts/pipeline/validate_submission.py`: file-only package validation.
- `scripts/pipeline/verify_rule_based_asr.py`: static verification of the BackdoorLLM rule-based ASR scorer.
- `scripts/pipeline/prepare_evaluation_data.py`: prepares official ASR and clean-reference evaluation files.
- `scripts/pipeline/evaluate_rule_based_asr.py`: deterministic generation plus official rule-based ASR scoring.
- `scripts/pipeline/evaluate_clean_perplexity.py`: reference-output NLL/perplexity evaluation.
- `scripts/pipeline/build_tradeoff_summary.py`: combines ASR, Wilson intervals, perplexity, and similarity evidence.
- `scripts/pipeline/audit_rule_based_asr.py`: sanity checks for the final ASR result.
- `scripts/pipeline/extract_spectral_statistics.py`: LoRA spectral statistics.
- `scripts/pipeline/generate_spectral_baselines.py`: spectral-only adapter generation.
- `scripts/pipeline/generate_uniform_baselines.py`: uniform scaling baseline generation.
- `scripts/pipeline/compute_sensitivity_scores.py`: sensitivity-aware scoring.
- `scripts/pipeline/generate_sensitivity_aware_adapter.py`: SensAware adapter generation.

## Expected Outputs

Primary result files:

- `outputs/results/official_rule_based_asr_eval_summary.csv`
- `outputs/results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/results/official_rule_based_asr_verification_summary.csv`
- `outputs/results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/results/clean_utility_perplexity_summary.csv`
- `outputs/results/validation_summary.csv`
- `reports/figures/official_rule_based_asr_clean_tradeoff.png`

Audit logs are written to `logs/audit/`. They are compact and are not required
for reviewing the professor-facing artifact bundle.

## Interpretation Rules

- Use the label `BackdoorLLM official rule-based jailbreak ASR`.
- State that ASR is not external-judged.
- State that the scorer is applied to project-local deterministic `[INST]`
  generations.
- Report Wilson 95% intervals for ASR rates.
- Treat clean-reference perplexity as a probe, not final human utility.
- Do not treat `base_model_only` nonzero ASR as evidence of a learned backdoor;
  it is an artifact of the no-refusal rule.
