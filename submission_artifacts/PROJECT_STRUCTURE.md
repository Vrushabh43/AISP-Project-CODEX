# Project Structure

The repository is organized as a compact submission package.

```text
.
|-- README.md
|-- REPRODUCIBILITY.md
|-- PROJECT_STRUCTURE.md
|-- PROJECT_VERDICT.md
|-- CLEANUP_SUMMARY.md
|-- IMPLEMENTATION_NOTES.md
|-- OFFICIAL_RULE_BASED_ASR_VERIFICATION.md
|-- OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md
|-- data/
|   `-- eval_prompts/
|-- external_sources/
|-- logs/
|   `-- audit/
|-- outputs/
|   `-- results/
|-- reports/
|   `-- figures/
|-- scripts/
|   `-- pipeline/
|-- src/
|   `-- lora_sanitise/
`-- submission_artifacts/
```

## Visible Deliverables

- `README.md`: project overview and main result.
- `PROJECT_VERDICT.md`: final academic verdict and limitations.
- `REPRODUCIBILITY.md`: commands and expected outputs.
- `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`: static scorer verification.
- `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`: sanity checks for final ASR results.
- `outputs/results/`: concise result summaries.
- `reports/figures/official_rule_based_asr_clean_tradeoff.png`: final figure.
- `submission_artifacts/`: professor-facing bundle copied from the visible
  documents, concise results, and final figure.

## Retained Code

The final pipeline code lives in `scripts/pipeline/`. Numeric script prefixes
were removed so the script names describe their purpose directly. Reusable
helper modules remain in `src/lora_sanitise/`.

## Excluded From The Compact Package

The submitted tree intentionally excludes generated model/adapter binaries,
large Hugging Face cache folders, generated `.safetensors` files, development
logs, intermediate CSVs, and generated cache files. Reproducibility is provided
through scripts, documented commands, final summaries, and external model/cache
requirements rather than by shipping large generated artifacts.

The required external assets for full reproduction are:

- Base model: `NousResearch/Llama-2-7b-chat-hf`
- Backdoored LoRA adapter: `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
- Compatible base-model family: `meta-llama/Llama-2-7b-chat-hf` /
  Llama-2-7B-Chat family

These assets should be provided through `HF_HUB_CACHE` and
`TRANSFORMERS_CACHE` when running heavy reproduction.

## Result Files

The reviewer-facing result files are:

- `outputs/results/official_rule_based_asr_eval_summary.csv`
- `outputs/results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/results/official_rule_based_asr_verification_summary.csv`
- `outputs/results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/results/clean_utility_perplexity_summary.csv`
- `outputs/results/clean_behaviour_similarity_summary.csv`
- `outputs/results/report_ready_main_results.md`
- `outputs/results/report_ready_key_findings.md`
- `outputs/results/report_ready_case_diagnostics_summary.md`
- `outputs/results/validation_summary.csv`

## Artifact Bundle

`submission_artifacts/` contains only professor-facing documents, concise result
summaries, and the final figure. It does not include logs, model files, adapter
binaries, prompt payloads, project-memory files, generated caches, or
intermediate development material.
