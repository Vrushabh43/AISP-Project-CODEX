# Submission Structure

This file describes the cleaned, professor-facing repository layout.

## Reviewer Starting Points

Read these first:

- `README.md`
- `RUN_ORDER.md`
- `SUBMISSION_STRUCTURE.md`
- `FINAL_SUBMISSION_VERDICT.md`
- `FINAL_CLEANUP_SUMMARY.md`
- `IMPLEMENTATION_NOTES.md`
- `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`
- `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`

The copied submission bundle is:

- `final_submission_artifacts/`

## Final Pipeline

Final reproduction and audit scripts are separated into:

- `scripts/final_pipeline/`

Development, smoke, pilot, and older diagnostic scripts are preserved in:

- `archive/final_minimal_cleanup_<timestamp>/dev_smoke_scripts/`

The retained prompt-file script is:

- `scripts/final_pipeline/40_create_real_asr_and_clean_utility_prompt_files.py`

The older small-prompt creator is archived and should not be used for the final
minimal structure.

## Final Results

Final result summaries are collected in:

- `outputs/final_results/`

Key files:

- `outputs/final_results/official_rule_based_asr_eval_summary.csv`
- `outputs/final_results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/final_results/official_rule_based_asr_verification_summary.csv`
- `outputs/final_results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/final_results/clean_utility_perplexity_summary.csv`
- `outputs/final_results/clean_behaviour_similarity_summary.csv`
- `outputs/final_results/final_submission_consistency_check_summary.csv`
- `outputs/final_results/report_ready_main_results.md`
- `outputs/final_results/report_ready_key_findings.md`
- `outputs/final_results/report_ready_case_diagnostics_summary.md`

Intermediate CSVs, per-record outputs, smoke outputs, and old heuristic
summaries were archived unless they are explicitly part of the final result set.

## Final Logs

Final logs are collected in:

- `logs/final_logs/`

This folder keeps only the latest final verification/evaluation/trade-off,
sanity-check, and consistency-check logs. Older logs and child-process folders
are preserved under the final minimal cleanup archive.

## Data and Provenance

Kept final data/provenance files include:

- `data/eval_prompts/official_badnets_jailbreak_full.jsonl`
- `data/eval_prompts/real_asr_official_badnets.jsonl`
- `data/eval_prompts/clean_utility_reference_eval.jsonl`
- `data/eval_prompts/clean_utility_medium.jsonl`
- `external_sources/backdoorllm_official/`
- official ASR verification summaries in `outputs/final_results/`

If the fetched BackdoorLLM source tree is not present in this local Windows
workspace, the verified scorer evidence remains in the saved verification
Markdown/CSV/log artifacts. Re-running source verification requires restoring
the source files used to create those artifacts.

## Source Code

Reusable source code remains in:

- `src/lora_sanitise/__init__.py`
- `src/lora_sanitise/attenuation.py`
- `src/lora_sanitise/lora_io.py`
- `src/lora_sanitise/svd_tools.py`

Blank/docstring-only unused stubs were archived in the earlier final cleanup.

## Final Figures

Kept report-ready figures:

- `reports/figures/report_ready/official_rule_based_asr_clean_tradeoff.png`
- `reports/figures/report_ready/report_clean_utility_key_methods.png`
- `reports/figures/report_ready/report_tradeoff_scatter_key_methods.png`
- `reports/figures/report_ready/report_trigger_rate_key_methods.png`
- `reports/figures/report_ready/report_trigger_reduction_key_methods.png`

Exploratory figures are archived.

## Final Submission Artifacts

`final_submission_artifacts/` contains only professor-facing documents, final
summary results, and final figures. It intentionally excludes:

- `AGENT.md`
- `status.md`
- raw logs
- smoke scripts
- old CSVs
- pilot outputs
- backup files
- full harmful prompts or full generated trigger outputs

## Archive

Archive folders preserve removed material:

- `archive/submission_cleanup_20260523T170729/`
- `archive/final_cleanup_20260530T113656/`
- `archive/final_minimal_cleanup_<timestamp>/`

Each archive includes:

- `ARCHIVE_MANIFEST.md`
- `archive_manifest.csv`

Nothing important was permanently deleted.

## Final Claim Boundary

The final result wording must remain:

BackdoorLLM official rule-based jailbreak ASR scorer applied to project-local
deterministic `[INST]` generations. It is not external-judged ASR. Clean
perplexity is a reference-likelihood probe, not final human clean utility.
