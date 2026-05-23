# Submission Structure

This file explains which project files are final-facing, which are raw
artifacts, and which are preserved as historical or archived material.

## Final-Facing Files

These are the files a reviewer should start with:

- `README.md`: concise project overview, honest result summary, reproduction
  pointers, and caveats.
- `RUN_ORDER.md`: reproducible script order grouped by phase.
- `IMPLEMENTATION_NOTES.md`: what was actually implemented and how it differs
  from the original plan.
- `FINAL_SUBMISSION_VERDICT.md`: submission readiness, risks, limitations, and
  what should not be changed before submission.
- `outputs/report_ready_main_results.md`: compact report-ready result table.
- `outputs/report_ready_key_findings.md`: cautious key findings.
- `outputs/report_ready_case_diagnostics_summary.md`: aggregate diagnostics
  without full prompt/output text.
- `reports/figures/report_ready/`: report-ready figures.
- `final_submission_artifacts/`: copied final-facing bundle.

## Important Project Memory

- `AGENT.md`: stable project memory, scope, threat model, and workflow rules.
- `status.md`: chronological project log, commands, outputs, failures, and
  decisions.
- `task.txt`: original course/project task description.
- `lora_sanitisation_master_project.md`: historical phase-1 planning document.
  It is preserved for provenance and should not be read as the exact final
  implementation.

## Source Code

- `src/lora_sanitise/`: reusable adapter I/O, compact SVD, attenuation, and
  related helper code.
- `scripts/`: numbered experiment, analysis, plotting, packaging, and
  consistency-check scripts.
- `scripts/35_final_submission_consistency_check.py`: lightweight final
  packaging check. It does not load models or run inference.

## Configs

- `configs/experiment.yaml`: final lightweight configuration summary matching
  the implemented bounded study.
- Superseded configs are preserved under
  `archive/submission_cleanup_<timestamp>/stale_or_superseded/`.

## Data and Prompt Provenance

- `data/eval_prompts/clean_utility_small.jsonl`: small clean prompt set.
- `data/eval_prompts/clean_utility_medium.jsonl`: bounded clean-utility prompt
  set used in final heuristic comparisons.
- `data/eval_prompts/official_badnets_jailbreak_small.jsonl`: small official
  BadNets prompt subset.
- `data/eval_prompts/official_badnets_jailbreak_full.jsonl`: 99-record official
  BadNets trigger prompt file used in bounded heuristic evaluation.
- `external_sources/backdoorllm_official/`: fetched official BackdoorLLM
  metadata and test-data assets used to verify the trigger source.

The unverified placeholder trigger probe file was superseded by official
BackdoorLLM prompts and archived during final cleanup.

## Final Result Tables

- `outputs/report_ready_main_results.csv`
- `outputs/report_ready_main_results.md`
- `outputs/report_ready_key_findings.md`
- `outputs/report_ready_case_diagnostics_summary.md`
- `outputs/consolidated_tradeoff_results.csv`
- `outputs/expanded_sensaware_asr_utility_tradeoff_summary.csv`
- `outputs/eval_case_diagnostics.csv`

## Final Figures

- `reports/figures/report_ready/report_tradeoff_scatter_key_methods.png`
- `reports/figures/report_ready/report_trigger_rate_key_methods.png`
- `reports/figures/report_ready/report_trigger_reduction_key_methods.png`
- `reports/figures/report_ready/report_clean_utility_key_methods.png`

Older figure backups are archived, not deleted.

## Raw Logs and Intermediate Artifacts

- `logs/`: timestamped JSON logs for environment checks, inspections,
  generation, evaluations, analyses, and report-ready scripts.
- `outputs/`: CSV/Markdown outputs, including intermediate diagnostics and
  final report-ready outputs.
- `outputs/sanitised_adapters/`: generated adapter variants and their
  sanitisation reports.

These are intentionally preserved. Failed or less successful experiments are not
hidden because they document the research path.

## Archived Material

Archive folders use:

`archive/submission_cleanup_<timestamp>/`

The archive contains duplicate backups, bytecode caches, superseded unverified
prompt plumbing, and stale/template-only files. Each cleanup archive includes an
`ARCHIVE_MANIFEST.md` listing every moved item, its reason, and whether it is
needed for reproducibility.

## What Not To Use As Final Evidence

- Unverified placeholder trigger probes.
- Early smoke-test-only outputs.
- First small SensAware variants as the main proposed result.
- Any metric described as final judged ASR or final judged clean utility.

The final result wording must remain: bounded heuristic evaluation, not final
judged ASR/utility.
