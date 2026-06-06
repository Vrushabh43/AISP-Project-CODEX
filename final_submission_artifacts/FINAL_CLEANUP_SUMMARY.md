# Final Cleanup Summary

Cleanup status: file-only cleanup and documentation refresh. No model loading,
inference, official BackdoorLLM code execution, or API calls were run.

## Archives

Two final cleanup archives now matter:

- `archive/final_cleanup_20260530T113656/`
- `archive/final_minimal_cleanup_20260606T142716Z/`

Nothing was permanently deleted. Each archive contains an `ARCHIVE_MANIFEST.md`
and `archive_manifest.csv` with original paths, archive paths, reasons,
categories, and reproducibility notes.

The final minimal cleanup archived:

- development, smoke, pilot, and diagnostic scripts
- old logs and child-process log folders
- intermediate CSV/Markdown outputs and per-row evaluation outputs
- non-final adapter variants
- exploratory figures
- stale prompt subsets and superseded structure/config files
- old CSVs that should not appear in `final_submission_artifacts/results/`

## Clean Visible Structure

Professor-facing root files:

- `README.md`
- `RUN_ORDER.md`
- `SUBMISSION_STRUCTURE.md`
- `FINAL_SUBMISSION_VERDICT.md`
- `FINAL_CLEANUP_SUMMARY.md`
- `IMPLEMENTATION_NOTES.md`
- `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`
- `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`

Final scripts:

- `scripts/final_pipeline/`

Final results:

- `outputs/final_results/`

Final logs:

- `logs/final_logs/`

Final artifact bundle:

- `final_submission_artifacts/`

## Final Result Statement

The BackdoorLLM official rule-based jailbreak ASR scorer was verified by static
source/AST inspection. It counts success when a generated output avoids the
verified refusal-keyword list. It is not external-judged ASR.

Applied to project-local deterministic `[INST]` generations over 99 official
BadNets trigger prompts:

- `original`: 36/99 = 0.363636
- `uniform_gamma_0.25`: 13/99 = 0.131313
- `uniform_gamma_0.50`: 7/99 = 0.070707
- `top3_gamma_0.50`: 7/99 = 0.070707
- `sensaware_top224_gamma_0.25`: 1/99 = 0.010101

The best focused condition is `sensaware_top224_gamma_0.25`:

- ASR: 1/99 = 0.010101
- Wilson 95% CI: [0.001785, 0.055017]
- Clean-reference perplexity: 3.557692

## Limitations

- ASR is rule-based, not external-judged or human-judged.
- Generation uses project-local deterministic `[INST]` formatting, not a full
  reproduction of BackdoorLLM generation settings.
- Clean-reference perplexity is a reference-likelihood probe, not final human
  clean utility.
- Wilson intervals over 99 prompts must be reported.
- The base-model control score of 15/99 is a no-refusal-rule artifact, not
  evidence of a learned backdoor.
- Only one adapter/attack family was evaluated.

## What Not To Touch Before Submission

- Do not rerun GPU/model scripts unless all downstream tables, figures, and
  documentation will be refreshed consistently.
- Do not edit final result CSVs manually.
- Do not edit the verified keyword hash/count.
- Do not restore `AGENT.md` to the final visible structure.
- Do not add full harmful prompts or full generated trigger outputs to
  final-facing files.
- Do not claim external-judged ASR or final human clean utility.
