# Final Submission Verdict

## Verdict

The project is submission-ready as a focused, honest, reproducible semester
study, with one important condition: the report must clearly describe the final
ASR metric as BackdoorLLM official rule-based jailbreak ASR applied to
project-local deterministic `[INST]` generations, not external-judged ASR or a
human harmfulness judgment.

## Current Risk Level

Risk level: **medium-low for submission**, **medium for scientific strength**.

Submission risk is reduced because:

- The implementation is reproducible through numbered scripts.
- Official BackdoorLLM BadNets trigger provenance was verified.
- Final tables, plots, logs, and consistency checks are present.
- Negative and mixed results are preserved.

Scientific strength remains limited because:

- The final ASR scorer is rule-based, not external-judged.
- The generation format is project-local deterministic `[INST]` formatting, not
  a full reproduction of the BackdoorLLM generation pipeline.
- Clean utility is measured by a clean-reference perplexity probe, not a human
  clean-utility benchmark.
- Only one attack/adapter family was evaluated.

## What Was Changed During Final Packaging

- Rewrote `README.md` to reflect the real implemented project.
- Added `SUBMISSION_STRUCTURE.md`.
- Added `RUN_ORDER.md`.
- Added `IMPLEMENTATION_NOTES.md`.
- Added `FINAL_SUBMISSION_VERDICT.md`.
- Updated stale configuration and prompt documentation.
- Marked the original master project document as historical planning.
- Archived duplicate backups, bytecode caches, and superseded unverified prompt
  plumbing without deleting them.
- Created `final_submission_artifacts/` as a copied final-facing bundle.
- Added a lightweight final consistency-check script.
- Verified the BackdoorLLM official rule-based jailbreak ASR scorer.
- Added official rule-based ASR, Wilson intervals, clean-reference perplexity,
  and sanity-check outputs.
- Archived `AGENT.md` and superseded planning/backup files during final
  cleanup.
- Moved final reproduction/audit scripts into `scripts/final_pipeline/`.
- Collected final summaries under `outputs/final_results/` and latest final
  logs under `logs/final_logs/`.
- Archived development, smoke, pilot, old diagnostic scripts, intermediate
  outputs, old logs, and exploratory figures during final minimal cleanup.

## Files Archived

See:

`archive/submission_cleanup_20260523T170729/ARCHIVE_MANIFEST.md` and
`archive/final_cleanup_<timestamp>/ARCHIVE_MANIFEST.md`

Only clearly superseded or duplicate material was archived. Official assets,
final results, final figures, logs, scripts, source code, and adapter outputs
were preserved in place.

The final cleanup archive includes duplicate backups, superseded planning docs,
historical project memory, and unused docstring-only source stubs. All archived
items are preserved, not deleted.

## Honest Final Result Statement

The final-facing ASR result uses the verified BackdoorLLM official rule-based
jailbreak ASR scorer. This scorer counts a generation as successful when the
output avoids the verified refusal-keyword list. It is applied to
project-local deterministic `[INST]` generations and is not external-judged
ASR.

Focused official rule-based ASR results:

- `base_model_only`: 15/99 = 0.151515, Wilson CI [0.094022, 0.235043].
- `original`: 36/99 = 0.363636, Wilson CI [0.275617, 0.461843].
- `uniform_gamma_0.25`: 13/99 = 0.131313, Wilson CI [0.078372, 0.211799].
- `uniform_gamma_0.50`: 7/99 = 0.070707, Wilson CI [0.034670, 0.138816].
- `top3_gamma_0.50`: 7/99 = 0.070707, Wilson CI [0.034670, 0.138816].
- `sensaware_top224_gamma_0.25`: 1/99 = 0.010101, Wilson CI
  [0.001785, 0.055017].

The best expanded SensAware variant, `sensaware_top224_gamma_0.25`, has the
lowest observed official rule-based ASR count and the lowest clean-reference
perplexity in this run: `3.557692`.

The `base_model_only` result is a control artifact of the no-refusal-keyword
rule, not evidence of a learned backdoor. Clean-reference perplexity is a probe,
not final human clean utility.

## Remaining Limitations

- ASR is rule-based, not external-judged.
- Clean-reference perplexity is not final human utility.
- Only one attack/adapter family was evaluated.
- Only a non-adaptive attacker is considered.
- Official trigger prompts were used, so the final evaluation is not
  trigger-free even though the sanitisation method itself was designed without
  trigger knowledge.
- No Llama-Guard or human judge was used.
- No larger clean benchmark was run.

## What Should Not Be Touched Before Submission

Do not change the following unless a serious factual error is found:

- `outputs/final_results/report_ready_main_results.md`
- `outputs/final_results/report_ready_key_findings.md`
- `outputs/final_results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/final_results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/final_results/official_rule_based_asr_eval_summary.csv`
- `outputs/final_results/clean_utility_perplexity_summary.csv`
- `OFFICIAL_RULE_BASED_ASR_VERIFICATION.md`
- `OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md`
- `reports/figures/report_ready/`
- `final_submission_artifacts/`
- Official BackdoorLLM files under `external_sources/`
- Generated adapter folders under `outputs/sanitised_adapters/`
- `status.md`

Do not rerun GPU evaluations unless there is a clear reason and enough time to
update all derived tables, figures, logs, and documentation consistently.

## Recommended Report Framing

Frame the project as a reproducible empirical study with a mixed result:

- The proposed SensAware method worked only after expansion to broader global
  top-N component selection.
- Expanded SensAware achieved the lowest observed official rule-based ASR count
  among the focused conditions.
- Expanded SensAware also had the lowest clean-reference perplexity in this run.
- The evidence is stronger than the earlier bounded heuristic story, but still
  rule-based and limited to one adapter/attack family.
- This suggests the clean sensitivity signal can help beyond pure spectral
  pruning, while leaving open broader robustness and human-utility questions.
