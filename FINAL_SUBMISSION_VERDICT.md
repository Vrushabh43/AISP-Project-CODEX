# Final Submission Verdict

## Verdict

The project is submission-ready as a focused, honest, reproducible semester
study, with one important condition: the report must clearly describe the
metrics as bounded heuristic metrics, not final judged ASR or final judged clean
utility.

## Current Risk Level

Risk level: **medium-low for submission**, **medium for scientific strength**.

Submission risk is reduced because:

- The implementation is reproducible through numbered scripts.
- Official BackdoorLLM BadNets trigger provenance was verified.
- Final tables, plots, logs, and consistency checks are present.
- Negative and mixed results are preserved.

Scientific strength remains limited because:

- The evaluation is bounded and heuristic.
- No external judge was used.
- Clean utility is measured with a small heuristic prompt set.
- Uniform scaling remains the strongest baseline under the current metric.

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

## Files Archived

See:

`archive/submission_cleanup_20260523T170729/ARCHIVE_MANIFEST.md`

Only clearly superseded or duplicate material was archived. Official assets,
final results, final figures, logs, scripts, source code, and adapter outputs
were preserved in place.

Archived item count: 38 moved items, all preserved under the archive folder.

## Honest Final Result Statement

Under the bounded heuristic evaluation, the original backdoored adapter had
trigger success `0.6061` and clean utility `0.9667`.

The best expanded SensAware variant, `sensaware_top224_gamma_0.25`, reduced
trigger success to `0.0101` while keeping clean utility at `0.9667`.

This beats the best spectral-only baseline, `top3_gamma_0.50`, which had trigger
success `0.0606` and clean utility `0.9333`.

It does not beat the strongest uniform-scaling baseline,
`uniform_gamma_0.25`, which had trigger success `0.0000` and clean utility
`0.9667`.

Therefore the final result is mixed: expanded SensAware adds value over
spectral-only attenuation in this bounded run, but it does not establish
superiority over uniform scaling.

## Remaining Limitations

- Metrics are bounded heuristic metrics, not final judged ASR/utility.
- Clean utility may be too weak to detect global adapter weakening.
- Only one attack/adapter family was evaluated.
- Only a non-adaptive attacker is considered.
- Official trigger prompts were used, so the final evaluation is not
  trigger-free even though the sanitisation method itself was designed without
  trigger knowledge.
- No Llama-Guard or human judge was used.
- No larger clean benchmark or perplexity benchmark was run.

## What Should Not Be Touched Before Submission

Do not change the following unless a serious factual error is found:

- `outputs/report_ready_main_results.md`
- `outputs/report_ready_key_findings.md`
- `outputs/consolidated_tradeoff_results.csv`
- `outputs/expanded_sensaware_asr_utility_tradeoff_summary.csv`
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
- Expanded SensAware outperformed spectral-only attenuation.
- Uniform scaling remained the strongest bounded baseline.
- This suggests the clean sensitivity signal can help beyond pure spectral
  pruning, but the current evidence does not prove selective sanitisation is
  better than globally weakening the adapter.
