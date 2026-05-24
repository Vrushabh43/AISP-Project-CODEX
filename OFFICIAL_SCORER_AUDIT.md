# Official Scorer Audit

Status: static audit workflow prepared. No BackdoorLLM code was executed, no
model was loaded, no inference was run, and no external API was called.

## Current Discovery Finding

Script `39` located the official scorer function:

- Function: `eval_ASR_of_backdoor_models`
- File:
  `external_sources/backdoorllm_official_source/attack/DPA/backdoor_evaluate.py`
- Discovery classification: `external_judge`

Because the discovered scoring path is classified as an external-judge scorer,
the project should not automatically treat future local results as reproducible
official ASR. The line-level static audit in script `39b` is required before
implementing scripts `40` to `43`.

## Local Reproducibility Decision

Conservative current decision:

- Official scorer found: `true`
- Official scorer reproducible locally: `false` unless the static audit shows
  purely local rule-based scoring, or the exact external judge environment is
  reproduced and explicitly approved
- Can call future results official ASR now: `false`
- Recommended label if external judging cannot be reproduced:
  `BackdoorLLM-aligned ASR proxy`

If script `39b` confirms that the scorer uses an external judge/API, then under
the current no-external-API constraint future local scoring should not be called
official judged ASR.

## Clean Dataset Finding

Script `39` found a clean dataset candidate:

`external_sources/backdoorllm_official_source/DefenseBox/data/test_data/clean/jailbreak/test_data_no_trigger.json`

Discovery metadata:

- Schema keys: `instruction`, `input`, `output`
- Perplexity/NLL feasible: `true`
- Appears official-clean: likely, based on source path

This dataset appears usable as a reference-answer source for a clean NLL or
perplexity probe, subject to the line-level dataset audit from script `39b`.

## Caveats

- Perplexity/NLL is a cheap probe, not a full clean-utility metric.
- If NLL/perplexity is nearly identical across `base_model_only`, `original`,
  uniform scaling, and SensAware, interpret that as a small measurable
  clean-task footprint on this dataset, not proof that all defences preserve
  adapter utility.
- A small blinded clean-output quality rubric remains recommended if time
  permits.
- Do not update final submission artifacts until stronger evaluation results
  are fully run, reviewed, and consistency-checked.

## Next Step

Run the static audit on the server where the fetched BackdoorLLM source snapshot
exists:

```bash
python scripts/39b_audit_official_asr_scorer.py
```

Expected outputs:

- `logs/official_asr_scorer_audit_<timestamp>.json`
- `outputs/official_asr_scorer_audit_summary.csv`
- `outputs/clean_utility_dataset_audit_summary.csv`
- `OFFICIAL_SCORER_AUDIT.md` updated with the line-level audit result
