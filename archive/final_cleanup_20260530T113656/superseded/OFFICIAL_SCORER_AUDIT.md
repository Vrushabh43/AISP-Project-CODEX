# Official Scorer Audit

Static audit status: file/text and AST inspection only. No BackdoorLLM code was
executed, no model was loaded, no inference was run, and no external API was
called.

## Scorer Finding

`eval_ASR_of_backdoor_models` was found in `external_sources/backdoorllm_official_source/attack/DPA/backdoor_evaluate.py` at lines `198` to `261`.

- Scoring mechanism classification: `external_judge`
- Local reproducibility under current constraints: `false`
- Can call future metric official ASR without more setup: `false`
- Recommended label if used now: `BackdoorLLM-aligned ASR proxy unless the external judge environment is reproduced`

## Reproducibility Decision

Official scorer function found, but static audit indicates external judge/API scoring. Under current no-external-API constraints it is not locally reproducible.

If an external judge/API is required, future results should not be called
locally reproducible official ASR unless the exact judge, credentials,
dependency versions, prompts, and scoring path are reproduced and explicitly
approved. Under the current no-external-API constraint, use an ASR-proxy label
instead.

## Clean Dataset

- Dataset path: `external_sources/backdoorllm_official_source/DefenseBox/data/test_data/clean/jailbreak/test_data_no_trigger.json`
- Dataset readable: `true`
- Record count: `99`
- Schema keys: `input, instruction, output`
- Usable as reference answers for NLL/perplexity: `true`
- Appears official-clean: `true`
- Appears generic-clean: `false`

## Caveats

- This audit does not execute the official scorer.
- Full harmful prompts and full generated outputs are not printed or stored.
- Clean-reference NLL/perplexity remains a cheap probe. If values are nearly
  identical across base, original, uniform scaling, and SensAware, interpret
  that as a small measurable clean-task footprint on this dataset, not as proof
  that all defences preserve adapter utility.
- A small blinded clean-output quality rubric remains recommended if time
  permits.
