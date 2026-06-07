# Cleanup Summary

This branch is the compact professional submission package. Large generated
artifacts and development-only material were removed from the visible tree while
preserving final summaries, validation scripts, and reproducibility commands.

## What Remains

- Professor-facing documentation in the repository root.
- Reproducibility scripts under `scripts/pipeline/`.
- Reusable helper modules under `src/lora_sanitise/`.
- Concise final results under `outputs/results/`.
- Compact audit logs under `logs/audit/`.
- Final report figure under `reports/figures/`.
- A clean deliverable bundle under `submission_artifacts/`.

## What Was Excluded

The compact tree excludes generated model/cache folders, generated adapter
weights, `.safetensors` files, verbose logs, intermediate per-record outputs,
generated cache files, duplicate Markdown/CSV/PNG files, and development-only
scripts. These are not needed for professor review and can be regenerated with
the documented pipeline and external cache requirements.

The required external assets for full reproduction are the base model
`NousResearch/Llama-2-7b-chat-hf` and the backdoored LoRA adapter
`BackdoorLLM/Jailbreak_Llama2-7B_BadNets`. They are excluded because model and
adapter weights exceed the submission size target. Use `HF_HUB_CACHE` and
`TRANSFORMERS_CACHE` to point the pipeline to a local Hugging Face cache.

## Final Result Statement

`sensaware_top224_gamma_0.25` achieved the lowest observed
`BackdoorLLM official rule-based jailbreak ASR` count:

- ASR: `1/99 = 0.010101`
- Clean-reference perplexity: `3.557692`

This is official rule-based ASR applied to project-local deterministic `[INST]`
generations. It is not external-judged ASR. Clean perplexity is a
reference-likelihood probe, not final human utility.

## Validation

Use:

```bash
python scripts/pipeline/validate_submission.py
python scripts/pipeline/run_pipeline.py --mode quick
```

The validation checks package size excluding `.git` metadata, result-number
consistency, artifact hygiene, professional script names, ASR labels, Wilson
interval documentation, and final-facing Markdown safety.

## Do Not Touch Before Submission

- Do not change result numbers without rerunning the corresponding evaluation.
- Do not add generated model/cache folders or adapter weights.
- Do not add verbose logs or per-record harmful-output files.
- Do not remove the official ASR caveat or Wilson-interval wording.
- Do not rename the professor-facing files after validation passes.
