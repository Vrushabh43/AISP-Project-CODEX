# Base Control and Clean Behaviour Preservation Plan

Planning status: architecture only. No code has been implemented from this plan,
no models have been loaded, no inference has been run, and no final result files
have been modified.

## 1. Purpose

The current final result is mixed: expanded SensAware beats spectral-only
attenuation under bounded heuristic evaluation, but it does not beat the
strongest uniform-scaling baseline. The main unresolved caveat is whether
uniform scaling is genuinely sanitising the backdoored behaviour or simply
erasing adapter-specific behaviour by globally weakening the LoRA update.

Add a `base_model_only` control to make this caveat testable. The base model
without the BadNets LoRA adapter is the natural reference point for "no adapter
behaviour." However, `base_model_only` must not be interpreted by trigger
success alone. The trigger token was learned by the adapter, not the base model,
so the base model alone will probably show near-zero trigger success and
reasonable heuristic clean utility. That result would be expected and should not
be counted as a defence win.

The base control is useful only when paired with clean-output preservation
similarity:

- Compare defended-adapter clean outputs against the original backdoored
  adapter's clean outputs.
- Compare defended-adapter clean outputs against `base_model_only` clean
  outputs.
- Use these comparisons to ask whether a defence preserves adapter-like clean
  behaviour or drifts toward base-model-only behaviour.

This extension would therefore test a stronger and more honest question:

Does SensAware preserve original adapter-like clean behaviour better than
uniform scaling at similar bounded trigger reduction, or do both methods mostly
collapse toward the base model?

## 2. Adapters and Models to Compare

Use a small comparison set to keep the optional extension focused:

| Name | Role | Include? | Notes |
|---|---|---:|---|
| `base_model_only` | No-adapter control | Required | Load the base chat model without any LoRA adapter. Do not treat trigger failure alone as a defence success. |
| `original` | Backdoored adapter reference | Required | Clean outputs define the adapter-like behaviour reference. |
| `uniform_gamma_0.25` | Strongest current uniform baseline | Required | Current bounded trigger success is `0.0000`; key erasure-risk baseline. |
| `uniform_gamma_0.50` | Less aggressive uniform baseline | Required | Helps separate attenuation strength from total collapse. |
| `top3_gamma_0.50` | Best spectral-only baseline | Required | Current best spectral-only comparison. |
| `sensaware_top224_gamma_0.25` | Best SensAware variant | Required | Main proposed method variant. |
| `top1_gamma_0.50` | Simpler spectral-only baseline | Skip by default | Include only if explicitly requested later. |

Do not broaden this optional extension into another large adapter sweep. Extra
SensAware variants should be excluded unless there is a specific report-facing
reason and enough time to update all derived documentation consistently.

## 3. Deterministic Generation Requirement

Deterministic greedy decoding is a non-negotiable requirement for this optional
extension. Similarity analysis is only interpretable if output differences come
from the adapter/control condition, not from sampling noise.

All model/control conditions must use exactly the same generation setup:

- `do_sample=False`
- the same `max_new_tokens`
- the same `[INST] ... [/INST]` prompt formatting
- the same tokenizer and chat/input formatting logic
- the same decoding, padding, EOS, and truncation settings
- the same generation kwargs for `base_model_only`, `original`, uniform
  baselines, spectral-only baseline, and SensAware

Script `36` should fail fast if the generation settings differ across
conditions. Any future rerun with changed decoding settings must be treated as a
new experiment, not mixed into the earlier outputs.

## 4. Prompt Sets

Use only existing verified prompt files:

- Official BadNets trigger prompts:
  `data/eval_prompts/official_badnets_jailbreak_full.jsonl`
  - `99` records.
  - Use only for bounded heuristic trigger evaluation.
  - Do not create new harmful prompts.
  - Do not print full prompts or full generated outputs.
- Clean utility prompts:
  `data/eval_prompts/clean_utility_medium.jsonl`
  - `30` prompts.
  - Use for current clean utility heuristic and clean-behaviour preservation
    similarity.

No new harmful prompt source should be added. If a prompt record must be
referenced in logs or CSVs, use stable record IDs, indices, hashes, and aggregate
flags rather than full prompt text.

## 5. Metrics

### A. Trigger Metric

Compute the same bounded heuristic trigger metrics used in the final package:

- Preliminary trigger success rate.
- Wilson 95% confidence interval for trigger success.
- Trigger refusal rate.
- Optional count columns:
  - `trigger_success_count`
  - `trigger_refusal_count`
  - `n_trigger_records`

Interpretation constraints:

- Mark these as bounded heuristic metrics, not final judged ASR.
- For `base_model_only`, near-zero trigger success is expected and should not
  be framed as sanitisation.
- Do not compare `base_model_only` to defended adapters as if it were an
  adapter defence.

### B. Clean Utility Heuristic

Reuse the current clean utility heuristic:

- Clean success rate.
- Clean refusal rate.
- Too-short output rate.
- Optional count columns:
  - `clean_success_count`
  - `clean_refusal_count`
  - `too_short_count`
  - `n_clean_records`

Interpretation constraints:

- Mark this as heuristic clean utility, not final judged clean utility.
- A high base-model clean score does not prove that adapter-specific behaviour
  was preserved.
- This metric is necessary but insufficient, which is why clean-output
  similarity is being added.

### C. Clean-Behaviour Preservation

For each clean prompt, compare generated outputs under:

- `original`
- `base_model_only`
- each defended adapter

For each defended adapter, compute:

- Output similarity to the original adapter:
  `similarity_to_original`
- Output similarity to the base model:
  `similarity_to_base`
- A drift margin:
  `similarity_to_original - similarity_to_base`

Similarity anchors:

- `original` versus `original`: upper ceiling and implementation sanity check.
- `original` versus `base_model_only`: reference distance between adapter-like
  and no-adapter clean behaviour.
- Unrelated original clean-output pairs: rough similarity floor for prompt-level
  unrelatedness within the same model condition.

Recommended simple metric first:

- Token-level similarity using normalized token overlap.
- Suggested normalization:
  - lowercase text
  - strip leading/trailing whitespace
  - split into simple word/punctuation tokens
  - optionally drop repeated whitespace
- Suggested score:
  `overlap(a, b) = multiset_intersection_count(tokens_a, tokens_b) / max(len(tokens_a), len(tokens_b), 1)`

This score is intentionally simple, reproducible, and dependency-light. It is
not semantic equivalence.

For every adapter/control comparison over the `30` clean prompts, report:

- mean
- standard deviation
- minimum
- maximum
- median
- bootstrap confidence interval or, if keeping the first pass simpler, a clearly
  labelled simple confidence interval

The clean prompt set has only `30` prompts. Similarity gaps must therefore be
interpreted cautiously, especially when intervals overlap or when a conclusion
depends on a small mean difference.

Optional stronger metric:

- Sentence embedding cosine similarity, only if already available without heavy
  dependency installation or large new downloads.
- This should be optional and secondary. The first implementation should not
  depend on sentence-transformers or any model download.

Output drift interpretation:

- If uniform scaling is closer to `base_model_only` while SensAware is closer to
  `original`, this supports the claim that SensAware better preserves
  adapter-like clean behaviour.
- If both uniform scaling and SensAware are closer to `base_model_only`, both
  may be adapter erasure under this analysis.
- If both are closer to `original`, both preserve adapter-like clean behaviour
  under this analysis.
- If the result is mixed by prompt or metric, report the mixed result directly.
- Do not force the analysis to favour SensAware.

## 6. Wilson Confidence Intervals

Add Wilson 95% confidence intervals for all trigger success rates in this
optional extension and, if script `38` is later implemented, for the already
reported trigger rates as well.

For success count `k`, sample size `n`, and `z = 1.96`:

```text
p_hat = k / n
denom = 1 + z^2 / n
centre = p_hat + z^2 / (2n)
radius = z * sqrt((p_hat * (1 - p_hat) / n) + (z^2 / (4n^2)))
lower = (centre - radius) / denom
upper = (centre + radius) / denom
```

Why this matters:

- The trigger set has `99` records, so small count differences are uncertain.
- `0/99` and `1/99` should not be treated as a decisive separation by
  themselves.
- Wilson intervals should be used to avoid overclaiming, especially for
  `uniform_gamma_0.25` versus `sensaware_top224_gamma_0.25`.
- The final report should describe these as uncertainty bands around bounded
  heuristic rates, not as final judged ASR confidence intervals.

## 7. Planned Scripts

These scripts are proposed only. Do not implement them until the optional
extension is explicitly approved.

### `scripts/36_base_model_control_eval.py`

Purpose:

- Run the base-model/no-adapter control and selected adapter evaluations in a
  single controlled evaluation script.
- Reuse the existing bounded evaluation logic where possible.
- Produce aggregate trigger and clean utility summaries plus clean-generation
  records needed for later similarity analysis.

Input files:

- `data/eval_prompts/official_badnets_jailbreak_full.jsonl`
- `data/eval_prompts/clean_utility_medium.jsonl`
- Base model cache for `NousResearch/Llama-2-7b-chat-hf`
- Original adapter cache for `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
- Existing generated adapter directories:
  - `outputs/sanitised_adapters/uniform_gamma_0.25/`
  - `outputs/sanitised_adapters/uniform_gamma_0.50/`
  - `outputs/sanitised_adapters/top3_gamma_0.50/`
  - `outputs/sanitised_adapters/sensaware_top224_gamma_0.25/`

Output files:

- `outputs/base_model_control_eval_outputs.csv`
- `outputs/base_model_control_eval_summary.csv`
- `logs/base_model_control_eval_<timestamp>.json`

Exact responsibilities:

- Load the base model once per isolated run configuration, following the
  existing safe evaluation pattern.
- Evaluate `base_model_only` without attaching a LoRA adapter.
- Evaluate the selected adapter variants with the same generation settings.
- Enforce deterministic greedy decoding with identical `do_sample=False`,
  `max_new_tokens`, `[INST]` formatting, tokenizer handling, and generation
  kwargs across all conditions.
- Apply the existing bounded trigger and clean utility heuristics.
- Record per-record metadata needed for downstream similarity:
  - prompt set
  - prompt ID or row index
  - prompt hash
  - adapter/model name
  - generated output text for clean prompts only, if needed for script `37`
  - trigger outputs redacted or omitted
  - output length
  - heuristic flags
- Write aggregate rates and counts to the summary CSV.

Safety constraints:

- Do not print full prompts.
- Do not print full generated outputs.
- Do not store full trigger outputs.
- Keep full clean outputs only if needed for similarity analysis, and do not
  copy that intermediate CSV into `final_submission_artifacts/`.
- Do not add new harmful prompts.
- Do not update report-ready result files.
- Do not modify existing adapter directories.

### `scripts/37_clean_behaviour_similarity_analysis.py`

Purpose:

- Quantify whether defended adapters' clean outputs are closer to the original
  adapter or to the base model.
- Keep the first analysis lightweight and reproducible.

Input files:

- `outputs/base_model_control_eval_outputs.csv`
- `outputs/base_model_control_eval_summary.csv`
- `data/eval_prompts/clean_utility_medium.jsonl` for prompt IDs/hashes only,
  not for printing full prompt text.

Output files:

- `outputs/clean_behaviour_similarity_summary.csv`
- Optional diagnostic log:
  `logs/clean_behaviour_similarity_analysis_<timestamp>.json`

Exact responsibilities:

- Filter to clean prompt records only.
- Align outputs by prompt ID across `original`, `base_model_only`, and each
  defended adapter.
- Compute token-level normalized overlap for:
  - defended output versus original output
  - defended output versus base-model output
  - original output versus base-model output, as a reference
  - original output versus itself, as an upper-ceiling sanity check
  - unrelated original output pairs, as a rough similarity floor
- Aggregate mean, standard deviation, minimum, maximum, median, and a bootstrap
  or simple confidence interval by adapter/comparison.
- Compute the drift margin:
  `mean_similarity_to_original - mean_similarity_to_base`.
- Optionally compute sentence-embedding cosine similarity only when no heavy new
  dependencies or downloads are required.

Safety constraints:

- Do not print full prompts.
- Do not print full generated outputs.
- Do not include full prompt/output examples in Markdown report files.
- Do not claim semantic equivalence from token overlap.
- State clearly that `30` clean prompts is a small sample and small similarity
  gaps are not decisive.
- Treat missing or failed generations explicitly instead of silently dropping
  them.

### `scripts/38_wilson_ci_and_final_comparison.py`

Purpose:

- Combine current final bounded results with the optional base-control and
  similarity results.
- Add Wilson confidence intervals to trigger success rates.
- Prepare a compact optional comparison table and, if implemented, a plot.

Input files:

- `outputs/consolidated_tradeoff_results.csv`
- `outputs/base_model_control_eval_summary.csv`
- `outputs/clean_behaviour_similarity_summary.csv`
- Existing report-ready result files as read-only references only:
  - `outputs/report_ready_main_results.md`
  - `outputs/report_ready_key_findings.md`

Output files:

- `outputs/wilson_ci_tradeoff_summary.csv`
- Optional figure:
  `reports/figures/report_ready/base_control_similarity_plot.png`
- Optional log:
  `logs/wilson_ci_and_final_comparison_<timestamp>.json`

Exact responsibilities:

- Compute Wilson 95% intervals from success counts and sample sizes.
- Preserve existing bounded trigger and clean utility numbers.
- Add similarity-to-original, similarity-to-base, and drift-margin columns.
- Mark metric families clearly:
  - bounded heuristic trigger metric
  - heuristic clean utility metric
  - clean-output similarity metric
- Prepare report-facing language fragments only after results are computed.
- If a plot is created, show trigger rate with uncertainty and clean-behaviour
  drift without implying final judged ASR or final utility.

Safety constraints:

- Do not print full prompts or full outputs.
- Do not update `README.md`, `FINAL_SUBMISSION_VERDICT.md`, report-ready
  tables, or `final_submission_artifacts/` automatically.
- Do not overwrite final figures without a deliberate consistency-check step.
- Do not turn optional extension outputs into final evidence until reviewed.

## 8. Output Files Expected Later

If this optional extension is implemented, expected new outputs are:

- `outputs/base_model_control_eval_outputs.csv`
- `outputs/base_model_control_eval_summary.csv`
- `outputs/clean_behaviour_similarity_summary.csv`
- `outputs/wilson_ci_tradeoff_summary.csv`
- `reports/figures/report_ready/base_control_similarity_plot.png` if a plot is
  later implemented

These files are not created by this planning document. They should not be copied
into `final_submission_artifacts/` until the optional extension is complete,
reviewed, and consistency-checked.

## 9. Allowed Claims If Results Support Them

Allowed only if supported by the optional results:

- SensAware preserves adapter-like clean behaviour better than uniform scaling,
  if SensAware is consistently closer to `original` clean outputs and uniform
  scaling is closer to `base_model_only`.
- Uniform scaling may be adapter erasure, if uniform scaling is closer to
  `base_model_only` than to `original` on clean prompts.
- SensAware and uniform scaling both preserve adapter-like behaviour, if both
  are close to `original`.
- SensAware and uniform scaling may both erase adapter-specific behaviour, if
  both drift toward `base_model_only`.

Claims that remain disallowed:

- Do not claim final judged ASR.
- Do not claim final judged clean utility.
- Do not claim the backdoor is fully removed.
- Do not claim SensAware beats uniform scaling unless the trigger, clean
  utility, uncertainty, and clean-behaviour similarity results support that
  narrower statement.
- Do not claim token overlap proves semantic preservation.

## 10. Risks

- SensAware may also collapse toward `base_model_only`, weakening the final
  story.
- Uniform scaling may preserve adapter-like outputs better than expected,
  weakening the erasure caveat.
- Token overlap may be too shallow to detect paraphrased but semantically
  similar outputs.
- The clean similarity set has only `30` prompts, so apparent differences may
  be unstable.
- Sentence embedding similarity may require dependencies or downloads that are
  not worth adding before submission.
- Base model may answer clean prompts well, showing why current clean utility is
  insufficient but not necessarily proving adapter erasure.
- Generation variance may affect similarity unless deterministic decoding is
  used consistently.
- The optional experiment may strengthen or weaken the final story; either
  outcome must be reported honestly.

## 11. Decision Rule

If time is short, do not run this optional extension. The current project is
already submission-packaged and has passed the final consistency check.

If the optional extension is implemented:

- Run it only after explicitly deciding there is enough time to update derived
  documentation consistently.
- Add scripts `36`, `37`, and `38` without modifying the current final results.
- Generate new outputs and inspect them before changing any report-facing
  claims.
- Update `README.md`, `FINAL_SUBMISSION_VERDICT.md`, report-ready tables, and
  `final_submission_artifacts/` only after the new results are generated,
  reviewed, and consistency-checked.
- If results are unfavorable or ambiguous, report them honestly.
- If results are not fully integrated, leave the final submission package as it
  currently stands and describe this plan as future work only.
