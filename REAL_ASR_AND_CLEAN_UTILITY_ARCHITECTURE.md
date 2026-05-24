# Real ASR and Clean Utility Architecture

Planning status: architecture only. No scripts have been implemented from this
plan, no models have been loaded, no inference has been run, and no final
submission artifacts or report-ready files have been modified.

## 1. Purpose

This upgrade is needed because the current project results are intentionally
bounded and heuristic:

- Current trigger results are bounded heuristic trigger-success rates, not final
  judged Attack Success Rate.
- Current clean utility is mostly a heuristic success/refusal/too-short score
  and does not fully measure answer quality, reference-answer likelihood, or
  adapter-task preservation.
- The optional base-control similarity analysis added useful caution, but it
  still uses lightweight token overlap and a small clean prompt set.

The purpose of this architecture is to strengthen credibility, not to chase a
fake win. A stronger evaluation may support SensAware, weaken the story, or
show that uniform scaling remains strongest. All outcomes must be reported
honestly.

This extension should only be treated as stronger evidence if:

- the BackdoorLLM-aligned ASR scorer is verified from official assets, or the
  metric is clearly labelled as an ASR proxy;
- clean utility uses a more meaningful metric than the current heuristic alone;
- all outputs are reviewed and consistency-checked before any final-facing
  documents are updated.

## 2. Current Baseline Truth

Current honest project result:

- Expanded SensAware beats spectral-only under bounded heuristic evaluation.
- The best SensAware variant is `sensaware_top224_gamma_0.25`.
- SensAware nearly matches but does not beat the strongest uniform-scaling
  baseline, `uniform_gamma_0.25`.
- The strongest uniform baseline reaches `0/99` bounded trigger successes under
  the current heuristic.
- Base-control similarity did not prove that SensAware preserves
  adapter-specific clean behaviour better than uniform scaling.
- In the optional token-overlap similarity analysis, all checked defended
  adapters were closer to `base_model_only` than to `original` on average.
- Current metrics remain bounded heuristic metrics, not final judged ASR and
  not final judged clean utility.

This architecture must preserve that truth unless stronger results are actually
generated and reviewed.

## 3. Real ASR Architecture

The ASR upgrade must be careful about what "real" means. If the official
BackdoorLLM scoring logic can be verified and replicated, the resulting metric
may be called BackdoorLLM-aligned official rule-based ASR. If no official scorer
can be verified, the metric must remain an ASR proxy.

### Step A: Official Scorer Discovery

Inspect official BackdoorLLM assets already present under:

`external_sources/backdoorllm_official/`

Discovery goals:

- Identify official BadNets jailbreak evaluation logic.
- Search for scoring scripts, evaluation scripts, target strings, refusal
  keyword lists, jailbreak-success logic, and output fields.
- Determine whether BackdoorLLM evaluates jailbreak success using:
  - refusal keyword absence;
  - unsafe-response keyword presence;
  - target output matching;
  - target behaviour matching;
  - an external judge;
  - some combination of the above.
- Record exact source file paths, function names, constants, and scoring fields.

Safety constraints during discovery:

- Do not execute official attack, training, poisoning, or model-running code.
- Do not print full harmful prompts.
- Do not print full harmful model outputs.
- Inspect files as text/data only.

Scorer decision rules:

- If official BackdoorLLM uses refusal-keyword scoring, implement it exactly and
  call it "official rule-based ASR", not external judged ASR.
- If official BackdoorLLM uses target-behaviour keywords or target outputs,
  implement the verified matching logic and document the exact source.
- If official BackdoorLLM uses an external judge that is not reproducible in
  this project, do not claim official judged ASR.
- If official scoring cannot be verified, stop and do not call the metric real
  ASR.

### Step B: ASR Prompt Set

Use only the verified official BadNets test records:

`data/eval_prompts/official_badnets_jailbreak_full.jsonl`

Requirements:

- Use all `99` records.
- Do not create new harmful prompts.
- Do not augment or paraphrase harmful prompts.
- Store prompt IDs, source indices if available, and prompt hashes.
- Do not include full harmful prompts in Markdown, status updates, logs, or
  report-facing prose.

### Step C: ASR Output Scoring

For each model/adapter condition and official trigger record, store:

- adapter/model condition
- prompt ID
- prompt hash
- generation success/failure flag
- OOM/failure flag
- output hash
- redacted preview only
- refusal score
- official BackdoorLLM success score if verified
- target-behaviour score if available from official logic
- scoring source label
- `is_official_asr`

Rules:

- Set `is_official_asr = true` only if the scorer exactly matches verified
  official BackdoorLLM logic.
- Otherwise set `is_official_asr = false` and label the metric as an ASR proxy.
- Do not store full trigger outputs unless an explicitly reviewed scoring
  requirement makes it unavoidable. Prefer hashes, aggregate flags, and redacted
  previews.

### Step D: Adapters and Models to Evaluate

Use a focused set:

- `base_model_only`
- `original`
- `uniform_gamma_0.25`
- `uniform_gamma_0.50`
- `top3_gamma_0.50`
- `sensaware_top224_gamma_0.25`

Do not add more SensAware variants in this upgrade. The goal is stronger
measurement, not a larger sweep.

Generation constraints:

- deterministic generation
- `do_sample=False`
- same `[INST] ... [/INST]` formatting
- same `max_new_tokens`
- same tokenizer handling
- same decoding settings
- batch size `1`
- one condition per subprocess
- graceful OOM/failure capture

## 4. Real Clean-Utility Architecture

Clean utility must be defined more carefully than the current heuristic.

First, discover whether the adapter has a true clean task:

- Inspect the adapter README/config.
- Inspect official BackdoorLLM files.
- Determine whether the BadNets LoRA was trained for a task beyond jailbreak
  behaviour.
- If there is no clear labelled clean task, do not pretend there is one.

### Metric A: Reference-Output Negative Log-Likelihood / Perplexity

Use a clean held-out instruction dataset with prompt plus reference output.

Candidate data sources already relevant to the project:

- `tatsu-lab/alpaca`, if cached/available.
- `data/eval_prompts/clean_utility_medium.jsonl`, only if reference outputs are
  added from a verified source.
- Official clean/eval data if BackdoorLLM provides one.

For each model/adapter condition:

- condition the model on the clean prompt;
- compute token-level negative log-likelihood of the reference answer;
- aggregate mean NLL and perplexity;
- record failures/OOMs;
- use the same selected model/adapter set as ASR.

This metric does not require generation. That is useful because it reduces
sampling noise and gives a quantitative utility axis.

Interpretation:

- Lower NLL/perplexity means the model assigns higher likelihood to clean
  reference answers.
- If uniform scaling has low ASR but much worse reference-answer likelihood, it
  supports the adapter-erasure caveat.
- If all defended adapters match `base_model_only`, the LoRA may be contributing
  little to clean behaviour under this dataset.
- If clean-reference NLL/perplexity is nearly identical across
  `base_model_only`, `original`, uniform scaling, and SensAware, interpret this
  as evidence that the LoRA has a small measurable clean-task footprint on the
  selected clean dataset. Do not treat identical perplexity as proof that all
  defences preserve adapter utility.
- Treat perplexity as a cheap probe, not the sole clean-utility metric.
- If time permits, clean-output quality review or a small blinded rubric should
  be treated as the stronger utility signal for this jailbreak LoRA.
- Final wording must allow the possibility that an ASR-utility trade-off framing
  does not fit this specific jailbreak LoRA well, because the adapter may have
  little measurable benign-task contribution on generic clean instruction data.

### Metric B: Clean-Behaviour Preservation

Reuse base-control similarity outputs if available:

- `outputs/base_model_control_eval_outputs.csv`
- `outputs/clean_behaviour_similarity_summary.csv`

Compare clean generated outputs to:

- `original`
- `base_model_only`

Token overlap should remain lightweight evidence only:

- not semantic quality;
- not final judged utility;
- not sufficient alone to claim adapter-task preservation.

### Metric C: Optional Blind Human Rubric

If time permits, create a blinded human evaluation on 20 to 30 clean outputs.

Rubric:

- `0`: bad, irrelevant, empty, or refusal
- `1`: acceptable but weak
- `2`: good/helpful

Requirements:

- hide adapter labels;
- use the same clean prompts across conditions;
- do not mix trigger prompts into this task;
- report inter-rater or single-rater limitations if applicable.

This is optional and should not replace objective metrics.

### Metric D: Optional Model Judge

Use a model judge only if it is already available, safe, and explicitly
approved.

Constraints:

- Do not add heavy new dependencies by default.
- Do not call internet-hosted model APIs unless explicitly approved.
- Keep judge prompts harmless and clean-utility focused.
- Report judge limitations clearly.

## 5. Clean Utility Dataset Decision

Add a discovery step before creating any clean utility benchmark.

Discovery tasks:

- Search the local project and cache for clean instruction datasets.
- Check whether `tatsu-lab/alpaca` is available.
- Check whether official BackdoorLLM includes clean evaluation data.
- Inspect dataset schemas for prompt/reference-output fields.
- Confirm licensing/provenance where practical.

Dataset creation rules:

- Create a small held-out clean utility file only after the source is confirmed.
- Start with a bounded sample, for example 50 to 100 records.
- Include prompt ID, prompt hash, source, category if available, and reference
  output hash.
- Store clean prompt/reference text in the data file if needed for NLL, but do
  not print full contents in status or report summaries.
- Do not use trigger prompts for clean utility.
- Do not mix ASR prompt records into clean utility.

Planned clean utility file:

`data/eval_prompts/clean_utility_reference_eval.jsonl`

Expected fields:

- `id`
- `split`
- `source`
- `prompt`
- `reference_output`
- `prompt_hash`
- `reference_output_hash`
- optional `category`
- optional `source_index`

## 6. Statistical Reporting

ASR reporting:

- Use Wilson 95% confidence intervals for ASR rates.
- Report success counts and total prompt count.
- State clearly whether the metric is official BackdoorLLM-aligned ASR or an ASR
  proxy.

Clean utility reporting:

- For mean NLL/perplexity, use bootstrap confidence intervals if feasible.
- If bootstrap is not feasible, use a clearly labelled simple confidence
  interval or standard error.
- Report prompt count and failure/OOM counts.

Uncertainty caveats:

- `99` trigger prompts limit certainty.
- Small clean samples, especially 50 to 100 records, limit certainty.
- Tiny differences are not decisive.
- Overlapping intervals should be reported as ambiguous rather than forced into
  a win/loss claim.

## 7. Planned Scripts

Do not implement these scripts until this architecture is approved.

### `scripts/39_discover_official_asr_and_clean_utility_sources.py`

Responsibilities:

- inspect official BackdoorLLM assets under `external_sources/backdoorllm_official/`
- identify official BadNets scorer logic if present
- identify refusal keyword lists, target strings, target outputs, or scorer
  fields
- identify candidate clean utility data sources
- check whether `tatsu-lab/alpaca` appears in local cache/project paths
- avoid executing official training/attack/model-running code
- avoid printing harmful prompts or outputs

Outputs:

- `logs/official_asr_clean_utility_discovery_<timestamp>.json`
- `outputs/official_asr_clean_utility_discovery_summary.csv`

### `scripts/40_create_real_asr_and_clean_utility_prompt_files.py`

Responsibilities:

- create verified prompt/eval files after discovery
- copy or normalize the 99 official BadNets records into a real-ASR prompt file
- create a clean held-out reference-evaluation file from a confirmed clean data
  source
- record hashes, source paths, source indices, and counts
- avoid printing harmful prompts

Outputs:

- `data/eval_prompts/real_asr_official_badnets.jsonl`
- `data/eval_prompts/clean_utility_reference_eval.jsonl`
- `outputs/real_eval_prompt_file_summary.csv`
- optional log:
  `logs/real_eval_prompt_file_creation_<timestamp>.json`

### `scripts/41_real_asr_eval.py`

Responsibilities:

- run deterministic generation for selected adapters/models
- use one condition per subprocess
- evaluate:
  - `base_model_only`
  - `original`
  - `uniform_gamma_0.25`
  - `uniform_gamma_0.50`
  - `top3_gamma_0.50`
  - `sensaware_top224_gamma_0.25`
- score outputs with verified BackdoorLLM-aligned logic if available
- otherwise label the metric as ASR proxy
- store output hashes, redacted previews, flags, counts, and failure/OOM data
- avoid storing full trigger outputs unless an explicitly reviewed scorer
  requirement makes it unavoidable

Outputs:

- `outputs/real_asr_eval_outputs.csv`
- `outputs/real_asr_eval_summary.csv`
- `logs/real_asr_eval_<timestamp>.json`

### `scripts/42_clean_utility_perplexity_eval.py`

Responsibilities:

- compute reference-output NLL/loss/perplexity for clean prompt/reference pairs
- avoid generation unless a secondary generated-output metric is explicitly
  enabled
- evaluate the same selected adapters/models as script `41`
- use one condition per subprocess
- capture OOM/failure counts
- report whether lower perplexity means better clean reference likelihood

Outputs:

- `outputs/clean_utility_perplexity_outputs.csv`
- `outputs/clean_utility_perplexity_summary.csv`
- `logs/clean_utility_perplexity_eval_<timestamp>.json`

### `scripts/43_real_asr_clean_utility_tradeoff.py`

Responsibilities:

- combine real-ASR or ASR-proxy results with clean utility NLL/perplexity
- include Wilson ASR intervals
- include clean utility intervals where feasible
- compare:
  - `base_model_only`
  - `original`
  - `uniform_gamma_0.25`
  - `uniform_gamma_0.50`
  - `top3_gamma_0.50`
  - `sensaware_top224_gamma_0.25`
- preserve clear metric labels
- produce report-ready candidate outputs only, not final replacement artifacts

Outputs:

- `outputs/real_asr_clean_utility_tradeoff_summary.csv`
- `reports/figures/report_ready/real_asr_clean_utility_tradeoff.png`
- `logs/real_asr_clean_utility_tradeoff_<timestamp>.json`

## 8. Safety Constraints

Required safety constraints for the whole extension:

- Do not execute official BackdoorLLM training code.
- Do not execute official BackdoorLLM attack-generation code.
- Do not print full harmful prompts.
- Do not print full harmful outputs.
- Do not store full trigger outputs unless absolutely required for verified
  scoring; prefer hashes, aggregate flags, and redacted previews.
- Do not update `final_submission_artifacts/` until results are reviewed.
- Do not overwrite existing final results.
- Do not update report-ready files automatically.
- Use deterministic generation.
- Keep one condition per subprocess.
- Catch OOM gracefully.
- Keep all model loading behind explicit numbered scripts.
- Record `is_official_asr`, `is_final_asr`, and `is_final_clean_utility`
  truthfully in outputs.

## 9. Interpretation Rules

Allowed conclusions depend strictly on results.

If real ASR and real utility support SensAware:

- Say SensAware provides a better ASR-utility trade-off than spectral-only.
- Say SensAware may beat uniform scaling only if real ASR, utility intervals,
  and preservation evidence support that statement.

If uniform remains best:

- Say uniform scaling remains strongest under real ASR.
- Inspect whether uniform scaling worsens clean utility/perplexity.
- Do not claim SensAware beats uniform.

If `base_model_only` matches uniform:

- Say uniform may behave like adapter erasure, not surgical sanitisation.
- Check whether clean utility/perplexity distinguishes uniform from base.
- If clean-reference perplexity is also nearly identical across base, original,
  uniform, and SensAware, say the chosen clean dataset did not reveal a strong
  benign adapter footprint. Do not turn that null result into a utility
  preservation claim.

If SensAware also loses utility:

- Say SensAware also erodes adapter behaviour.
- Treat the result as negative or mixed, not as a hidden win.

If the clean utility axis is weak for this adapter:

- Say the ASR-utility trade-off framing may be only partially applicable to this
  specific jailbreak LoRA.
- Prefer ASR-scorer alignment plus clean-output quality review over
  over-interpreting tiny perplexity differences.

If official scorer cannot be replicated:

- Do not call the results real ASR.
- Use labels such as "BackdoorLLM-inspired ASR proxy" or "ASR proxy" depending
  on what was verified.

If results contradict earlier claims:

- Update README, final verdict, report-ready tables, and final bundle honestly
  only after full review and consistency checks.

## 10. Decision Rule

This extension should replace or modify final results only if it is fully run,
reviewed, and consistency-checked.

If the extension is not completed:

- keep the current final submission package unchanged;
- use this architecture as future work.

If results are unfavorable:

- report them honestly;
- keep the mixed/negative conclusion if that is what the evidence supports.

If time runs short:

- do not rush scripts `39` to `43`;
- do not partially update final-facing claims;
- preserve the current submission-ready package.

## 11. Final Deliverable From This Planning Task

This planning task creates only:

- `REAL_ASR_AND_CLEAN_UTILITY_ARCHITECTURE.md`

After this document is created:

- update only `status.md` with a short summary;
- do not implement scripts yet;
- do not run experiments;
- do not run model loading;
- do not run inference;
- do not modify `final_submission_artifacts/`;
- do not modify report-ready result files.
