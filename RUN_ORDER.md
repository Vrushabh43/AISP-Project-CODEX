# Reproducible Run Order

This cleaned branch separates the final professor-facing pipeline from archived
development, smoke, pilot, and diagnostic scripts.

## Recommended One-Command Check

For final submission review, run:

```bash
python scripts/final_pipeline/00_run_final_submission.py --mode quick
```

This is file-only. It does not load models, run inference, execute official
BackdoorLLM code, or call external APIs.

Expected outputs:

- `logs/final_logs/final_submission_runner_quick_<timestamp>.json`
- `outputs/final_results/final_submission_runner_quick_summary.csv`
- `logs/final_logs/final_submission_consistency_check_<timestamp>.json`
- `outputs/final_results/final_submission_consistency_check_summary.csv`

Expected terminal result:

- `Overall: PASS`
- `Checks failed: 0`

## Runner Modes

| Mode | Command | Model loading? | Purpose |
|---|---|---|---|
| `quick` | `python scripts/final_pipeline/00_run_final_submission.py --mode quick` | No | Professor-facing final file/artifact consistency check. |
| `verify` | `python scripts/final_pipeline/00_run_final_submission.py --mode verify` | No | Regenerate final CSV/Markdown summaries from existing final CSV inputs, then run quick check. |
| `full` | `python scripts/final_pipeline/00_run_final_submission.py --mode full` | Yes | Re-run official rule-based ASR generation and clean perplexity evaluation on a configured GPU server. |

Use `quick` for normal grading/review. Use `verify` only if refreshing
file-only summaries. Use `full` only on the original-style GPU environment with
the Hugging Face cache available.

## Final Pipeline Scripts

The retained scripts are in `scripts/final_pipeline/`:

- `00_run_final_submission.py`
- `02_extract_spectral_stats.py`
- `07_generate_spectral_sanitised_adapters.py`
- `18_generate_uniform_scaling_adapters.py`
- `26_clean_sensitivity_probe_expanded.py`
- `27_generate_sensitivity_aware_expanded_adapters.py`
- `35_final_submission_consistency_check.py`
- `39c_verify_rule_based_jailbreak_asr.py`
- `40_create_real_asr_and_clean_utility_prompt_files.py`
- `42_clean_utility_perplexity_eval.py`
- `45_official_rule_based_asr_eval.py`
- `46_official_rule_based_asr_clean_tradeoff.py`
- `47_official_rule_based_asr_sanity_checks.py`

Development, smoke, pilot, and older diagnostic scripts are preserved under:

- `archive/final_minimal_cleanup_<timestamp>/dev_smoke_scripts/`

## Final Outputs

Primary final result files live in:

- `outputs/final_results/`

Important files:

- `outputs/final_results/official_rule_based_asr_clean_tradeoff_summary.csv`
- `outputs/final_results/official_rule_based_asr_eval_summary.csv`
- `outputs/final_results/official_rule_based_asr_sanity_checks_summary.csv`
- `outputs/final_results/official_rule_based_asr_verification_summary.csv`
- `outputs/final_results/clean_utility_perplexity_summary.csv`
- `outputs/final_results/clean_behaviour_similarity_summary.csv`
- `outputs/final_results/report_ready_main_results.md`
- `outputs/final_results/report_ready_key_findings.md`
- `outputs/final_results/report_ready_case_diagnostics_summary.md`
- `outputs/final_results/final_submission_consistency_check_summary.csv`

Final logs live in:

- `logs/final_logs/`

Final figures:

- `reports/figures/report_ready/official_rule_based_asr_clean_tradeoff.png`
- `reports/figures/report_ready/report_tradeoff_scatter_key_methods.png`
- `reports/figures/report_ready/report_trigger_rate_key_methods.png`
- `reports/figures/report_ready/report_trigger_reduction_key_methods.png`
- `reports/figures/report_ready/report_clean_utility_key_methods.png`

Copied professor-facing bundle:

- `final_submission_artifacts/`

## Final Scientific Result

The final ASR evidence uses the verified BackdoorLLM official rule-based
jailbreak ASR scorer. The scorer is applied to project-local deterministic
`[INST]` generations and is not external-judged ASR.

Focused official rule-based ASR results over 99 trigger prompts:

- `original`: 36/99
- `uniform_gamma_0.25`: 13/99
- `uniform_gamma_0.50`: 7/99
- `top3_gamma_0.50`: 7/99
- `sensaware_top224_gamma_0.25`: 1/99

`sensaware_top224_gamma_0.25` is the best focused condition in this run:

- ASR: 1/99 = 0.010101
- Wilson 95% CI: [0.001785, 0.055017]
- Clean-reference perplexity: 3.557692

Clean-reference perplexity is a reference-likelihood probe, not final human
clean utility.

## Full-Mode Environment

Full mode requires:

- Python environment with project dependencies.
- `PYTHONNOUSERSITE=1`.
- `HF_HUB_CACHE` pointing to a cache containing:
  - `models--NousResearch--Llama-2-7b-chat-hf`
  - `models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets`
- GPU memory sufficient for 4-bit Llama-2-7B-Chat evaluation.

Typical Linux server prefix:

```bash
cd ~/solr-home/AISP-Project-CODEX
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
source .venv/bin/activate
```

## Archive

Archived material is preserved, not deleted:

- `archive/submission_cleanup_20260523T170729/`
- `archive/final_cleanup_20260530T113656/`
- `archive/final_minimal_cleanup_<timestamp>/`

Each cleanup archive includes an `ARCHIVE_MANIFEST.md` and
`archive_manifest.csv`.
