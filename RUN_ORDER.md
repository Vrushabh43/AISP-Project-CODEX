# Reproducible Run Order

This file lists the main execution order used by the project. Commands assume
the Linux evaluation server environment unless otherwise noted:

```bash
cd ~/solr-home/AISP-Project-CODEX
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
source .venv/bin/activate
```

Some early scripts are diagnostic or smoke-test only. They are preserved for
reproducibility but are not all required to regenerate the final result table.

## Phase 1: Environment Setup

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/00_check_env.py` | Basic environment/CUDA check. | `logs/env_check_*.json` | Diagnostic |
| `scripts/00_check_env_verbose.py` | Diagnose venv, pip, user-site leakage, package import paths. | `logs/env_check_verbose_*.json` | Diagnostic |
| `scripts/10_gpu_memory_diagnosis.py` | Check free VRAM and GPU processes before model attach. | `logs/gpu_memory_diagnosis_*.json` | Diagnostic |

## Phase 2: Hugging Face Cache and Adapter Inspection

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/01_check_hf_cache_and_adapter.py` | Verify/download cache metadata and BackdoorLLM adapter. | `logs/hf_cache_adapter_check_*.json` | Required setup |
| `scripts/01_inspect_adapter.py` | Inspect adapter config and LoRA A/B tensors. | `logs/adapter_inspection_*.json`, `outputs/adapter_tensor_summary.csv` | Required evidence |

## Phase 3: Spectral Analysis

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/02_extract_spectral_stats.py` | Compute compact singular values and spectral metrics for the backdoored adapter. | `outputs/spectral_stats.csv`, `reports/figures/singular_value_spectra_by_module.png` | Required |
| `scripts/03_clean_reference_search.py` | Search clean LoRA candidates. | `outputs/clean_reference_candidates.csv` | Diagnostic |
| `scripts/04_inspect_flagalpha_clean_adapter.py` | Safely inspect provisional FlagAlpha clean reference. | `outputs/flagalpha_adapter_tensor_summary.csv` | Diagnostic |
| `scripts/05_inspect_clean_adapter_candidates.py` | Compare clean reference candidates. | `outputs/clean_adapter_candidate_comparison.csv` | Diagnostic |
| `scripts/06_compare_clean_vs_backdoor_spectra.py` | Compare BackdoorLLM vs FlagAlpha spectra. | `outputs/clean_vs_backdoor_spectral_comparison.csv` | Required sanity check |

## Phase 4: Initial Sanitised Adapter Generation

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/07_generate_spectral_sanitised_adapters.py` | Generate spectral-only adapter variants. | `outputs/sanitised_adapters/top*_gamma_*/` | Required baseline generation |
| `scripts/08_smoke_check_sanitised_adapters.py` | Validate generated spectral-only adapter files. | `outputs/sanitised_adapter_smoke_check_summary.csv` | Required validation |
| `scripts/09_peft_loading_smoke_test.py` | Adapter-only and optional 4-bit base attach smoke tests. | `outputs/peft_loading_smoke_test_summary.csv` | Diagnostic/smoke |
| `scripts/11_tiny_inference_smoke_test.py` | Tiny generation smoke test for original and one variant. | `outputs/tiny_inference_smoke_test_summary.csv` | Diagnostic/smoke |

## Phase 5: Official Trigger Verification

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/14_find_backdoorllm_trigger_source.py` | Local search for BackdoorLLM trigger/source evidence. | `outputs/backdoorllm_trigger_source_candidates.csv` | Diagnostic |
| `scripts/15_fetch_and_inspect_backdoorllm_official_assets.py` | Fetch/inspect official BackdoorLLM assets without executing code. | `outputs/backdoorllm_official_trigger_verification.csv` | Required provenance |
| `scripts/16_create_official_badnets_prompt_files.py` | Create official BadNets prompt files from verified source. | `data/eval_prompts/official_badnets_jailbreak_*.jsonl` | Required |

## Phase 6: Bounded ASR and Utility Evaluation

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/12_small_baseline_evaluation.py` | Small pilot evaluation with placeholder trigger probes. | `outputs/small_baseline_evaluation*.csv` | Pilot only |
| `scripts/13_bounded_eval_from_prompt_files.py` | Bounded prompt-file evaluation framework. | `outputs/bounded_eval_*.csv` | Pilot/framework |
| `scripts/17_official_badnets_asr_pilot.py` | Small official BadNets ASR pilot. | `outputs/official_badnets_asr_pilot_*.csv` | Pilot |
| `scripts/18_generate_uniform_scaling_adapters.py` | Generate uniform-scaling baselines. | `outputs/sanitised_adapters/uniform_gamma_*/` | Required baseline generation |
| `scripts/19_official_badnets_full_bounded_eval.py` | Evaluate original, uniform, and spectral-only baselines on all 99 official trigger records. | `outputs/official_badnets_full_bounded_eval_*.csv` | Required baseline evidence |
| `scripts/20_clean_utility_and_tradeoff_eval.py` | Evaluate bounded clean utility and combine ASR-utility trade-off for baselines. | `outputs/asr_utility_tradeoff_summary.csv` | Required baseline evidence |

## Phase 7: SensAware Generation and Evaluation

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/21_clean_sensitivity_probe.py` | First bounded clean sensitivity probe over 50 components. | `outputs/clean_sensitivity_component_scores.csv` | Diagnostic; first attempt |
| `scripts/22_generate_sensitivity_aware_adapters.py` | Generate first small SensAware variants. | `outputs/sanitised_adapters/sensaware_top16_*`, `sensaware_top32_*` | Diagnostic; first attempt |
| `scripts/23_smoke_check_sensitivity_aware_adapters.py` | Smoke-check first SensAware variants. | `outputs/sensaware_adapter_smoke_check_summary.csv` | Diagnostic |
| `scripts/24_sensaware_official_bounded_eval.py` | Evaluate first SensAware variants. | `outputs/sensaware_asr_utility_tradeoff_summary.csv` | Required negative result |
| `scripts/25_sensaware_failure_analysis.py` | Diagnose why first SensAware variants underperformed. | `outputs/sensaware_failure_analysis_summary.csv` | Required explanation |

## Phase 8: Expanded SensAware Evaluation

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/26_clean_sensitivity_probe_expanded.py` | Score expanded 672-component candidate set. | `outputs/clean_sensitivity_component_scores_expanded.csv` | Required proposed method |
| `scripts/27_generate_sensitivity_aware_expanded_adapters.py` | Generate expanded SensAware variants. | `outputs/sanitised_adapters/sensaware_top128_*`, `sensaware_top224_*`, `sensaware_top336_*` | Required proposed method |
| `scripts/28_smoke_check_expanded_sensaware_adapters.py` | Smoke-check expanded SensAware variants. | `outputs/expanded_sensaware_adapter_smoke_check_summary.csv` | Required validation |
| `scripts/29_expanded_sensaware_official_bounded_eval.py` | Evaluate expanded SensAware vs baselines. | `outputs/expanded_sensaware_*` | Required final bounded result |

## Phase 9: Final Result Analysis and Plots

| Script | Purpose | Key outputs | Required for final reproduction? |
|---|---|---|---|
| `scripts/30_analyze_tradeoff_results.py` | Consolidate trade-off tables and identify best methods. | `outputs/consolidated_tradeoff_results.csv` | Required |
| `scripts/31_plot_tradeoff_results.py` | Generate broader trade-off plots. | `reports/figures/*.png` | Useful |
| `scripts/32_extract_eval_case_diagnostics.py` | Compare per-prompt outcomes for selected adapters without full prompt/output text. | `outputs/eval_case_diagnostics.csv` | Required diagnostics |
| `scripts/33_create_report_ready_results.py` | Create report-ready tables and key findings. | `outputs/report_ready_*` | Required final packaging |
| `scripts/34_create_report_ready_plots.py` | Create report-ready figures. | `reports/figures/report_ready/*.png` | Required final packaging |
| `scripts/35_final_submission_consistency_check.py` | Check final submission files and core numbers. | `outputs/final_submission_consistency_check_summary.csv` | Required final packaging check |

## Minimal Final Reproduction Path

If the environment and cached assets already exist, the minimal final result
path starts at official prompt creation and proceeds through scripts `18` to
`35`. For a clean machine, start from scripts `00`, `01`, and cache setup.

All reported final numbers are bounded heuristic metrics, not final judged ASR
or final judged clean utility.
