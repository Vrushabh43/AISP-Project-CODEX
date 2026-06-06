AISP-Project-CODEX/
├── archive/
│   ├── final_cleanup_20260530T113656/
│   │   ├── duplicate_or_backup/
│   │   │   ├── data/
│   │   │   │   └── eval_prompts/
│   │   │   │       ├── clean_utility_reference_eval.bak_20260524T151304Z.jsonl
│   │   │   │       └── real_asr_official_badnets.bak_20260524T151304Z.jsonl
│   │   │   ├── outputs/
│   │   │   │   ├── clean_utility_perplexity_outputs.bak_20260524T151304Z.csv
│   │   │   │   ├── clean_utility_perplexity_summary.bak_20260524T151304Z.csv
│   │   │   │   ├── official_rule_based_asr_sanity_checks_summary.bak_20260530T073520Z.csv
│   │   │   │   ├── real_asr_clean_utility_tradeoff_summary.bak_20260524T151630Z.csv
│   │   │   │   └── real_eval_prompt_file_summary.bak_20260524T151304Z.csv
│   │   │   └── OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.bak_20260530T073520Z.md
│   │   ├── superseded/
│   │   │   ├── BASE_CONTROL_AND_CLEAN_BEHAVIOUR_PRESERVATION_PLAN.md
│   │   │   ├── lora_sanitisation_master_project.md
│   │   │   ├── OFFICIAL_RULE_BASED_ASR_ARCHITECTURE.md
│   │   │   ├── OFFICIAL_SCORER_AUDIT.md
│   │   │   └── REAL_ASR_AND_CLEAN_UTILITY_ARCHITECTURE.md
│   │   ├── superseded_project_memory/
│   │   │   └── AGENT.md
│   │   ├── unused_src_stubs/
│   │   │   ├── baselines.py
│   │   │   ├── config.py
│   │   │   ├── delta_w.py
│   │   │   ├── eval_asr.py
│   │   │   ├── eval_clean.py
│   │   │   ├── logging_utils.py
│   │   │   ├── plots.py
│   │   │   └── sensitivity.py
│   │   ├── archive_manifest.csv
│   │   └── ARCHIVE_MANIFEST.md
│   └── submission_cleanup_20260523T170729/
│       ├── configs/
│       │   └── eval_small.yaml
│       ├── data/
│       │   └── eval_prompts/
│       │       └── trigger_probe_small_unverified.jsonl
│       ├── outputs/
│       │   ├── sanitised_adapters/
│       │   │   ├── sensaware_top16_gamma_0.50.bak_20260520T165056Z/
│       │   │   │   ├── adapter_config.json
│       │   │   │   ├── adapter_model.safetensors
│       │   │   │   └── sanitisation_report.json
│       │   │   ├── sensaware_top32_gamma_0.25.bak_20260520T165056Z/
│       │   │   │   ├── adapter_config.json
│       │   │   │   ├── adapter_model.safetensors
│       │   │   │   └── sanitisation_report.json
│       │   │   └── sensaware_top32_gamma_0.50.bak_20260520T165056Z/
│       │   │       ├── adapter_config.json
│       │   │       ├── adapter_model.safetensors
│       │   │       └── sanitisation_report.json
│       │   ├── backdoorllm_official_trigger_verification.bak_20260519T210839Z.csv
│       │   ├── backdoorllm_official_trigger_verification.bak_20260520T122700Z.csv
│       │   ├── backdoorllm_trigger_source_candidates.bak_20260519T205255Z.csv
│       │   ├── clean_sensitivity_component_scores.bak_20260520T164955Z.csv
│       │   ├── peft_loading_smoke_test_summary.bak_20260517T192009Z.csv
│       │   ├── peft_loading_smoke_test_summary.bak_20260517T192025Z.csv
│       │   ├── peft_loading_smoke_test_summary.bak_20260519T165423Z.csv
│       │   ├── peft_loading_smoke_test_summary.bak_20260519T165754Z.csv
│       │   ├── report_ready_case_diagnostics_summary.bak_20260523T121759Z.md
│       │   ├── report_ready_case_diagnostics_summary.bak_20260523T122722Z.md
│       │   ├── report_ready_key_findings.bak_20260523T121759Z.md
│       │   ├── report_ready_key_findings.bak_20260523T122722Z.md
│       │   ├── report_ready_main_results.bak_20260523T121759Z.csv
│       │   ├── report_ready_main_results.bak_20260523T121759Z.md
│       │   ├── report_ready_main_results.bak_20260523T122722Z.csv
│       │   ├── report_ready_main_results.bak_20260523T122722Z.md
│       │   ├── sensitivity_aware_adapter_generation_summary.bak_20260520T165056Z.csv
│       │   ├── sensitivity_candidate_components.bak_20260520T164955Z.csv
│       │   ├── tiny_inference_smoke_test_summary.bak_20260519T193244Z.csv
│       │   ├── tiny_inference_smoke_test_summary.bak_20260519T193742Z.csv
│       │   ├── tiny_inference_smoke_test_summary.bak_20260519T193953Z.csv
│       │   └── tiny_inference_smoke_test_summary.bak_20260519T194531Z.csv
│       ├── reports/
│       │   ├── figures/
│       │   │   └── report_ready/
│       │   │       ├── report_clean_utility_key_methods.bak_20260523T121804Z.png
│       │   │       ├── report_clean_utility_key_methods.bak_20260523T122722Z.png
│       │   │       ├── report_tradeoff_scatter_key_methods.bak_20260523T121804Z.png
│       │   │       ├── report_tradeoff_scatter_key_methods.bak_20260523T122722Z.png
│       │   │       ├── report_trigger_rate_key_methods.bak_20260523T121804Z.png
│       │   │       ├── report_trigger_rate_key_methods.bak_20260523T122722Z.png
│       │   │       ├── report_trigger_reduction_key_methods.bak_20260523T121804Z.png
│       │   │       └── report_trigger_reduction_key_methods.bak_20260523T122722Z.png
│       │   └── experiment_summary_template.md
│       ├── scripts/
│       ├── src/
│       │   └── lora_sanitise/
│       └── ARCHIVE_MANIFEST.md
├── configs/
│   └── experiment.yaml
├── data/
│   ├── eval_prompts/
│   │   ├── clean_utility_medium.jsonl
│   │   ├── clean_utility_reference_eval.jsonl
│   │   ├── clean_utility_small.jsonl
│   │   ├── official_badnets_jailbreak_full.jsonl
│   │   ├── official_badnets_jailbreak_small.jsonl
│   │   ├── README.md
│   │   └── real_asr_official_badnets.jsonl
│   └── README.md
├── external_sources/
│   └── backdoorllm_official/
│       ├── attack/
│       │   └── DPA/
│       │       └── examples/
│       │           └── llama2-7b-chat/
│       │               └── jailbreak/
│       │                   └── badnet/
│       │                       ├── adapter_config.json
│       │                       ├── README.md
│       │                       ├── special_tokens_map.json
│       │                       ├── tokenizer.json
│       │                       ├── tokenizer_config.json
│       │                       └── trainer_state.json
│       └── README.md
├── final_submission_artifacts/
│   ├── figures/
│   │   ├── official_rule_based_asr_clean_tradeoff.png
│   │   ├── report_clean_utility_key_methods.png
│   │   ├── report_tradeoff_scatter_key_methods.png
│   │   ├── report_trigger_rate_key_methods.png
│   │   └── report_trigger_reduction_key_methods.png
│   ├── results/
│   │   ├── clean_utility_perplexity_summary.csv
│   │   ├── consolidated_tradeoff_results.csv
│   │   ├── eval_case_diagnostics.csv
│   │   ├── expanded_sensaware_asr_utility_tradeoff_summary.csv
│   │   ├── official_rule_based_asr_clean_tradeoff_summary.csv
│   │   ├── official_rule_based_asr_eval_summary.csv
│   │   ├── official_rule_based_asr_sanity_checks_summary.csv
│   │   ├── official_rule_based_asr_verification_summary.csv
│   │   ├── report_ready_case_diagnostics_summary.md
│   │   ├── report_ready_key_findings.md
│   │   └── report_ready_main_results.md
│   ├── FINAL_CLEANUP_SUMMARY.md
│   ├── FINAL_SUBMISSION_VERDICT.md
│   ├── IMPLEMENTATION_NOTES.md
│   ├── OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md
│   ├── OFFICIAL_RULE_BASED_ASR_VERIFICATION.md
│   ├── README.md
│   ├── RUN_ORDER.md
│   └── SUBMISSION_STRUCTURE.md
├── logs/
│   ├── base_model_control_eval_children_20260523T170204Z/
│   │   ├── base_model_only.json
│   │   ├── original.json
│   │   ├── sensaware_top224_gamma_0_25.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── bounded_eval_children_20260519T203137Z/
│   │   ├── original.json
│   │   ├── top1_gamma_0_50.json
│   │   └── top3_gamma_0_50.json
│   ├── clean_utility_eval_children_20260520T152445Z/
│   │   ├── original.json
│   │   ├── top1_gamma_0_50.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── clean_utility_perplexity_children_20260524T150011Z/
│   │   ├── base_model_only.json
│   │   ├── original.json
│   │   ├── sensaware_top224_gamma_0_25.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── clean_utility_perplexity_children_20260524T151304Z/
│   │   ├── base_model_only.json
│   │   ├── original.json
│   │   ├── sensaware_top224_gamma_0_25.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── expanded_sensaware_official_bounded_eval_children_20260521T142706Z/
│   │   ├── original.json
│   │   ├── sensaware_top128_gamma_0_25.json
│   │   ├── sensaware_top128_gamma_0_50.json
│   │   ├── sensaware_top224_gamma_0_25.json
│   │   ├── sensaware_top224_gamma_0_50.json
│   │   ├── sensaware_top336_gamma_0_50.json
│   │   ├── top1_gamma_0_50.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── official_badnets_asr_pilot_children_20260520T131535Z/
│   │   └── original.json
│   ├── official_badnets_asr_pilot_children_20260520T132046Z/
│   │   ├── original.json
│   │   ├── top1_gamma_0_50.json
│   │   └── top3_gamma_0_50.json
│   ├── official_badnets_full_bounded_eval_children_20260520T140042Z/
│   │   ├── original.json
│   │   ├── top1_gamma_0_50.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── official_rule_based_asr_eval_children_20260525T120958Z/
│   │   ├── base_model_only.json
│   │   ├── original.json
│   │   ├── sensaware_top224_gamma_0_25.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── sensaware_official_bounded_eval_children_20260520T170614Z/
│   │   ├── original.json
│   │   ├── sensaware_top16_gamma_0_50.json
│   │   ├── sensaware_top32_gamma_0_25.json
│   │   ├── sensaware_top32_gamma_0_50.json
│   │   ├── top1_gamma_0_50.json
│   │   ├── top3_gamma_0_50.json
│   │   ├── uniform_gamma_0_25.json
│   │   └── uniform_gamma_0_50.json
│   ├── small_baseline_children_20260519T195636Z/
│   │   ├── original.json
│   │   ├── top1_gamma_0_50.json
│   │   └── top3_gamma_0_50.json
│   ├── tiny_inference_children_20260519T193244Z/
│   │   ├── original.json
│   │   └── top1_gamma_0_50.json
│   ├── tiny_inference_children_20260519T193742Z/
│   │   ├── original.json
│   │   └── top1_gamma_0_50.json
│   ├── tiny_inference_children_20260519T193953Z/
│   │   ├── original.json
│   │   └── top1_gamma_0_50.json
│   ├── tiny_inference_children_20260519T194531Z/
│   │   ├── original.json
│   │   └── top1_gamma_0_50.json
│   ├── adapter_inspection_20260517T143951Z.json
│   ├── backdoorllm_official_asset_inspection_20260519T210331Z.json
│   ├── backdoorllm_official_asset_inspection_20260519T210839Z.json
│   ├── backdoorllm_official_asset_inspection_20260520T122700Z.json
│   ├── backdoorllm_trigger_source_search_20260519T204914Z.json
│   ├── backdoorllm_trigger_source_search_20260519T205255Z.json
│   ├── base_control_preflight_check_20260523T164258Z.json
│   ├── base_model_control_eval_20260523T170204Z.json
│   ├── bounded_eval_from_prompt_files_20260519T203137Z.json
│   ├── clean_adapter_candidate_inspection_20260517T154855Z.json
│   ├── clean_behaviour_similarity_analysis_20260523T182958Z.json
│   ├── clean_reference_search_20260517T151217Z.json
│   ├── clean_sensitivity_probe_20260520T162149Z.json
│   ├── clean_sensitivity_probe_20260520T164955Z.json
│   ├── clean_sensitivity_probe_expanded_20260521T140843Z.json
│   ├── clean_utility_and_tradeoff_eval_20260520T152445Z.json
│   ├── clean_utility_perplexity_eval_20260524T150011Z.json
│   ├── clean_utility_perplexity_eval_20260524T151304Z.json
│   ├── clean_vs_backdoor_spectral_comparison_20260517T160214Z.json
│   ├── env_check_20260517T131640Z.json
│   ├── env_check_20260517T132409Z.json
│   ├── env_check_verbose_20260517T135810Z.json
│   ├── env_check_verbose_20260517T141803Z.json
│   ├── env_check_verbose_20260517T142024Z.json
│   ├── env_check_verbose_20260517T142304Z.json
│   ├── eval_case_diagnostics_20260523T111940Z.json
│   ├── expanded_sensaware_adapter_smoke_check_20260521T142658Z.json
│   ├── expanded_sensaware_official_bounded_eval_20260521T142706Z.json
│   ├── final_submission_consistency_check_20260523T150806Z.json
│   ├── final_submission_consistency_check_20260523T150930Z.json
│   ├── final_submission_consistency_check_20260523T151233Z.json
│   ├── final_submission_consistency_check_20260523T153016Z.json
│   ├── final_submission_consistency_check_20260523T153200Z.json
│   ├── final_submission_consistency_check_20260530T094435Z.json
│   ├── final_submission_consistency_check_20260530T094459Z.json
│   ├── final_submission_runner_quick_20260523T153016Z.json
│   ├── final_submission_runner_quick_20260523T153200Z.json
│   ├── flagalpha_clean_adapter_inspection_20260517T152925Z.json
│   ├── gpu_memory_diagnosis_20260517T192845Z.json
│   ├── gpu_memory_diagnosis_20260519T165210Z.json
│   ├── hf_cache_adapter_check_20260517T141439Z.json
│   ├── hf_cache_adapter_check_20260517T141541Z.json
│   ├── hf_cache_adapter_check_20260517T142343Z.json
│   ├── hf_cache_adapter_check_20260519T193141Z.json
│   ├── official_asr_clean_utility_discovery_20260524T113047Z.json
│   ├── official_asr_clean_utility_discovery_20260524T113332Z.json
│   ├── official_asr_clean_utility_discovery_20260524T121057Z.json
│   ├── official_asr_clean_utility_discovery_20260524T121318Z.json
│   ├── official_asr_scorer_audit_20260524T122607Z.json
│   ├── official_asr_scorer_audit_20260524T123020Z.json
│   ├── official_badnets_asr_pilot_20260520T132046Z.json
│   ├── official_badnets_full_bounded_eval_20260520T140042Z.json
│   ├── official_badnets_prompt_extraction_20260520T131505Z.json
│   ├── official_rule_based_asr_clean_tradeoff_20260525T142217Z.json
│   ├── official_rule_based_asr_eval_20260525T120958Z.json
│   ├── official_rule_based_asr_rescore_existing_20260525T120801Z.json
│   ├── official_rule_based_asr_sanity_checks_20260526T161737Z.json
│   ├── official_rule_based_asr_sanity_checks_20260530T073520Z.json
│   ├── official_rule_based_asr_verification_20260525T114142Z.json
│   ├── official_rule_based_asr_verification_20260525T114645Z.json
│   ├── official_rule_based_asr_verification_20260525T114859Z.json
│   ├── peft_loading_smoke_test_20260517T191954Z.json
│   ├── peft_loading_smoke_test_20260517T192009Z.json
│   ├── peft_loading_smoke_test_20260517T192025Z.json
│   ├── peft_loading_smoke_test_20260519T165423Z.json
│   ├── peft_loading_smoke_test_20260519T165754Z.json
│   ├── README.md
│   ├── real_asr_clean_utility_tradeoff_20260524T150116Z.json
│   ├── real_asr_clean_utility_tradeoff_20260524T151630Z.json
│   ├── real_eval_prompt_file_creation_20260524T145930Z.json
│   ├── real_eval_prompt_file_creation_20260524T151304Z.json
│   ├── report_ready_plots_20260523T122722Z.json
│   ├── report_ready_results_20260523T122722Z.json
│   ├── sanitised_adapter_generation_20260517T163541Z.json
│   ├── sanitised_adapter_smoke_check_20260517T190339Z.json
│   ├── sensaware_adapter_smoke_check_20260520T170607Z.json
│   ├── sensaware_failure_analysis_20260521T140822Z.json
│   ├── sensaware_official_bounded_eval_20260520T170614Z.json
│   ├── sensitivity_aware_adapter_generation_20260520T162258Z.json
│   ├── sensitivity_aware_adapter_generation_20260520T165056Z.json
│   ├── sensitivity_aware_expanded_adapter_generation_20260521T141058Z.json
│   ├── small_baseline_evaluation_20260519T195636Z.json
│   ├── spectral_stats_20260517T145251Z.json
│   ├── tiny_inference_smoke_test_20260519T170722Z.json
│   ├── tiny_inference_smoke_test_20260519T193244Z.json
│   ├── tiny_inference_smoke_test_20260519T193742Z.json
│   ├── tiny_inference_smoke_test_20260519T193953Z.json
│   ├── tiny_inference_smoke_test_20260519T194531Z.json
│   ├── tradeoff_analysis_20260523T111939Z.json
│   ├── tradeoff_plots_20260523T111939Z.json
│   ├── uniform_scaling_adapter_generation_20260520T140024Z.json
│   └── wilson_ci_and_final_comparison_20260523T183002Z.json
├── models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/
│   ├── blobs/
│   │   ├── 0a9c45e4643c0593b9050a0581d840f3f438fb7acb3a7e9b72515eb5b2b6f85a
│   │   ├── 332b91726f5f9c2173356e52750d363cde910e5c
│   │   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   │   └── bd78858e4903e2c2d3b71f53de73f06965f10cff
│   ├── refs/
│   │   └── main
│   └── snapshots/
│       └── 408295cd17df70e5164e7692e2aa3c5b9e2e4f3b/
│           ├── .gitattributes
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           └── README.md
├── outputs/
│   ├── sanitised_adapters/
│   │   ├── sensaware_top128_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top128_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top16_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top224_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top224_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top32_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top32_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── sensaware_top336_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top1_gamma_0.0/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top1_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top1_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top3_gamma_0.0/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top3_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── top3_gamma_0.50/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   ├── uniform_gamma_0.25/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── sanitisation_report.json
│   │   └── uniform_gamma_0.50/
│   │       ├── adapter_config.json
│   │       ├── adapter_model.safetensors
│   │       └── sanitisation_report.json
│   ├── adapter_tensor_summary.csv
│   ├── asr_utility_tradeoff_summary.csv
│   ├── backdoorllm_official_trigger_verification.csv
│   ├── backdoorllm_trigger_source_candidates.csv
│   ├── base_control_preflight_check.csv
│   ├── base_model_control_eval_outputs.csv
│   ├── base_model_control_eval_summary.csv
│   ├── bounded_eval_outputs.csv
│   ├── bounded_eval_summary.csv
│   ├── clean_adapter_candidate_comparison.csv
│   ├── clean_behaviour_similarity_summary.csv
│   ├── clean_candidate_FlagAlpha_Llama2_Chinese_7b_Chat_LoRA_tensor_summary.csv
│   ├── clean_candidate_Luciano_lora_4bit_Llama_2_7b_chat_hf_lener_br_tensor_summary.csv
│   ├── clean_candidate_manojpatil_llama_2_7b_chat_lora_adaptor_tensor_summary.csv
│   ├── clean_reference_candidates.csv
│   ├── clean_sensitivity_component_scores.csv
│   ├── clean_sensitivity_component_scores_expanded.csv
│   ├── clean_utility_dataset_audit_summary.csv
│   ├── clean_utility_eval_outputs.csv
│   ├── clean_utility_eval_summary.csv
│   ├── clean_utility_perplexity_outputs.csv
│   ├── clean_utility_perplexity_summary.csv
│   ├── clean_vs_backdoor_spectral_comparison.csv
│   ├── consolidated_tradeoff_results.csv
│   ├── eval_case_diagnostics.csv
│   ├── expanded_sensaware_adapter_smoke_check_summary.csv
│   ├── expanded_sensaware_asr_utility_tradeoff_summary.csv
│   ├── expanded_sensaware_official_bounded_eval_outputs.csv
│   ├── expanded_sensaware_official_bounded_eval_summary.csv
│   ├── final_cleanup_inventory.csv
│   ├── final_cleanup_plan.csv
│   ├── final_submission_consistency_check_summary.csv
│   ├── final_submission_runner_quick_summary.csv
│   ├── flagalpha_adapter_tensor_summary.csv
│   ├── official_asr_clean_utility_discovery_summary.csv
│   ├── official_asr_scorer_audit_summary.csv
│   ├── official_badnets_asr_pilot_outputs.csv
│   ├── official_badnets_asr_pilot_summary.csv
│   ├── official_badnets_full_bounded_eval_outputs.csv
│   ├── official_badnets_full_bounded_eval_summary.csv
│   ├── official_badnets_prompt_file_summary.csv
│   ├── official_rule_based_asr_clean_tradeoff_summary.csv
│   ├── official_rule_based_asr_eval_outputs.csv
│   ├── official_rule_based_asr_eval_summary.csv
│   ├── official_rule_based_asr_rescore_existing_summary.csv
│   ├── official_rule_based_asr_sanity_checks_summary.csv
│   ├── official_rule_based_asr_verification_summary.csv
│   ├── peft_loading_smoke_test_summary.csv
│   ├── README.md
│   ├── real_asr_clean_utility_tradeoff_summary.csv
│   ├── real_eval_prompt_file_summary.csv
│   ├── report_ready_case_diagnostics_summary.md
│   ├── report_ready_key_findings.md
│   ├── report_ready_main_results.csv
│   ├── report_ready_main_results.md
│   ├── sanitised_adapter_generation_summary.csv
│   ├── sanitised_adapter_smoke_check_summary.csv
│   ├── sensaware_adapter_smoke_check_summary.csv
│   ├── sensaware_asr_utility_tradeoff_summary.csv
│   ├── sensaware_failure_analysis_summary.csv
│   ├── sensaware_official_bounded_eval_outputs.csv
│   ├── sensaware_official_bounded_eval_summary.csv
│   ├── sensitivity_aware_adapter_generation_summary.csv
│   ├── sensitivity_aware_expanded_adapter_generation_summary.csv
│   ├── sensitivity_candidate_components.csv
│   ├── sensitivity_candidate_components_expanded.csv
│   ├── small_baseline_evaluation.csv
│   ├── small_baseline_evaluation_summary.csv
│   ├── spectral_stats.csv
│   ├── src_cleanup_audit.csv
│   ├── tiny_inference_smoke_test_summary.csv
│   ├── uniform_scaling_adapter_generation_summary.csv
│   └── wilson_ci_tradeoff_summary.csv
├── reports/
│   ├── figures/
│   │   ├── report_ready/
│   │   │   ├── official_rule_based_asr_clean_tradeoff.png
│   │   │   ├── real_asr_clean_utility_tradeoff.png
│   │   │   ├── report_clean_utility_key_methods.png
│   │   │   ├── report_tradeoff_scatter_key_methods.png
│   │   │   ├── report_trigger_rate_key_methods.png
│   │   │   └── report_trigger_reduction_key_methods.png
│   │   ├── asr_utility_tradeoff_scatter.png
│   │   ├── clean_utility_bar_by_method.png
│   │   ├── clean_vs_backdoor_entropy_by_layer.png
│   │   ├── clean_vs_backdoor_singular_curve_mean.png
│   │   ├── clean_vs_backdoor_top1_by_layer.png
│   │   ├── clean_vs_backdoor_top3_by_layer.png
│   │   ├── singular_value_spectra_by_module.png
│   │   ├── trigger_rate_bar_by_method.png
│   │   └── trigger_reduction_vs_clean_delta.png
│   ├── tables/
│   ├── experiment_journal.md
│   └── known_issues.md
├── scripts/
│   ├── 00_check_env.py
│   ├── 00_check_env_verbose.py
│   ├── 00_run_final_submission.py
│   ├── 01_check_hf_cache_and_adapter.py
│   ├── 01_inspect_adapter.py
│   ├── 02_extract_spectral_stats.py
│   ├── 03_clean_reference_search.py
│   ├── 04_inspect_flagalpha_clean_adapter.py
│   ├── 05_inspect_clean_adapter_candidates.py
│   ├── 06_compare_clean_vs_backdoor_spectra.py
│   ├── 07_generate_spectral_sanitised_adapters.py
│   ├── 08_smoke_check_sanitised_adapters.py
│   ├── 09_peft_loading_smoke_test.py
│   ├── 10_gpu_memory_diagnosis.py
│   ├── 11_tiny_inference_smoke_test.py
│   ├── 12_small_baseline_evaluation.py
│   ├── 13_bounded_eval_from_prompt_files.py
│   ├── 14_find_backdoorllm_trigger_source.py
│   ├── 15_fetch_and_inspect_backdoorllm_official_assets.py
│   ├── 16_create_official_badnets_prompt_files.py
│   ├── 17_official_badnets_asr_pilot.py
│   ├── 18_generate_uniform_scaling_adapters.py
│   ├── 19_official_badnets_full_bounded_eval.py
│   ├── 20_clean_utility_and_tradeoff_eval.py
│   ├── 21_clean_sensitivity_probe.py
│   ├── 22_generate_sensitivity_aware_adapters.py
│   ├── 23_smoke_check_sensitivity_aware_adapters.py
│   ├── 24_sensaware_official_bounded_eval.py
│   ├── 25_sensaware_failure_analysis.py
│   ├── 26_clean_sensitivity_probe_expanded.py
│   ├── 27_generate_sensitivity_aware_expanded_adapters.py
│   ├── 28_smoke_check_expanded_sensaware_adapters.py
│   ├── 29_expanded_sensaware_official_bounded_eval.py
│   ├── 30_analyze_tradeoff_results.py
│   ├── 31_plot_tradeoff_results.py
│   ├── 32_extract_eval_case_diagnostics.py
│   ├── 33_create_report_ready_results.py
│   ├── 34_create_report_ready_plots.py
│   ├── 35_final_submission_consistency_check.py
│   ├── 36_base_model_control_eval.py
│   ├── 37_clean_behaviour_similarity_analysis.py
│   ├── 38_wilson_ci_and_final_comparison.py
│   ├── 39_discover_official_asr_and_clean_utility_sources.py
│   ├── 39b_audit_official_asr_scorer.py
│   ├── 39c_verify_rule_based_jailbreak_asr.py
│   ├── 40_create_real_asr_and_clean_utility_prompt_files.py
│   ├── 42_clean_utility_perplexity_eval.py
│   ├── 43_real_asr_clean_utility_tradeoff.py
│   ├── 44_official_rule_based_asr_rescore_existing_outputs.py
│   ├── 45_official_rule_based_asr_eval.py
│   ├── 46_official_rule_based_asr_clean_tradeoff.py
│   └── 47_official_rule_based_asr_sanity_checks.py
├── src/
│   └── lora_sanitise/
│       ├── __init__.py
│       ├── attenuation.py
│       ├── lora_io.py
│       └── svd_tools.py
├── .gitignore
├── FILE_STRUCTURE.md
├── FINAL_CLEANUP_SUMMARY.md
├── FINAL_SUBMISSION_VERDICT.md
├── IMPLEMENTATION_NOTES.md
├── OFFICIAL_RULE_BASED_ASR_SANITY_CHECKS.md
├── OFFICIAL_RULE_BASED_ASR_VERIFICATION.md
├── README.md
├── requirements.txt
├── RUN_ORDER.md
├── status.md
├── SUBMISSION_STRUCTURE.md
└── task.txt

Notes:

* Generated output files are stored in outputs/.
* Logs are stored in logs/.
* Report figures and tables are stored in reports/.
