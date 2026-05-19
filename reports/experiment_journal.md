# Experiment Journal

Report-friendly project journal for:

Post-Hoc Sanitisation of Backdoored LoRA Adapters via Sensitivity-Aware Singular-Component Attenuation.

This journal summarizes completed milestones. It is meant to help future report writing, but it is not a final-results section. Do not convert pilot or smoke-test findings into research claims.

## Current Caveats

- No final ASR evaluation has been run.
- No final clean utility evaluation has been run.
- Trigger-pilot prompts used so far are unverified placeholders, not official BackdoorLLM trigger prompts.
- FlagAlpha is only a structural clean reference, not a task/distribution-matched clean LoRA.
- The small baseline evaluation was a pilot to verify plumbing, not the final experiment.

## 1. Environment Setup

Goal:

Establish a reproducible Python environment for adapter inspection, CPU spectral analysis, and later low-memory Llama-2/PEFT loading.

Command/script used:

```bash
python scripts/00_check_env.py
python scripts/00_check_env_verbose.py
```

Key result:

- The active project environment was fixed on the server by using the project `.venv`.
- Required packages eventually imported from `.venv`:
  - torch `2.7.1+cu126`
  - transformers `4.53.3`
  - peft `0.15.2`
  - accelerate `1.1.1`
  - bitsandbytes `0.46.1`
  - huggingface_hub `0.33.5`
- CUDA was available on the RTX 2080 SUPER.
- The server requires:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
```

Problems faced:

- User-site package leakage put `/home/43e3/.local/lib/python3.12/site-packages` ahead of the venv.
- This caused mismatched packages and import failures for `transformers` and `peft`.
- The same command could behave differently depending on active Python and `PYTHONPATH`.

How problem was fixed:

- Cleared `PYTHONPATH`.
- Set `PYTHONNOUSERSITE=1`.
- Installed missing venv dependencies with `python -m pip`.
- Verified module paths with `scripts/00_check_env_verbose.py`.

Output files generated:

- `logs/env_check_verbose_20260517T142304Z.json`
- Earlier diagnostic logs under `logs/env_check*.json`

What this means for the research:

- The environment can support CPU adapter analysis and low-memory PEFT loading when shell variables are set correctly.

What not to claim yet:

- Do not claim model evaluation reproducibility across machines solely from this setup. Cache paths and GPU state still matter.

## 2. Hugging Face Cache And Adapter Availability

Goal:

Verify that the base model, dataset cache, and backdoored adapter are reachable/cached without loading Llama-2.

Command/script used:

```bash
python scripts/01_check_hf_cache_and_adapter.py
python scripts/01_check_hf_cache_and_adapter.py --download-adapter
```

Key result:

- Base model cache was found on `ki-016`:
  `NousResearch/Llama-2-7b-chat-hf`
- Alpaca cache was found on `ki-016`:
  `tatsu-lab/alpaca`
- Backdoored adapter was downloaded and cached:
  `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
- Adapter snapshot:
  `/home/huggingface/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b`

Problems faced:

- The adapter was not initially cached.
- Later, when moving to `ki-010`, copied Hugging Face cache files were corrupted because snapshot symlink structure did not survive cleanly.

How problem was fixed:

- Downloaded only the adapter snapshot first.
- On `ki-010`, created a fresh cache at `/home/43e3/hf-cache-aisp` and downloaded the base model and adapter there.

Output files generated:

- `logs/hf_cache_adapter_check_20260517T142343Z.json`
- `logs/hf_cache_adapter_check_20260519T193141Z.json`

What this means for the research:

- The exact backdoored adapter is available and can be inspected without modifying third-party source code.

What not to claim yet:

- Cache availability is not model correctness. A copied cache with 0-byte snapshot files can still appear superficially present but fail at load time.

## 3. Adapter Inspection

Goal:

Inspect the BackdoorLLM BadNets LoRA adapter files without loading the base model.

Command/script used:

```bash
python scripts/01_inspect_adapter.py
```

Key result:

- Adapter type: `LORA`
- Task type: `CAUSAL_LM`
- Rank `r`: `8`
- LoRA alpha: `16`
- Tensor dtype: F16
- Target modules:
  - `down_proj`
  - `gate_proj`
  - `k_proj`
  - `o_proj`
  - `q_proj`
  - `up_proj`
  - `v_proj`
- Total tensors: `448`
- LoRA A tensors: `224`
- LoRA B tensors: `224`
- Complete A/B pairs: `224`
- Incomplete A/B pairs: `0`
- DoRA inactive.
- rsLoRA inactive.

Problems faced:

- Needed to confirm PEFT naming and tensor structure before any spectral work.

How problem was fixed:

- Used safetensors metadata/tensor inspection only.
- Did not load Llama-2 or run inference.

Output files generated:

- `logs/adapter_inspection_20260517T143951Z.json`
- `outputs/adapter_tensor_summary.csv`

What this means for the research:

- Adapter-only A/B extraction is structurally possible for all expected LoRA target modules.

What not to claim yet:

- This does not show the adapter is backdoored in execution. It only verifies file structure.

## 4. Spectral Analysis

Goal:

Compute compact singular-value statistics for each LoRA update without forming dense `B @ A` matrices by default.

Command/script used:

```bash
python scripts/02_extract_spectral_stats.py
```

Key result:

- A/B pairs processed: `224`
- Rank values: `[8]`
- Mean top-1 energy share: `0.718263`
- Mean top-3 energy share: `0.902279`
- Warnings: none
- Highest concentration examples included:
  - layer 1 `down_proj`, top1 `0.997012`
  - layer 30 `gate_proj`, top1 `0.986115`
  - layer 30 `up_proj`, top1 `0.980528`

Problems faced:

- Dense LoRA deltas for MLP projections can be large.

How problem was fixed:

- Implemented compact low-rank SVD via QR factors and small rank-space SVD.
- Kept computation CPU-only and adapter-only.

Output files generated:

- `logs/spectral_stats_20260517T145251Z.json`
- `outputs/spectral_stats.csv`
- `reports/figures/singular_value_spectra_by_module.png`

What this means for the research:

- The backdoored adapter has strong singular-value concentration, which motivates continuing to comparison and sanitisation.

What not to claim yet:

- Spectral concentration alone is not a backdoor-specific signal. A clean-reference comparison and downstream evaluation are required.

## 5. Clean Reference Selection

Goal:

Find a clean LoRA reference adapter with structure similar enough for a spectral sanity comparison.

Command/script used:

```bash
python scripts/03_clean_reference_search.py --fetch-remote-configs
python scripts/04_inspect_flagalpha_clean_adapter.py --download --inspect-bin
python scripts/05_inspect_clean_adapter_candidates.py --download --inspect-bin
```

Key result:

- Best provisional structural reference:
  `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- FlagAlpha properties:
  - `.bin` loaded safely with `torch.load(..., map_location="cpu", weights_only=True)`
  - rank `8`
  - LoRA alpha `32`
  - all 7 target modules
  - `224` complete A/B pairs
- Alternative candidates:
  - `manojpatil/llama-2-7b-chat-lora-adaptor`: safe but rank 64 and only q/v-style subset.
  - `Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br`: safe and rank 8 but only q/v-style subset.

Problems faced:

- No perfect clean reference was found.
- FlagAlpha is Chinese chat and not task/distribution matched.
- FlagAlpha uses `adapter_model.bin`, requiring safe weight-only loading.

How problem was fixed:

- Treated FlagAlpha only as a structural reference.
- Used `weights_only=True` for `.bin` inspection.
- Explicitly recorded caveats.

Output files generated:

- `outputs/clean_reference_candidates.csv`
- `logs/clean_reference_search_20260517T151217Z.json`
- `logs/flagalpha_clean_adapter_inspection_20260517T152925Z.json`
- `outputs/flagalpha_adapter_tensor_summary.csv`
- `logs/clean_adapter_candidate_inspection_20260517T154855Z.json`
- `outputs/clean_adapter_candidate_comparison.csv`
- per-candidate tensor summaries under `outputs/`

What this means for the research:

- A structural sanity comparison is possible.

What not to claim yet:

- Do not claim FlagAlpha is a matched clean baseline. It is only a structural reference.

## 6. Clean-vs-Backdoor Spectral Comparison

Goal:

Compare normalized spectral metrics between the BackdoorLLM adapter and the FlagAlpha structural reference.

Command/script used:

```bash
python scripts/06_compare_clean_vs_backdoor_spectra.py
```

Key result:

- Matched module count: `224`
- Unmatched backdoor modules: `0`
- Unmatched clean modules: `0`
- Mean backdoor top1: `0.718263`
- Mean clean top1: `0.566749`
- Mean backdoor top3: `0.902279`
- Mean clean top3: `0.841243`
- Mean backdoor entropy: `0.461946`
- Mean clean entropy: `0.628052`
- Backdoor showed higher concentration in several target module types, including:
  - `down_proj`
  - `gate_proj`
  - `o_proj`
  - `q_proj`
  - `up_proj`
  - `v_proj`

Problems faced:

- Raw singular magnitudes are not directly comparable because LoRA alpha differs:
  Backdoor alpha `16`, FlagAlpha alpha `32`.

How problem was fixed:

- Focused on normalized metrics:
  top1 share, top3 share, entropy, effective rank, normalized singular curves.

Output files generated:

- `logs/clean_vs_backdoor_spectral_comparison_20260517T160214Z.json`
- `outputs/clean_vs_backdoor_spectral_comparison.csv`
- `reports/figures/clean_vs_backdoor_top1_by_layer.png`
- `reports/figures/clean_vs_backdoor_top3_by_layer.png`
- `reports/figures/clean_vs_backdoor_entropy_by_layer.png`
- `reports/figures/clean_vs_backdoor_singular_curve_mean.png`

What this means for the research:

- Week-2 spectral GO/NO-GO sanity check supports continuing to adapter-only sanitisation design.

What not to claim yet:

- Do not claim a detected backdoor mechanism or ASR reduction. This is spectral evidence only.

## 7. Sanitised Adapter Generation

Goal:

Create spectral-only sanitised adapter variants by attenuating top singular components and refactoring back into LoRA A/B tensors.

Command/script used:

```bash
python scripts/07_generate_spectral_sanitised_adapters.py
```

Key result:

Generated six variants under `outputs/sanitised_adapters/`:

- `top1_gamma_0.0`
- `top1_gamma_0.25`
- `top1_gamma_0.50`
- `top3_gamma_0.0`
- `top3_gamma_0.25`
- `top3_gamma_0.50`

Each variant contains:

- `adapter_config.json`
- `adapter_model.safetensors`
- `sanitisation_report.json`

Summary examples:

- `top1_gamma_0.50`: mean top1 `0.718263 -> 0.442466`, entropy `0.461946 -> 0.695001`
- `top3_gamma_0.50`: mean top1 `0.718263 -> 0.582896`, entropy `0.461946 -> 0.649288`

Problems faced:

- Need to edit LoRA updates without modifying original cached adapter.

How problem was fixed:

- Saved each edited adapter to a separate `outputs/sanitised_adapters/<variant>/` folder.
- Preserved adapter config.
- Wrote PEFT-compatible safetensors files.

Output files generated:

- `logs/sanitised_adapter_generation_20260517T163541Z.json`
- `outputs/sanitised_adapter_generation_summary.csv`
- six variant directories under `outputs/sanitised_adapters/`

What this means for the research:

- Spectral-only candidate defences exist and can be tested downstream.

What not to claim yet:

- Do not claim any safety or utility improvement until model-based evaluation is run.

## 8. Adapter File Smoke Checks

Goal:

Validate generated sanitised adapter files before PEFT loading.

Command/script used:

```bash
python scripts/08_smoke_check_sanitised_adapters.py
```

Key result:

- Variants checked: `6`
- Passed: `6`
- Failed: `0`
- Each variant:
  - `448` tensors
  - `224` complete A/B pairs
  - ranks `[8]`
  - all required target modules present
  - tensor shapes match original
  - all tensors finite

Problems faced:

- Initially local sync did not include safetensors files because large weight files were ignored by git.

How problem was fixed:

- Verified files after syncing from server.
- Used safetensors/file-level smoke check.

Output files generated:

- `logs/sanitised_adapter_smoke_check_20260517T190339Z.json`
- `outputs/sanitised_adapter_smoke_check_summary.csv`

What this means for the research:

- The generated adapters are structurally valid enough for PEFT loading tests.

What not to claim yet:

- File validity is not model behavior. No generation or evaluation was involved here.

## 9. PEFT Loading Smoke Test

Goal:

Verify PEFT config loading, 4-bit base loading, adapter attachment, and tiny forward shape.

Command/script used:

```bash
python scripts/09_peft_loading_smoke_test.py
python scripts/09_peft_loading_smoke_test.py --load-base-4bit --variant top1_gamma_0.50
python scripts/09_peft_loading_smoke_test.py --load-base-4bit --variant top1_gamma_0.50 --tiny-forward-check
```

Key result:

- Adapter-only mode passed for original plus six sanitised adapters.
- After freeing GPU memory, 4-bit base attach passed:
  - base loaded: `True`
  - adapter attached: `True`
  - OOM: `False`
- Tiny forward shape check passed:
  - tiny forward ok: `True`
  - logits shape: `[1, 5, 32000]`

Problems faced:

- First base attach failed on `ki-016` because only about 437 MB VRAM was free.
- GPU was occupied by other Python processes.

How problem was fixed:

- Added `scripts/10_gpu_memory_diagnosis.py`.
- Freed GPU memory before retrying.

Output files generated:

- `logs/peft_loading_smoke_test_20260517T191954Z.json`
- `logs/peft_loading_smoke_test_20260519T165423Z.json`
- `logs/peft_loading_smoke_test_20260519T165754Z.json`
- `outputs/peft_loading_smoke_test_summary.csv`
- `logs/gpu_memory_diagnosis_20260517T192845Z.json`
- `logs/gpu_memory_diagnosis_20260519T165210Z.json`

What this means for the research:

- The base model and sanitised adapter can be attached successfully in 4-bit mode when enough VRAM is available.

What not to claim yet:

- A successful forward pass is not an evaluation of jailbreak behavior or utility.

## 10. Tiny Inference Smoke Test

Goal:

Verify very short deterministic generation for the original adapter and one sanitised adapter.

Command/script used:

```bash
python scripts/11_tiny_inference_smoke_test.py
```

Key result:

- On `ki-010`, after using a fresh HF cache:
  - execution mode: `isolated_subprocess_per_adapter`
  - variants checked: `original`, `top1_gamma_0.50`
  - prompts per variant: `2`
  - all generation succeeded: `True`
  - OOM: `False`
- GPU memory was released properly between adapter subprocesses.

Problems faced:

- Copied HF cache from `ki-016` to `ki-010` had 0-byte snapshot files.
- Initial sequential in-process generation retained CUDA memory and prevented the second adapter from loading.

How problem was fixed:

- Created fresh HF cache on `ki-010`:
  `/home/43e3/hf-cache-aisp`
- Patched `scripts/11_tiny_inference_smoke_test.py` to use isolated subprocesses per adapter.

Output files generated:

- `logs/tiny_inference_smoke_test_20260519T194531Z.json`
- `outputs/tiny_inference_smoke_test_summary.csv`
- child logs under `logs/tiny_inference_children_20260519T194531Z/`

What this means for the research:

- Generation plumbing works for both original and a sanitised adapter with subprocess isolation.

What not to claim yet:

- These harmless prompt generations are not clean utility scores and not ASR.

## 11. Small Baseline Pilot Evaluation

Goal:

Run a pilot-only bounded evaluation to verify multi-adapter generation, prompt logging, and simple indicators.

Command/script used:

```bash
python scripts/12_small_baseline_evaluation.py
```

Key result:

- Adapters tested:
  - `original`
  - `top1_gamma_0.50`
  - `top3_gamma_0.50`
- Prompt rows: `30`
- Clean prompts completed: `15`
- Trigger-pilot prompts completed: `15`
- Failures: `0`
- OOM count: `0`
- Refusal count: `0`
- Unsafe keyword flags: `3`
- All generation succeeded: `True`

Problems faced:

- No official trigger prompts were found locally.
- Simple unsafe keyword flags can produce benign matches, e.g. the word `harm` in safe-AI explanations.

How problem was fixed:

- Trigger-pilot prompts were labeled unverified.
- Unsafe keyword flags were interpreted as crude indicators only, not judgments.
- The journal/status explicitly record that this is not ASR.

Output files generated:

- `logs/small_baseline_evaluation_20260519T195636Z.json`
- `outputs/small_baseline_evaluation.csv`
- `outputs/small_baseline_evaluation_summary.csv`
- child logs under `logs/small_baseline_children_20260519T195636Z/`

What this means for the research:

- Evaluation plumbing works for three adapters with bounded prompts and structured logs.
- It is reasonable to implement a real bounded ASR/clean-utility script next.

What not to claim yet:

- Do not claim ASR reduction.
- Do not claim clean utility preservation.
- Do not compare defences scientifically from this pilot.

## Next Report-Relevant Step

Define explicit prompt files and scoring rules for a real bounded ASR/clean-utility evaluation. The trigger source must either be verified from BackdoorLLM documentation/code or clearly labeled as unverified. Keep the first real evaluation small before scaling.

## 12. Bounded Evaluation Framework From Prompt Files

Goal:

Create explicit prompt files and a reusable bounded evaluation runner so future evaluation runs are driven by versioned inputs instead of hard-coded prompt lists.

Command/script used:

```bash
python scripts/13_bounded_eval_from_prompt_files.py
```

Key result:

- Created `data/eval_prompts/clean_utility_small.jsonl` with 10 harmless clean prompts.
- Created `data/eval_prompts/trigger_probe_small_unverified.jsonl` with 5 benign unverified trigger-probe placeholders.
- Created `configs/eval_small.yaml` for the first bounded prompt-file run.
- Created `scripts/13_bounded_eval_from_prompt_files.py`.
- Updated `.gitignore` so `data/eval_prompts/` prompt files can be tracked while generated data remains ignored.
- The runner uses isolated subprocesses per adapter, 4-bit base loading, deterministic generation, Llama-2 `[INST] ... [/INST]` formatting, and structured JSON/CSV outputs.
- This step was syntax-checked only. No model loading or inference was run locally by Codex.

Problems faced:

- Official BackdoorLLM trigger format is still not verified locally.
- The evaluation must avoid converting placeholder trigger probes into ASR claims.

How problem was fixed:

- Trigger-probe prompts are explicitly labeled `trigger_probe_unverified`.
- The JSON report and summary CSV include `is_final_asr: false`.
- The runner separates clean completion rate from unverified trigger-probe completion rate.

Output files generated:

- `data/eval_prompts/clean_utility_small.jsonl`
- `data/eval_prompts/trigger_probe_small_unverified.jsonl`
- `data/eval_prompts/README.md`
- `configs/eval_small.yaml`
- `scripts/13_bounded_eval_from_prompt_files.py`
- `.gitignore`

Expected output files after running:

- `logs/bounded_eval_from_prompt_files_<timestamp>.json`
- `outputs/bounded_eval_outputs.csv`
- `outputs/bounded_eval_summary.csv`

What this means for the research:

- The project now has a clearer bridge from smoke tests to bounded evaluation with explicit, reviewable prompt files and transparent heuristic metrics.

What not to claim yet:

- Do not call unverified trigger-probe completion rate ASR.
- Do not claim clean utility preservation from this framework alone.
- Do not make final defence claims until official trigger/evaluation sources are verified and the final experiment is run.
