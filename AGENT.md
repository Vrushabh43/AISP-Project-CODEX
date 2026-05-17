# AGENT.md

Stable project memory for future Codex sessions. Read this file and `status.md`
before doing any work.

## Project Title

Post-Hoc Sanitisation of Backdoored LoRA Adapters via Sensitivity-Aware
Singular-Component Attenuation.

## Research Question

Can suspicious singular components of an already-trained backdoored LoRA update
matrix be attenuated post-hoc to reduce Attack Success Rate while preserving
clean utility?

Operational version:
Does combining spectral energy with clean-prompt sensitivity identify LoRA
singular components whose attenuation reduces ASR more effectively than uniform
adapter scaling or top-sigma cutoff while retaining clean utility?

## Final Scope

- Course: MAI/MKI - AI Security and Privacy.
- Topic: Backdooring of Large Language Models.
- Study type: focused 7-8 week empirical study, not a broad framework.
- Threat model: trusted base model, untrusted third-party LoRA adapter,
  white-box adapter access, no trigger knowledge, no original training data,
  no retraining budget, and a small clean calibration set.
- Main model family: Llama-2-7B-Chat or a compatible mirror.
- Main adapter: one publicly released BackdoorLLM BadNets jailbreak LoRA
  adapter.
- Main metrics: Attack Success Rate, clean utility on held-out clean prompts,
  and ASR-utility trade-off curves.
- Stretch items only: Llama-Guard judge, C4 perplexity, second random seed,
  and RoRA-style global rescaling.

## Exact Hugging Face Adapter ID

`BackdoorLLM/Jailbreak_Llama2-7B_BadNets`

Important: `BadNets` is plural in the Hugging Face model ID.

## Base Model Choices

Primary:

- `meta-llama/Llama-2-7b-chat-hf`

Compatible fallback mirror, if licensing/access requires it:

- `NousResearch/Llama-2-7b-chat-hf`

Do not silently switch base models. Record the choice in `status.md`, config,
and logs.

## Core Baselines

- B1: no defence / original backdoored adapter.
- B2: uniform adapter scaling.
- B3: top-sigma cutoff.

No extra baseline should become required work unless the user explicitly changes
the scope.

## Proposed Method Summary

For each LoRA target module:

1. Extract LoRA factors `A` and `B`.
2. Compute the LoRA update matrix `Delta W = B @ A` per module.
3. Apply thin SVD: `Delta W = U diag(S) V^T`.
4. Score singular components using spectral energy share and clean-prompt
   sensitivity.
5. Attenuate suspicious singular components multiplicatively.
6. Refactor the edited update back into LoRA `A`/`B` form.
7. Save the sanitised adapter in a normal PEFT-compatible format.

Do not write final research claims before experiments are run and logged.

## Implementation Rules

- Keep the codebase simple and research-friendly.
- Prefer wrappers in this repository's `src/lora_sanitise/` package.
- Do not modify third-party BackdoorLLM source directly unless absolutely
  necessary.
- Keep heavy model downloads, model inference, and GPU evaluations behind
  explicit scripts and explicit user commands.
- Prefer small smoke tests before heavy experiments.
- Use deterministic seeds where possible.
- Make scripts safe to run on a remote server.
- Every script must write logs to `logs/`.
- Every experiment must save structured JSON and/or CSV outputs to `outputs/`.
- Every script must print a short final summary.
- Update `status.md` after each major step.

## What Not To Change

- Do not broaden the project into a general LoRA defence framework.
- Do not silently add more models, attacks, datasets, or baselines.
- Do not turn stretch goals into core requirements.
- Do not claim novelty or success before experiments are complete.
- Do not run heavy GPU jobs without clear scripts and logging.
- Do not download large models unless the user explicitly asks.
- Do not change the exact adapter ID spelling.

## Logging Requirements

- Use timestamped log files in `logs/`.
- Include command-line arguments, relevant environment details, seed values,
  model IDs, adapter IDs, and output paths.
- Save experiment metrics as structured files in `outputs/`.
- Keep generated figures in `reports/figures/`.
- Keep generated tables in `reports/tables/`.
- Record latest log files and latest results in `status.md`.

## Current Coding Style

No substantial Python implementation existed at bootstrap. The intended style is:

- Python package code under `src/lora_sanitise/`.
- Script entry points under `scripts/`.
- Conservative dependencies and explicit imports.
- `pathlib` for filesystem paths.
- `argparse` for script CLIs.
- Standard-library logging plus structured JSON/CSV outputs.
- Clear function boundaries, type hints where useful, and minimal comments.
- No broad abstractions until repeated experimental code actually needs them.
