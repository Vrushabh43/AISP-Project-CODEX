# Post-Hoc Sanitisation of Backdoored LoRA Adapters

Focused semester-project implementation for:

**Post-Hoc Sanitisation of Backdoored LoRA Adapters via
Sensitivity-Aware Singular-Component Attenuation**

Research question:

Can suspicious singular components of an already-trained backdoored LoRA update
matrix be attenuated post-hoc to reduce Attack Success Rate while preserving
clean utility?

This repository is intentionally small. It is not a broad framework.

## Scope

- Base model: `meta-llama/Llama-2-7b-chat-hf` or compatible mirror
  `NousResearch/Llama-2-7b-chat-hf`.
- Backdoored adapter: `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`.
- Threat model: trusted base model, untrusted third-party LoRA adapter,
  white-box adapter access, no trigger knowledge, no original training data,
  no retraining budget, and a small clean calibration set.
- Core baselines: original adapter, uniform adapter scaling, top-sigma cutoff.
- Proposed method: sensitivity-aware singular-component attenuation.

Read `AGENT.md` and `status.md` before continuing work in a new session.

## Project Layout

```text
AISP-Project-CODEX/
  AGENT.md
  status.md
  README.md
  requirements.txt
  .gitignore
  configs/
    experiment.yaml
  src/
    lora_sanitise/
      __init__.py
      config.py
      logging_utils.py
      lora_io.py
      delta_w.py
      svd_tools.py
      attenuation.py
      sensitivity.py
      baselines.py
      eval_asr.py
      eval_clean.py
      plots.py
  scripts/
    00_check_env.py
  data/
    README.md
  logs/
    README.md
  outputs/
    README.md
  reports/
    figures/
    tables/
```

Future experiment scripts should be added under `scripts/` as the study is
implemented.

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## CUDA and GPU Check

Run the lightweight environment check:

```bash
python scripts/00_check_env.py
```

The script prints a short summary and writes a timestamped JSON log to `logs/`.
It checks Python, platform, installed package availability, PyTorch CUDA
visibility, and selected environment variables. It does not download model
weights and does not run model inference.

For CUDA servers, install a PyTorch build matching the server's CUDA driver if
the default wheel is not suitable. See the official PyTorch installation page
for the exact command for that machine.

## Hugging Face Access

Some Llama-2 model repositories are gated. You may need a Hugging Face token
with accepted model access.

Set a token in one of these ways:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN = "your_token_here"
```

Then run the optional API smoke check:

```bash
python scripts/00_check_env.py --check-hf --adapter-id BackdoorLLM/Jailbreak_Llama2-7B_BadNets
```

This checks Hugging Face API reachability and adapter repository metadata. It
does not download the adapter weights.

## Hugging Face Cache and Adapter Check

Before loading any model, check cache state and adapter repo availability:

```bash
python scripts/01_check_hf_cache_and_adapter.py
```

This script scans the Hugging Face cache, checks metadata for:

- `NousResearch/Llama-2-7b-chat-hf`
- `tatsu-lab/alpaca`
- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`

It writes a JSON log to `logs/` and does not load Llama-2, load the adapter,
run inference, or use the GPU.

If the server has a non-default Hugging Face cache location, pass it explicitly:

```bash
python scripts/01_check_hf_cache_and_adapter.py --cache-root /path/to/huggingface/hub
```

For an offline/local-cache-only check:

```bash
python scripts/01_check_hf_cache_and_adapter.py --offline
```

If the adapter repo is reachable but not cached, first fetch only small metadata
files:

```bash
python scripts/01_check_hf_cache_and_adapter.py --download-adapter-metadata-only
```

To download only the backdoored LoRA adapter snapshot into the normal Hugging
Face cache, without downloading the base model:

```bash
python scripts/01_check_hf_cache_and_adapter.py --download-adapter
```

## Reproducibility Rules

- Use deterministic seeds where possible.
- Write logs to `logs/`.
- Save structured experiment outputs to `outputs/`.
- Save figures to `reports/figures/`.
- Save tables to `reports/tables/`.
- Prefer small smoke tests before running expensive jobs.
- Record latest commands, logs, results, and blockers in `status.md`.

## Current Status

Bootstrap only. The sanitisation method, baselines, ASR evaluation, and clean
utility evaluation are not implemented yet.
