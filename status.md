# status.md

Dynamic project status for future Codex sessions. Read this after `AGENT.md`
and update it after every major step.

## Current Phase

Phase 2 implementation bootstrap.

## Completed Work

- Inspected the existing repository on 2026-05-17.
- Found existing top-level files:
  - `AGENT.md`
  - `status.md`
  - `task.txt`
  - `lora_sanitisation_master_project.md`
- Rewrote `AGENT.md` with clean stable project memory and fixed encoding
  artifacts.
- Created the research-friendly project skeleton:
  - `configs/`
  - `src/lora_sanitise/`
  - `scripts/`
  - `data/`
  - `logs/`
  - `outputs/`
  - `reports/figures/`
  - `reports/tables/`
- Added `README.md` with setup and safe smoke-test instructions.
- Added `.gitignore`.
- Added `requirements.txt`.
- Added `configs/experiment.yaml`.
- Added placeholder package modules under `src/lora_sanitise/`.
- Added `scripts/00_check_env.py` as a lightweight environment and optional
  Hugging Face access check.
- Added README placeholders for `data/`, `logs/`, and `outputs/`.
- Verified `scripts/00_check_env.py` syntax with a no-bytecode AST parse.
- Ran `python scripts/00_check_env.py` once in the current base interpreter. It
  performed no network access, no model download, and no inference.

## Next Immediate Task

Create and activate a virtual environment, install requirements, and run the
lightweight environment check. Do not run model inference yet.

## Commands To Run

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/00_check_env.py
```

Linux/macOS shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/00_check_env.py
```

Optional Hugging Face API smoke check, after setting a token if needed:

```bash
python scripts/00_check_env.py --check-hf --adapter-id BackdoorLLM/Jailbreak_Llama2-7B_BadNets
```

## Known Blockers

- Need to verify the actual server GPU and CUDA environment.
- Need to verify Hugging Face authentication and Llama-2 access.
- Need to verify that `BackdoorLLM/Jailbreak_Llama2-7B_BadNets` is reachable
  from the target machine.
- Need to inspect the adapter file format before implementing LoRA A/B
  extraction.

## Latest Log Files

- `logs/env_check_20260517T131640Z.json`

## Latest Results

- No experiments have been run.
- No model downloads have been performed.
- No model inference has been performed.
- Local bootstrap env check result: script ran successfully as a diagnostic but
  exited with code 1 because the current base interpreter is missing required
  ML packages. This is expected before creating the virtual environment and
  installing `requirements.txt`.

## Decisions Made

- Keep the exact adapter ID as
  `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`.
- Use `meta-llama/Llama-2-7b-chat-hf` as the primary base model choice.
- Use `NousResearch/Llama-2-7b-chat-hf` only as a compatible fallback.
- Keep the first script limited to environment checks and optional HF metadata
  lookup. It must not download model weights or run inference.

## TODO

- [ ] Create virtual environment.
- [ ] Install requirements.
- [ ] Run `python scripts/00_check_env.py`.
- [ ] Run optional HF API smoke check.
- [ ] Inspect adapter metadata and file names.
- [ ] Implement adapter inspection script.
- [ ] Implement baseline reproduction script.
- [ ] Implement spectral sanity check.
- [ ] Implement uniform scaling baseline.
- [ ] Implement top-sigma baseline.
- [ ] Implement proposed method.

## 2026-05-17T13:58Z Environment Diagnosis Update

Scope of this step: diagnosis only. No packages were installed or uninstalled.
No files were deleted. No model was loaded or downloaded. No GPU-heavy code was
run.

Inspected:

- `AGENT.md`
- `status.md`
- `requirements.txt`
- `scripts/00_check_env.py`
- existing `logs/` contents

Added:

- `scripts/00_check_env_verbose.py`

The verbose diagnostic script prints and logs:

- Python executable, version, `sys.prefix`, `sys.base_prefix`, and `sys.path`
- whether Python believes a virtual environment is active
- user-site package status
- `python -m pip --version`
- import status, distribution version, and module path for:
  - `torch`
  - `transformers`
  - `peft`
  - `accelerate`
  - `bitsandbytes`
  - `huggingface_hub`
- PyTorch CUDA availability, CUDA build version, device names, and VRAM if
  torch imports successfully
- a JSON log under `logs/env_check_verbose_*.json`

Diagnostics run from the current Codex shell:

- `Get-Command python`
- `where.exe python`
- `python -c "import sys; print(sys.executable); print(sys.path)"`
- `Get-Command pip`
- `where.exe pip`
- `python -m pip --version`
- `pip --version`
- `python -m pip list | Select-String -Pattern 'torch|transformers|peft|accelerate|bitsandbytes|huggingface'`
- `python -c "import site; print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('USER_SITE=', site.getusersitepackages())"`
- `python -c "import importlib.util as u; print('torch', u.find_spec('torch')); print('transformers', u.find_spec('transformers')); print('peft', u.find_spec('peft'))"`
- `python scripts/00_check_env_verbose.py`

Observed in the current Codex shell:

- `python` resolves to `C:\Program Files\Python310\python.exe`.
- `pip` resolves to `C:\Program Files\Python310\Scripts\pip.exe`.
- `python -m pip --version` and `pip --version` point to the same system
  Python 3.10 installation.
- `VIRTUAL_ENV` is empty.
- `Test-Path .venv` returned `False` in this workspace view.
- `sys.prefix == sys.base_prefix`, so Python does not think a venv is active.
- user-site packages are enabled:
  `C:\Users\Vrushabh Vasoya\AppData\Roaming\Python\Python310\site-packages`.
- `torch`, `transformers`, and `peft` were not import-resolvable in this Codex
  shell.
- Latest local verbose log:
  `logs/env_check_verbose_20260517T135810Z.json`

Diagnosis so far:

- The current Codex shell is not using the expected project `.venv`.
- I cannot yet validate the terminal state where CUDA was visible on the RTX
  2080 SUPER, because that environment is not active in this tool process.
- The most likely explanation for `torch` being visible but `transformers` and
  `peft` missing in your terminal is that `pip install` and `python` were
  pointed at different Python environments, or the install completed only
  partially.
- A different reported torch version is also consistent with multiple Python
  installs/environments on PATH: Python 3.10, Python 3.11, Python 3.13, MSYS2
  Python, and possible user-site packages are all present on this machine.

Commands to run next in the terminal where you believe the venv is active:

Windows PowerShell:

```powershell
Get-Command python | Format-List *
where.exe python
python -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix); print(sys.path)"
Get-Command pip | Format-List *
where.exe pip
python -m pip --version
pip --version
python -m pip list | Select-String -Pattern 'torch|transformers|peft|accelerate|bitsandbytes|huggingface'
python -c "import site; print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('USER_SITE=', site.getusersitepackages())"
python -c "import importlib.util as u; print('torch', u.find_spec('torch')); print('transformers', u.find_spec('transformers')); print('peft', u.find_spec('peft'))"
python scripts/00_check_env_verbose.py
```

Linux/macOS equivalent:

```bash
which python
python -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix); print(sys.path)"
which pip
python -m pip --version
pip --version
python -m pip list | grep -E 'torch|transformers|peft|accelerate|bitsandbytes|huggingface'
python -c "import site; print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('USER_SITE=', site.getusersitepackages())"
python -c "import importlib.util as u; print('torch', u.find_spec('torch')); print('transformers', u.find_spec('transformers')); print('peft', u.find_spec('peft'))"
python scripts/00_check_env_verbose.py
```

Expected healthy output:

- `sys.executable` points inside the project venv, for example
  `...\AISP-Project-CODEX\.venv\Scripts\python.exe` on Windows.
- `python -m pip --version` points inside the same `.venv`.
- `pip --version` also points inside the same `.venv`; if it does not, use
  `python -m pip ...` instead of bare `pip`.
- `sys.prefix != sys.base_prefix`.
- `VIRTUAL_ENV` points to the project `.venv`.
- user-site packages are disabled or at least not ahead of `.venv` packages.
- `torch`, `transformers`, `peft`, `accelerate`, and `huggingface_hub` import
  successfully.
- `bitsandbytes` may be missing on Windows because `requirements.txt` excludes
  it on Windows.
- CUDA may be available if the active torch build supports the local NVIDIA
  driver.

Logs to share back:

- The terminal output from `python scripts/00_check_env_verbose.py`.
- The newest `logs/env_check_verbose_*.json` file created by that command.
- The output of `python -m pip --version` and `pip --version`.

Current blockers:

- Need the verbose diagnostic output from the actual activated terminal/server
  environment.
- Do not proceed to adapter inspection until `transformers`, `peft`,
  `accelerate`, `huggingface_hub`, and `torch` all import from the same intended
  environment.

## 2026-05-17T14:11:48Z HF Cache And Adapter Check Bootstrap

Scope of this step: safe Hugging Face cache/repo verification only. No files or
folders were deleted. No packages were installed or uninstalled. No base model
was loaded or downloaded. No adapter was loaded into a model. No inference or
GPU-heavy code was run.

Inspected:

- `AGENT.md`
- `status.md`
- `README.md`
- existing `scripts/` state

Created:

- `scripts/01_check_hf_cache_and_adapter.py`

Updated:

- `README.md` with a "Hugging Face Cache and Adapter Check" section and exact
  commands.

Script behavior:

- Prints Python executable, Python version, venv status, and selected
  environment variables.
- Prints Hugging Face cache roots checked.
- Scans Hub cache directories for:
  - `NousResearch/Llama-2-7b-chat-hf`
  - `tatsu-lab/alpaca`
  - `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
- Also checks the Hugging Face `datasets` cache pattern for `tatsu-lab/alpaca`.
- Checks Hugging Face repo metadata using `huggingface_hub` when not run with
  `--offline`.
- Does not load Llama-2, load the adapter, run inference, or use the GPU.
- Writes a JSON log to `logs/hf_cache_adapter_check_*.json`.
- Download behavior is opt-in only:
  - `--download-adapter-metadata-only` downloads only small metadata files.
  - `--download-adapter` downloads only the adapter repo snapshot, not the base
    model.

Cache status from the current user-provided server situation, pending script
verification in the activated server environment:

- `NousResearch/Llama-2-7b-chat-hf`: reported cached.
- `tatsu-lab/alpaca`: reported cached.
- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`: reported not visible in cache yet.

Validation performed in the current Codex shell:

- Syntax-only AST parse for `scripts/01_check_hf_cache_and_adapter.py` passed.
- The script was not run for authoritative cache status in this Codex shell
  because earlier diagnostics showed this shell is not the activated
  server/venv environment.

Exact command to run next in the activated server environment:

```bash
python scripts/01_check_hf_cache_and_adapter.py
```

If the server cache is non-default, pass it explicitly:

```bash
python scripts/01_check_hf_cache_and_adapter.py --cache-root /path/to/huggingface/hub
```

If the adapter repo is reachable but only metadata should be fetched first:

```bash
python scripts/01_check_hf_cache_and_adapter.py --download-adapter-metadata-only
```

If metadata is reachable and you want to cache only the LoRA adapter snapshot:

```bash
python scripts/01_check_hf_cache_and_adapter.py --download-adapter
```

Expected output:

- `huggingface_hub available: True`
- Cache roots checked are printed.
- Cache status should report:
  - `NousResearch/Llama-2-7b-chat-hf`: `cached=True`
  - `tatsu-lab/alpaca`: `cached=True`
  - `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`: likely `cached=False` before
    adapter download
- Repo metadata reachability should report
  `BackdoorLLM/Jailbreak_Llama2-7B_BadNets: reachable=True` if Hugging Face
  access/network are working.
- A new JSON log should appear under `logs/hf_cache_adapter_check_*.json`.

Current blockers:

- Need the script output from the activated server environment to confirm the
  cache state authoritatively.
- Adapter inspection should wait until the adapter repo is reachable and either
  cached or intentionally downloaded with the adapter-only command.
- Full Llama-2 loading is still unsafe on the RTX 2080 SUPER 7.6 GB VRAM unless
  a later script uses explicit low-memory/quantized settings.
