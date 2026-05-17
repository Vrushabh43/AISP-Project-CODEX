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

## 2026-05-17T14:23:43Z Environment Fixed And Adapter Cached

Scope of this step: environment repair verification and adapter-only download.
No full Llama-2 model was loaded. No inference was run. No BackdoorLLM
evaluation was run. No GPU-heavy code was run.

Completed by the user on the server:

- Confirmed the initial issue was `PYTHONPATH` leaking
  `/home/43e3/.local/lib/python3.12/site-packages` ahead of the project venv.
- Fixed the active shell by running:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

- Installed missing/broken venv dependencies only:

```bash
python -m pip install "typing_extensions>=4.12" "transformers>=4.44,<4.54" "accelerate>=0.33,<1.2"
```

- Re-ran verbose environment diagnostics:

```bash
python scripts/00_check_env_verbose.py
```

Verified environment:

- Python executable:
  `/home/43e3/solr-home/AISP-Project-CODEX/.venv/bin/python`
- venv active: `True`
- user-site disabled: `ENABLE_USER_SITE=False`
- `torch`: imports from `.venv`, version `2.7.1+cu126`
- `transformers`: imports from `.venv`, version `4.53.3`
- `peft`: imports from `.venv`, version `0.15.2`
- `accelerate`: imports from `.venv`, version `1.1.1`
- `bitsandbytes`: imports from `.venv`, version `0.46.1`
- `huggingface_hub`: imports from `.venv`, version `0.33.5`
- CUDA available: `True`
- GPU: NVIDIA GeForce RTX 2080 SUPER, 7.6 GB VRAM

Downloaded only the backdoored LoRA adapter snapshot:

```bash
python scripts/01_check_hf_cache_and_adapter.py --download-adapter
```

Verified cache/repo status after download:

- `NousResearch/Llama-2-7b-chat-hf`: cached `True`
- `tatsu-lab/alpaca`: cached `True`
- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`: cached `True`
- Adapter repo reachable: `True`
- Adapter snapshot path:
  `/home/huggingface/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b`

Latest log files:

- `logs/env_check_verbose_20260517T142304Z.json`
- `logs/hf_cache_adapter_check_20260517T142343Z.json`

Important shell requirement for future commands:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
```

Next immediate task:

- Implement and run adapter file inspection only. This must remain
  CPU/filesystem-level and must not load the full base model or run inference.

Suggested next command after the adapter inspection script exists:

```bash
python scripts/01_inspect_adapter.py
```

Current blockers:

- Need to inspect adapter files and PEFT config before implementing LoRA A/B
  extraction.
- Do not load the full Llama-2 model on the RTX 2080 SUPER without explicit
  low-memory/quantized loading controls.

## 2026-05-17T14:37:21Z Adapter Inspection Script Added

Scope of this step: adapter file inspection script only. No full Llama-2 model
was loaded. No inference was run. No BackdoorLLM evaluation was run. No GPU code
was added or executed.

Inspected before editing:

- `AGENT.md`
- `status.md`
- existing `scripts/01_inspect_adapter.py` state; the file did not exist.

Files created/modified:

- Created `scripts/01_inspect_adapter.py`.
- Appended this section to `status.md`.

What `scripts/01_inspect_adapter.py` does:

- Reads only cached adapter files:
  - `adapter_config.json`
  - `adapter_model.safetensors`
- Reports adapter snapshot path and file existence.
- Reports selected PEFT config fields:
  - `base_model_name_or_path`
  - `peft_type`
  - `task_type`
  - `r`
  - `lora_alpha`
  - `lora_dropout`
  - `target_modules`
  - `bias`
  - `inference_mode`
- Lists safetensors tensor keys, shapes, dtypes, inferred ranks, and likely
  LoRA A/B classification.
- Groups likely LoRA A/B tensors by module name.
- Counts LoRA module groups and complete A/B pairs.
- Warns about incomplete A/B pairs, rank mismatches, and DoRA/rsLoRA-like
  config fields.
- Writes JSON log to `logs/adapter_inspection_*.json`.
- Writes CSV tensor summary to `outputs/adapter_tensor_summary.csv`.
- Prints a short final summary.

Validation performed:

- Syntax-only AST parse passed:

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/01_inspect_adapter.py').read_text(encoding='utf-8')); print('adapter inspection script syntax OK')"
```

The script was not run in the current Codex shell because the authoritative
adapter cache path is on the activated server environment.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/01_inspect_adapter.py
```

Expected output:

- Adapter type should be reported from `adapter_config.json`, likely `LORA`.
- Rank should be reported from config and inferred from A/B tensor shapes.
- Target modules should be printed from config.
- Complete A/B pair count should be greater than zero.
- `A/B extraction looks possible` should be `True` if all likely LoRA A tensors
  have matching B tensors and ranks match.
- JSON log should be written to `logs/adapter_inspection_*.json`.
- CSV should be written to `outputs/adapter_tensor_summary.csv`.

Latest expected output files after running:

- `logs/adapter_inspection_<timestamp>.json`
- `outputs/adapter_tensor_summary.csv`

Safe to proceed to Delta W extraction next:

- Not yet, until the adapter inspection output confirms complete A/B pairs and
  no extraction blockers.
- If inspection reports `A/B extraction looks possible: True`, the next step
  should be CPU-level Delta W extraction/spectral sanity script, still without
  loading the full base model.

## 2026-05-17T14:41:31Z Adapter Inspection Completed On Server

Scope of this step: adapter file inspection only. No full Llama-2 model was
loaded. No inference was run. No BackdoorLLM evaluation was run. No GPU-heavy
code was run.

Command run by user on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/01_inspect_adapter.py
```

Inspection result from server output:

- Adapter snapshot:
  `/home/huggingface/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b`
- `adapter_config.json` exists: `True`
- `adapter_model.safetensors` exists: `True`
- Adapter type: `LORA`
- Task type: `CAUSAL_LM`
- Config rank `r`: `8`
- Inferred ranks: `[8]`
- Target modules:
  `['gate_proj', 'q_proj', 'o_proj', 'v_proj', 'k_proj', 'up_proj', 'down_proj']`
- Tensor count: `448`
- LoRA module groups: `224`
- Complete A/B pairs: `224`
- Incomplete A/B pairs: `0`
- A/B extraction looks possible: `True`

Warnings observed:

- `use_dora` present but inactive/empty: `False`
- `use_rslora` present but inactive/empty: `False`
- `rank_pattern` present but inactive/empty: `{}`
- `alpha_pattern` present but inactive/empty: `{}`

Latest server output files:

- `logs/adapter_inspection_20260517T143951Z.json`
- `outputs/adapter_tensor_summary.csv`

Note:

- These adapter inspection artifacts were not visible in the current local
  Codex workspace view when checked; they exist on the server according to the
  user-provided terminal output.

Decision:

- It is safe to proceed to CPU-level Delta W extraction and spectral sanity
  checks next.
- The next step must still avoid full base-model loading, inference, GPU-heavy
  work, and BackdoorLLM evaluation.

Next immediate task:

- Implement a small Delta W extraction/spectral sanity script that reads the
  adapter safetensors only, computes `Delta W = B @ A` per complete LoRA module
  on CPU, records shapes/ranks/norms/singular values, and writes structured
  logs/outputs.

## 2026-05-17T14:46:07Z Adapter Inspection Artifacts Verified Locally

The server-generated adapter inspection artifacts were pulled into the local
workspace and verified directly.

Verified files:

- `logs/adapter_inspection_20260517T143951Z.json`
- `outputs/adapter_tensor_summary.csv`

Direct verification results:

- Extraction possible: `True`
- PEFT type: `LORA`
- Task type: `CAUSAL_LM`
- Base model recorded in adapter config: `meta-llama/Llama-2-7b-chat-hf`
- Config rank `r`: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Bias: `none`
- Inference mode: `True`
- Tensor count: `448`
- LoRA A tensors: `224`
- LoRA B tensors: `224`
- Unique module groups: `224`
- Complete A/B pairs: `224`
- Incomplete A/B pairs: `0`
- Inferred ranks: `8`
- Rank consistency: `True`
- Tensor dtype: `F16`
- Target module hints:
  `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj`

Warnings are non-blocking:

- `use_dora` present but inactive/empty: `False`
- `use_rslora` present but inactive/empty: `False`
- `rank_pattern` present but inactive/empty: `{}`
- `alpha_pattern` present but inactive/empty: `{}`

Decision:

- Adapter artifacts are now locally verified.
- It is safe to proceed to CPU-only Delta W extraction and spectral sanity
  checks next.
- Continue avoiding full base-model loading, inference, BackdoorLLM evaluation,
  and GPU-heavy work.

## 2026-05-17T14:51:23Z CPU Compact Spectral Stats Implemented

Scope of this step: adapter-only CPU numerical analysis implementation. No
full Llama-2 model loading, no inference, no GPU use, no BackdoorLLM
evaluation, and no sanitisation implementation.

Files created/modified:

- Modified `src/lora_sanitise/lora_io.py`
- Modified `src/lora_sanitise/svd_tools.py`
- Created `scripts/02_extract_spectral_stats.py`
- Appended this section to `status.md`

Implemented in `src/lora_sanitise/lora_io.py`:

- Locate cached adapter snapshot without network access.
- Load `adapter_config.json`.
- Locate `adapter_model.safetensors`.
- Read LoRA tensor metadata from safetensors.
- Group complete LoRA A/B pairs by module prefix.
- Infer layer id and target module type from tensor keys.
- Load one A/B pair at a time from an open safetensors handle.

Implemented in `src/lora_sanitise/svd_tools.py`:

- `compact_svd_singular_values(A, B)` using QR + small `r x r` SVD.
- `compute_energy_shares(S)`.
- `compute_topk_energy(S, k)`.
- `compute_spectral_entropy(S)`.
- `compute_effective_rank(S)`.
- `frobenius_norm_from_singular_values(S)`.
- Robust handling for empty or near-zero singular values.

Implemented in `scripts/02_extract_spectral_stats.py`:

- Reads only `adapter_config.json` and `adapter_model.safetensors`.
- Processes each complete LoRA A/B pair on CPU.
- Avoids dense `Delta W = B @ A` construction.
- Computes compact singular values and per-module spectral stats:
  - module name
  - layer id
  - target module type
  - A/B shapes and dtypes
  - rank
  - Frobenius norm from singular values
  - top-1 energy share
  - top-3 energy share
  - normalized spectral entropy
  - effective rank
  - max singular value
  - singular values list
- Saves timestamped JSON log to `logs/spectral_stats_<timestamp>.json`.
- Saves fixed CSV to `outputs/spectral_stats.csv`.
- Saves combined matplotlib plot to
  `reports/figures/singular_value_spectra_by_module.png`.
- If the fixed CSV or plot already exists, moves the previous file to a
  timestamped `.bak_<timestamp>` backup before writing the new one.

Validation performed:

- Syntax-only AST parse passed for:
  - `src/lora_sanitise/lora_io.py`
  - `src/lora_sanitise/svd_tools.py`
  - `scripts/02_extract_spectral_stats.py`

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/02_extract_spectral_stats.py
```

Expected output:

- `A/B pairs processed: 224`
- `Rank values: [8]`
- Mean top-1 energy share printed.
- Mean top-3 energy share printed.
- Highest-concentration modules printed.
- `Warnings: none` unless unexpected incomplete/mismatched pairs appear.
- JSON log path printed.
- CSV path printed.
- Plot path printed.

Expected output files after running:

- `logs/spectral_stats_<timestamp>.json`
- `outputs/spectral_stats.csv`
- `reports/figures/singular_value_spectra_by_module.png`

Decision / next step:

- Not enough to proceed to clean-reference comparison until the spectral stats
  script has been run and its outputs reviewed.
- If spectral stats complete cleanly, next step should be clean-reference or
  clean-calibration comparison setup, not full baseline reproduction yet.
- Baseline reproduction requires model loading/inference and should remain
  behind explicit low-memory scripts and smoke tests.

## 2026-05-17T14:54:22Z Spectral Stats Completed On Server

Scope of this step: adapter-only CPU compact spectral analysis. No full Llama-2
model was loaded. No inference was run. No GPU-heavy code was run. No
BackdoorLLM evaluation was run. No sanitisation was implemented.

Command run by user on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/02_extract_spectral_stats.py
```

Server terminal output summary:

- Adapter snapshot:
  `/home/huggingface/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b`
- A/B pairs processed: `224`
- Rank values: `[8]`
- Mean top-1 energy share: `0.718263`
- Mean top-3 energy share: `0.902279`
- Warnings: `none`

Highest concentration modules from server output:

- layer `1`, target `down_proj`, top1 `0.997012`, top3 `0.999310`,
  max singular value `0.438961`
- layer `30`, target `gate_proj`, top1 `0.986115`, top3 `0.995720`,
  max singular value `2.135873`
- layer `30`, target `up_proj`, top1 `0.980528`, top3 `0.990659`,
  max singular value `1.425180`
- layer `31`, target `q_proj`, top1 `0.977472`, top3 `0.990559`,
  max singular value `0.881578`
- layer `21`, target `gate_proj`, top1 `0.975295`, top3 `0.990485`,
  max singular value `1.897132`

Expected output files from server run:

- `logs/spectral_stats_20260517T145251Z.json`
- `outputs/spectral_stats.csv`
- `reports/figures/singular_value_spectra_by_module.png`

Local verification status:

- The current local workspace was checked after the server output was provided.
- The new spectral output files were not visible locally yet.
- `rg --files` found only `scripts/02_extract_spectral_stats.py` for
  `spectral_stats` / `singular_value_spectra`.
- Local artifact verification is pending until the server-generated files are
  pulled/synced into this workspace.

Decision:

- Based on the server terminal output, the compact spectral analysis completed
  successfully.
- Before using these results for the next implementation step, verify the JSON,
  CSV, and plot locally after sync.
- Next conceptual step remains clean-reference / clean-calibration comparison
  setup, not baseline reproduction.

## 2026-05-17T15:01:07Z Spectral Artifacts Verified Locally

The server-generated spectral artifacts were pulled into the local workspace and
verified directly.

Verified files:

- `logs/spectral_stats_20260517T145251Z.json`
- `outputs/spectral_stats.csv`
- `reports/figures/singular_value_spectra_by_module.png`

Direct verification results:

- JSON record count: `224`
- CSV row count: `224`
- A/B pairs processed: `224`
- Rank values: `8`
- Warnings count: `0`
- PEFT type: `LORA`
- Base model recorded in adapter config: `meta-llama/Llama-2-7b-chat-hf`
- Config rank: `8`
- LoRA alpha: `16`
- Mean top-1 energy share: `0.718263`
- Mean top-3 energy share: `0.902279`
- Target module counts:
  - `down_proj`: `32`
  - `gate_proj`: `32`
  - `k_proj`: `32`
  - `o_proj`: `32`
  - `q_proj`: `32`
  - `up_proj`: `32`
  - `v_proj`: `32`
- Plot is a valid PNG:
  - dimensions: `1800 x 1100`
  - size: `196694` bytes

Top concentration modules verified from JSON/CSV:

- layer `1`, target `down_proj`, top1 `0.997012`, top3 `0.999310`
- layer `30`, target `gate_proj`, top1 `0.986115`, top3 `0.995720`
- layer `30`, target `up_proj`, top1 `0.980528`, top3 `0.990659`
- layer `31`, target `q_proj`, top1 `0.977472`, top3 `0.990559`
- layer `21`, target `gate_proj`, top1 `0.975295`, top3 `0.990485`

Decision:

- Spectral stats are locally verified and usable for the next implementation
  step.
- Next step should be clean-reference / clean-calibration comparison setup.
- Do not start baseline reproduction yet; that will require explicit low-memory
  model-loading and inference smoke tests.

## 2026-05-17T15:10:13Z Clean Reference Search Script Added

Scope of this step: clean-reference search planning and adapter/cache/metadata
script only. No base model was loaded. No inference was run. No GPU code was
run. No adapter or model was downloaded by Codex.

Files created/modified:

- Created `scripts/03_clean_reference_search.py`.
- Appended this section to `status.md`.

Hugging Face metadata search performed with the Hugging Face connector:

- Search for `BackdoorLLM` clean Llama-2 companion adapters found no obvious
  clean reference.
- Search for `BackdoorLLM/Jailbreak_Llama2-7B` found only attack-family
  adapters:
  - `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
  - `BackdoorLLM/Jailbreak_Llama2-7B_VPI`
  - `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper`
  - `BackdoorLLM/Jailbreak_Llama2-7B_MTBA`
  - `BackdoorLLM/Jailbreak_Llama2-7B_CTBA`
- Community Llama-2 chat LoRA candidates found from metadata include:
  - `Guilherme34/Jennifer2-ENGLISH-CHAT.MULTITURN-LORA-7B-LLAMA2`
  - `Aspik101/Llama-2-7b-chat-hf-pl-lora_adapter_model`
  - `Lajonbot/Llama-2-7b-chat-hf-instruct-pl-lora_adapter_model`
  - `Sparticle/llama-2-7b-chat-japanese-lora`
  - `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview`

Current best provisional candidate:

- `Guilherme34/Jennifer2-ENGLISH-CHAT.MULTITURN-LORA-7B-LLAMA2`
- Reason: metadata tags it as `peft`, and the repo name suggests an English
  multi-turn Llama-2 7B LoRA.
- Caveat: still unconfirmed until `adapter_config.json`,
  `adapter_model.safetensors`, base model compatibility, PEFT type, and target
  modules are verified by the new script.

What `scripts/03_clean_reference_search.py` does:

- Scans local Hugging Face cache roots for adapter snapshots.
- Identifies snapshots with `adapter_config.json`,
  `adapter_model.safetensors`, or `adapter_model.bin`.
- Reads cached adapter configs only when present locally.
- Optionally queries Hugging Face model metadata.
- Does not download remote config files unless `--fetch-remote-configs` is
  explicitly passed.
- Does not download models or load any model.
- Scores candidates based on:
  - Llama-2-7B-Chat compatibility
  - adapter config presence
  - adapter model file presence
  - PEFT type `LORA`
  - target-module overlap with `q_proj`, `k_proj`, `v_proj`, `o_proj`,
    `gate_proj`, `up_proj`, `down_proj`
  - attack/backdoor-like exclusion markers
- Writes CSV to `outputs/clean_reference_candidates.csv`.
- Writes JSON log to `logs/clean_reference_search_<timestamp>.json`.
- Backs up an existing fixed CSV before writing a new one.

Validation performed:

- Syntax-only AST parse passed for `scripts/03_clean_reference_search.py`.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/03_clean_reference_search.py --fetch-remote-configs
```

Expected output:

- Candidate count printed.
- Whether clean-reference comparison is possible printed.
- Best candidate printed if any candidate has suitable adapter files/config.
- CSV path printed:
  `outputs/clean_reference_candidates.csv`
- JSON log path printed:
  `logs/clean_reference_search_<timestamp>.json`
- Next script name printed, if a candidate is confirmed:
  `scripts/04_compare_clean_vs_backdoor_spectra.py`

Latest expected output paths after running:

- `outputs/clean_reference_candidates.csv`
- `logs/clean_reference_search_<timestamp>.json`

Clean-reference comparison status:

- Not confirmed yet.
- If the script confirms a cached or downloadable clean LoRA adapter with
  compatible base model, PEFT type `LORA`, and overlapping target modules, then
  proceed to `scripts/04_compare_clean_vs_backdoor_spectra.py`.
- If no suitable clean adapter is confirmed, fallback is to document that no
  clean reference was found and avoid making backdoor-specific spectral claims
  from the backdoored adapter alone.
