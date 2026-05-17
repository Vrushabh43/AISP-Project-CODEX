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

## 2026-05-17T15:14:23Z Clean Reference Search Outputs Verified

The server-generated clean-reference search artifacts were pulled into the
local workspace and verified directly.

Verified files:

- `logs/clean_reference_search_20260517T151217Z.json`
- `outputs/clean_reference_candidates.csv`

Direct verification results:

- Candidate count: `25`
- Hugging Face metadata errors: `0`
- Script-reported clean-reference comparison possible: `True`
- Script-reported best candidate:
  `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview`
- Important manual review: the script-reported best candidate is **not ideal**
  because it is an LLaVA/multimodal LoRA, rank `64`, and uses
  `adapter_model.bin` rather than `adapter_model.safetensors`.

Top candidates by score:

- `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
  - score `15`
  - base model `meta-llama/Llama-2-7b-chat-hf`
  - PEFT type `LORA`
  - rank `8`
  - full target-module overlap
  - has `adapter_model.bin`, not safetensors
  - caveat: Chinese chat adapter, not an English/alpaca-style clean reference
- `davidkim205/komt-Llama-2-7b-chat-hf-lora`
  - score `15`
  - base model `davidkim205/komt-Llama-2-7b-chat-hf`
  - PEFT type `LORA`
  - rank `8`
  - full target-module overlap
  - has `adapter_model.bin`, not safetensors
  - caveat: base model is not exactly `meta-llama` / `NousResearch`
- `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview`
  - score `15`
  - base path `./checkpoints/llama_2/llama-2-7b-chat`
  - PEFT type `LORA`
  - rank `64`
  - full target-module overlap
  - has `adapter_model.bin`, not safetensors
  - caveat: multimodal LLaVA LoRA, not a clean text-only instruction adapter
- `Guilherme34/Jennifer2-ENGLISH-CHAT.MULTITURN-LORA-7B-LLAMA2`
  - score `10`
  - base model `meta-llama/Llama-2-7b-chat-hf`
  - PEFT type `LORA`
  - rank `8`
  - target overlap only `q_proj,v_proj`
  - has `adapter_model.bin`, not safetensors
  - caveat: English text LoRA, but limited target-module overlap

Security/reproducibility caveat:

- No non-attack clean candidate with `adapter_model.safetensors` was found in
  the current search output.
- The clean candidates use `adapter_model.bin`, which generally requires
  PyTorch pickle deserialization. Treat this as less desirable for an AI
  security project unless we explicitly accept the risk or add a safer handling
  path.

Attack-family adapters correctly excluded:

- `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
- `BackdoorLLM/Jailbreak_Llama2-7B_VPI`
- `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper`
- `BackdoorLLM/Jailbreak_Llama2-7B_MTBA`
- `BackdoorLLM/Jailbreak_Llama2-7B_CTBA`

Decision:

- A perfect clean reference has not been confirmed.
- A limited/provisional clean-vs-backdoor spectral comparison is possible only
  after selecting, downloading, and inspecting a clean adapter candidate.
- Preferred provisional candidate for structural spectral comparison:
  `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA` because it matches
  `meta-llama/Llama-2-7b-chat-hf`, rank `8`, and all seven target module types.
- Preferred provisional candidate for English text/reference relevance:
  `Guilherme34/Jennifer2-ENGLISH-CHAT.MULTITURN-LORA-7B-LLAMA2`, but it only
  targets `q_proj` and `v_proj`, so comparison would be limited to attention
  projections.
- Do not implement clean-vs-backdoor comparison until the user chooses a clean
  candidate and we inspect its adapter files.

## 2026-05-17T15:25:21Z FlagAlpha Safety Gate Script Added

Scope of this step: prepare safe inspection/download of one provisional clean
adapter only: `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`. No clean-vs-backdoor
comparison implemented. No base model loading. No inference. No GPU use.

Files created/modified:

- Created `scripts/04_inspect_flagalpha_clean_adapter.py`.
- Appended this section to `status.md`.

Hugging Face metadata check:

- Repo: `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- Task: question-answering
- Tags include: `transformers`, `zh`, `en`, `license:apache-2.0`
- Metadata does not make it a perfect reference; it remains provisional.

What the script does:

- Targets only `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`.
- Uses `snapshot_download` only with explicit `--download`.
- Download allowlist:
  - `adapter_config.json`
  - `adapter_model.bin`
  - `adapter_model.safetensors`
  - `README.md`
  - `.gitattributes`
- Does not download or load the base Llama-2 model.
- Does not run inference.
- Does not use GPU.
- Queries metadata unless `--offline` is passed.
- Locates the cached snapshot.
- Reads `adapter_config.json` if present.
- Verifies `adapter_model.bin` only with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

- Does **not** fall back to unsafe `weights_only=False`.
- If `weights_only=True` fails, the script exits nonzero and the candidate
  should be rejected or handled manually.
- If `weights_only=True` succeeds, the script checks whether the object is a
  tensor state dict, counts LoRA A/B tensors, groups complete A/B pairs, and
  records target module hints/ranks.
- Writes JSON log to `logs/flagalpha_clean_adapter_inspection_<timestamp>.json`.
- Writes tensor CSV to `outputs/flagalpha_adapter_tensor_summary.csv` if tensor
  rows are available.
- Backs up any existing fixed CSV before writing a new one.

Validation performed:

- Syntax-only AST parse passed for
  `scripts/04_inspect_flagalpha_clean_adapter.py`.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/04_inspect_flagalpha_clean_adapter.py --download --inspect-bin
```

Expected output:

- Metadata reachable: `True`
- `adapter_config.json exists: True`
- `adapter_model.bin exists: True`
- `adapter_model.safetensors exists: False` unless the repo has changed
- PEFT type: `LORA`
- Base model: ideally `meta-llama/Llama-2-7b-chat-hf`
- Rank: ideally `8`
- `.bin inspected with weights_only=True: True`
- `.bin weights_only load ok: True` if safe tensor-state-dict loading works
- `.bin can be read as tensor weights safely: True` if all safety checks pass
- Complete A/B pairs and target module hints printed

Expected output files after running:

- `logs/flagalpha_clean_adapter_inspection_<timestamp>.json`
- `outputs/flagalpha_adapter_tensor_summary.csv`

Decision status:

- Do not proceed to clean-vs-backdoor spectral comparison until this script
  confirms that FlagAlpha's `.bin` can be read with `weights_only=True` and has
  complete LoRA A/B pairs.
- If safe read fails, do not use unsafe pickle loading; either choose a
  different candidate or document that no safe clean reference was available.

## 2026-05-17T15:30:59Z FlagAlpha Clean Adapter Safety Gate Passed

Scope of this step: inspect/download one provisional clean adapter only. No
base model was loaded. No inference was run. No GPU-heavy code was run. No
clean-vs-backdoor comparison was implemented or run.

Command run by user on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/04_inspect_flagalpha_clean_adapter.py --download --inspect-bin
```

Verified files:

- `logs/flagalpha_clean_adapter_inspection_20260517T152925Z.json`
- `outputs/flagalpha_adapter_tensor_summary.csv`

Verification results:

- Repo: `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- Snapshot:
  `/home/huggingface/hub/models--FlagAlpha--Llama2-Chinese-7b-Chat-LoRA/snapshots/9f68976884134bdc3d03620e6d3e42886d9c0561`
- Metadata reachable: `True`
- `adapter_config.json` exists: `True`
- `adapter_model.bin` exists: `True`
- `adapter_model.safetensors` exists: `False`
- `adapter_model.bin` size: `38.192` MB
- PEFT type: `LORA`
- Base model: `meta-llama/Llama-2-7b-chat-hf`
- Rank `r`: `8`
- LoRA alpha: `32`
- Target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`, `gate_proj`, `up_proj`
- `.bin` inspected with `torch.load(..., weights_only=True)`: `True`
- `.bin` weights-only load ok: `True`
- `.bin` can be read as tensor weights safely: `True`
- Tensor count: `448`
- LoRA A tensors: `224`
- LoRA B tensors: `224`
- Complete A/B pairs: `224`
- Incomplete A/B pairs: `0`
- Unique ranks: `8`
- Tensor dtype: `torch.bfloat16`
- Target module hints:
  `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj`
- `.bin` inspection error: none

Important caveats:

- FlagAlpha is a provisional clean structural reference, not a perfect matched
  clean reference.
- It is Chinese/English question-answering/chat, not the same training
  distribution as BackdoorLLM BadNets.
- It uses `adapter_model.bin`, but the safety gate passed because
  `weights_only=True` loaded a tensor-only state dict.
- Its tensor dtype is `bfloat16`; the spectral code converts tensors to
  float32 before SVD, so this is acceptable for numerical comparison.

Decision:

- It is now safe to implement adapter-only clean-vs-backdoor spectral comparison
  using FlagAlpha as a **provisional structural clean reference**.
- The comparison must stay CPU-only and adapter-only.
- Do not load the full base model, run inference, or make backdoor-specific
  research claims from this comparison alone.

## 2026-05-17T15:47:12Z Alternative Clean Candidate Inspection Script Added

Scope of this step: create a safe generic inspection script for two alternative
clean LoRA candidates only. No base model was loaded. No inference was run. No
GPU code was run. No clean-vs-backdoor spectral comparison was implemented.

Files created/modified:

- Created `scripts/05_inspect_clean_adapter_candidates.py`.
- Appended this section to `status.md`.

Candidates targeted by the new script:

- `manojpatil/llama-2-7b-chat-lora-adaptor`
- `Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br`

Comparison reference included when possible:

- `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- The script first tries to load FlagAlpha from the existing verified
  `logs/flagalpha_clean_adapter_inspection_*.json` safety-gate log, so it does
  not need to redownload or reinspect FlagAlpha.

What the script does:

- Optionally downloads only allowlisted adapter files:
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `adapter_model.bin`
  - `README.md`
  - `.gitattributes`
- Does not download or load any base model.
- Does not run inference.
- Does not use GPU.
- Reads `adapter_config.json` when available.
- Inspects `adapter_model.safetensors` directly if present.
- Inspects `adapter_model.bin` only when `--inspect-bin` is passed and only
  with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

- Does not use unsafe fallback loading.
- Marks a candidate unsafe if `weights_only=True` fails or if the loaded object
  is not a tensor state dict.
- Reports PEFT/config fields, LoRA A/B counts, complete/incomplete pairs,
  inferred ranks, dtypes, target-module hints, required target-module overlap,
  safety notes, and a structural suitability score.
- Writes a timestamped JSON log to
  `logs/clean_adapter_candidate_inspection_<timestamp>.json`.
- Writes a comparison CSV to
  `outputs/clean_adapter_candidate_comparison.csv`.
- Writes per-candidate tensor CSVs when tensor rows are available:
  `outputs/clean_candidate_<safe_repo_name>_tensor_summary.csv`.
- Backs up existing fixed CSV outputs before writing replacements.

Validation performed:

- Syntax-only AST parse passed for
  `scripts/05_inspect_clean_adapter_candidates.py`.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/05_inspect_clean_adapter_candidates.py --download --inspect-bin
```

Expected output:

- Both requested alternative repo IDs are listed.
- Each candidate reports whether metadata was reachable.
- Each candidate reports whether `adapter_config.json`,
  `adapter_model.safetensors`, and/or `adapter_model.bin` exists.
- For `.bin` candidates, output should show that inspection used
  `weights_only=True`.
- If safe, each candidate should report:
  - `safe weights: True`
  - LoRA A/B counts
  - complete A/B pair count
  - rank values
  - target-module hints
- The summary should list:
  - safe-loading candidates
  - rank-8 candidates
  - full 7-target-module candidates
  - safetensors vs `.bin` candidates
  - recommended best clean reference candidate
  - whether it is safe to proceed to clean-vs-backdoor spectral comparison

Known prior metadata before this safety inspection:

- `manojpatil/llama-2-7b-chat-lora-adaptor` looked like a compatible LoRA using
  `adapter_model.bin`, base `NousResearch/Llama-2-7b-chat-hf`, rank `64`, and
  only `q_proj,v_proj` targets.
- `Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br` looked like a compatible LoRA
  using `adapter_model.bin`, base `meta-llama/Llama-2-7b-chat-hf`, rank `8`,
  and only `q_proj,v_proj` targets.
- These metadata hints are not final; the new script must verify actual
  adapter files safely.

Latest expected output files after running:

- `logs/clean_adapter_candidate_inspection_<timestamp>.json`
- `outputs/clean_adapter_candidate_comparison.csv`
- `outputs/clean_candidate_manojpatil_llama_2_7b_chat_lora_adaptor_tensor_summary.csv`
- `outputs/clean_candidate_Luciano_lora_4bit_Llama_2_7b_chat_hf_lener_br_tensor_summary.csv`
- If FlagAlpha tensor rows are loaded from the existing log:
  `outputs/clean_candidate_FlagAlpha_Llama2_Chinese_7b_Chat_LoRA_tensor_summary.csv`

Current blockers:

- Need the server run output and generated JSON/CSV files before making the
  final clean-reference decision.
- Do not proceed to clean-vs-backdoor spectral comparison until the two
  alternative candidates have passed or failed the `.bin` safety gate.

Next suggested step:

- Run `scripts/05_inspect_clean_adapter_candidates.py --download --inspect-bin`
  on the server.
- Pull/sync the generated log and CSV files into the local workspace.
- Review the actual safety and structural results.
- If FlagAlpha remains the best structural reference, implement
  `scripts/04_compare_clean_vs_backdoor_spectra.py` as an adapter-only CPU
  comparison using the selected clean reference.

## 2026-05-17T15:50:35Z Alternative Clean Candidate Inspection Verified

Scope of this step: verify server-generated outputs from
`scripts/05_inspect_clean_adapter_candidates.py --download --inspect-bin`.
No base model was loaded. No inference was run. No GPU-heavy code was run. No
clean-vs-backdoor spectral comparison was implemented or run.

Command run by user on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/05_inspect_clean_adapter_candidates.py --download --inspect-bin
```

Verified output files:

- `logs/clean_adapter_candidate_inspection_20260517T154855Z.json`
- `outputs/clean_adapter_candidate_comparison.csv`
- `outputs/clean_candidate_manojpatil_llama_2_7b_chat_lora_adaptor_tensor_summary.csv`
- `outputs/clean_candidate_Luciano_lora_4bit_Llama_2_7b_chat_hf_lener_br_tensor_summary.csv`
- `outputs/clean_candidate_FlagAlpha_Llama2_Chinese_7b_Chat_LoRA_tensor_summary.csv`

Direct verification results:

- JSON result count: `3`
- Recommendation in JSON:
  `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- `safe_to_proceed_to_spectral_comparison`: `True`
- Tensor CSV row counts:
  - `manojpatil/llama-2-7b-chat-lora-adaptor`: `128`
  - `Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br`: `128`
  - `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`: `448`

Candidate safety and structure:

- `manojpatil/llama-2-7b-chat-lora-adaptor`
  - metadata reachable: `True`
  - base model: `NousResearch/Llama-2-7b-chat-hf`
  - PEFT type: `LORA`
  - task type: `CAUSAL_LM`
  - `adapter_model.bin` exists, no safetensors
  - `.bin` inspected with `weights_only=True`: `True`
  - safe weights: `True`
  - rank values: `[64]`
  - target modules/hints: `q_proj,v_proj`
  - full seven target-module overlap: `False`
  - complete A/B pairs: `64`
  - incomplete A/B pairs: `0`
  - dtype: `torch.float32`
  - structural suitability score: `14`

- `Luciano/lora-4bit-Llama-2-7b-chat-hf-lener_br`
  - metadata reachable: `True`
  - base model: `meta-llama/Llama-2-7b-chat-hf`
  - PEFT type: `LORA`
  - task type: `CAUSAL_LM`
  - `adapter_model.bin` exists, no safetensors
  - `.bin` inspected with `weights_only=True`: `True`
  - safe weights: `True`
  - rank values: `[8]`
  - target modules/hints: `q_proj,v_proj`
  - full seven target-module overlap: `False`
  - complete A/B pairs: `64`
  - incomplete A/B pairs: `0`
  - dtype: `torch.float32`
  - structural suitability score: `16`

- `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
  - loaded as comparison reference from existing verified FlagAlpha log
  - base model: `meta-llama/Llama-2-7b-chat-hf`
  - PEFT type: `LORA`
  - task type: `CAUSAL_LM`
  - `adapter_model.bin` exists, no safetensors
  - `.bin` inspected with `weights_only=True`: `True`
  - safe weights: `True`
  - rank values: `[8]`
  - target modules/hints:
    `down_proj,gate_proj,k_proj,o_proj,q_proj,up_proj,v_proj`
  - full seven target-module overlap: `True`
  - complete A/B pairs: `224`
  - incomplete A/B pairs: `0`
  - dtype: `torch.bfloat16`
  - structural suitability score: `18`

Decision:

- All three clean candidates passed safe tensor-weight loading.
- None of the three uses `adapter_model.safetensors`; all use
  `adapter_model.bin`, but each was inspected only with
  `torch.load(..., map_location="cpu", weights_only=True)`.
- `manojpatil` is not a good structural reference for the full adapter because
  it has rank `64` and only `q_proj,v_proj`.
- `Luciano` is useful as an exact-base rank-8 attention-only candidate, but it
  only covers `q_proj,v_proj`.
- `FlagAlpha` remains the best provisional structural clean reference because
  it is rank `8`, base-compatible with `meta-llama/Llama-2-7b-chat-hf`, and has
  all seven target module types with `224` complete A/B pairs.
- Clean-reference caveat remains: FlagAlpha is not a perfectly matched clean
  training distribution because it is Chinese/English QA/chat. Use it for a
  structural spectral sanity comparison only, not final research claims.

Next suggested step:

- Implement `scripts/04_compare_clean_vs_backdoor_spectra.py` as an adapter-only
  CPU comparison using:
  - backdoored adapter:
    `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
  - provisional clean structural reference:
    `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
- The comparison should compute the same compact spectral stats for both
  adapters and compare only structural/spectral distributions. It must not load
  the full base model, run inference, or make backdoor-specific success claims.

## 2026-05-17T16:00:03Z Clean-vs-Backdoor Spectral Comparison Script Added

Scope of this step: implement adapter-only CPU clean-vs-backdoor spectral
comparison. No full Llama-2 model was loaded. No inference was run. No GPU code
was run. No BackdoorLLM evaluation was run. No sanitisation was implemented.

Files created/modified:

- Created `scripts/06_compare_clean_vs_backdoor_spectra.py`.
- Appended this section to `status.md`.

Comparison implemented:

- Backdoored adapter:
  `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`
  - loads `adapter_model.safetensors`
  - rank `8`
  - alpha `16`
  - expected complete A/B pairs: `224`
- Clean structural reference:
  `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA`
  - loads `adapter_model.bin` only with
    `torch.load(..., map_location="cpu", weights_only=True)`
  - no unsafe fallback loading
  - rank `8`
  - alpha `32`
  - expected complete A/B pairs: `224`

What the script does:

- Locates cached snapshots for both adapters.
- Reads `adapter_config.json` for both adapters.
- Groups LoRA A/B pairs by `(layer_id, target_module)`.
- Matches common modules between adapters.
- Computes compact singular values with the existing QR + small-SVD method.
- Does not form dense `Delta W`.
- Computes per-adapter metrics:
  - top-1 energy share
  - top-3 energy share
  - normalized spectral entropy
  - effective rank
  - normalized singular values
  - Frobenius norm
  - max singular value
- Focuses GO/NO-GO and plots on normalized spectral metrics because
  BackdoorLLM uses `lora_alpha=16` while FlagAlpha uses `lora_alpha=32`.
- Writes timestamped JSON logs and fixed CSV/figure outputs with timestamped
  backups if fixed output paths already exist.

Validation performed:

- Syntax-only AST parse passed for
  `scripts/06_compare_clean_vs_backdoor_spectra.py`.
- Static safety check confirmed the only `torch.load` call uses
  `map_location="cpu", weights_only=True`.
- The script was not run locally because the authoritative adapter cache paths
  are on the server.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/06_compare_clean_vs_backdoor_spectra.py
```

Expected output:

- matched module count, ideally `224`
- unmatched backdoor modules, ideally `0`
- unmatched clean modules, ideally `0`
- mean backdoor top1 vs clean top1
- mean backdoor top3 vs clean top3
- mean backdoor entropy vs clean entropy
- module types where backdoor is more concentrated than clean under the
  conservative `delta_top1 > 0` and `delta_entropy < 0` criterion
- GO/NO-GO spectral sanity check result
- caveat that FlagAlpha is a structural, not task/distribution-matched, clean
  reference

Expected output files after running:

- `logs/clean_vs_backdoor_spectral_comparison_<timestamp>.json`
- `outputs/clean_vs_backdoor_spectral_comparison.csv`
- `reports/figures/clean_vs_backdoor_top1_by_layer.png`
- `reports/figures/clean_vs_backdoor_top3_by_layer.png`
- `reports/figures/clean_vs_backdoor_entropy_by_layer.png`
- `reports/figures/clean_vs_backdoor_singular_curve_mean.png`

GO/NO-GO status:

- Pending. The script has been implemented but not run in the server
  environment yet.
- The script will report
  `spectral_sanity_check_supports_continuing: True` only if all expected
  structural modules match and at least one target module shows higher
  backdoor top-1 energy with lower entropy than the clean structural reference.
- Even if GO is reported, this remains a spectral sanity check only and must
  not be written as a final backdoor-specific research claim.

Next suggested step:

- Run `scripts/06_compare_clean_vs_backdoor_spectra.py` on the server.
- Pull/sync the JSON, CSV, and four figure files into the local workspace.
- Verify outputs before deciding whether to proceed to sanitisation design or
  to adjust the clean-reference comparison.

## 2026-05-17T16:04:22Z Clean-vs-Backdoor Spectral Comparison Verified

Scope of this step: verify server-generated adapter-only CPU spectral
comparison artifacts. No full Llama-2 model was loaded. No inference was run.
No GPU-heavy code was run. No BackdoorLLM evaluation was run. No sanitisation
was implemented.

Command run by user on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/06_compare_clean_vs_backdoor_spectra.py
```

Verified output files:

- `logs/clean_vs_backdoor_spectral_comparison_20260517T160214Z.json`
- `outputs/clean_vs_backdoor_spectral_comparison.csv`
- `reports/figures/clean_vs_backdoor_top1_by_layer.png`
- `reports/figures/clean_vs_backdoor_top3_by_layer.png`
- `reports/figures/clean_vs_backdoor_entropy_by_layer.png`
- `reports/figures/clean_vs_backdoor_singular_curve_mean.png`

Direct verification results:

- JSON matched module count: `224`
- CSV row count: `224`
- Unmatched backdoor modules: `0`
- Unmatched clean modules: `0`
- Backdoor pair count in log: `224`
- Clean reference pair count in log: `224`
- Backdoor warnings: `0`
- Clean reference warnings: `0`
- CSV target modules and counts:
  - `down_proj`: `32`
  - `gate_proj`: `32`
  - `k_proj`: `32`
  - `o_proj`: `32`
  - `q_proj`: `32`
  - `up_proj`: `32`
  - `v_proj`: `32`
- Layer range: `0` to `31`
- Required CSV fields are present, including:
  - `layer_id`
  - `target_module`
  - `backdoor_top1`
  - `clean_top1`
  - `delta_top1`
  - `backdoor_top3`
  - `clean_top3`
  - `delta_top3`
  - `backdoor_entropy`
  - `clean_entropy`
  - `delta_entropy`
  - `backdoor_effective_rank`
  - `clean_effective_rank`
  - `delta_effective_rank`
  - `backdoor_fro_norm`
  - `clean_fro_norm`
  - `norm_comparison_note`
- Figure files are valid PNGs:
  - `clean_vs_backdoor_top1_by_layer.png`: `2000x960`
  - `clean_vs_backdoor_top3_by_layer.png`: `2000x960`
  - `clean_vs_backdoor_entropy_by_layer.png`: `2000x960`
  - `clean_vs_backdoor_singular_curve_mean.png`: `1600x960`

Main normalized spectral results:

- Mean backdoor top-1 energy share: `0.718263`
- Mean clean top-1 energy share: `0.566749`
- Mean delta top-1: `0.151514`
- Mean backdoor top-3 energy share: `0.902279`
- Mean clean top-3 energy share: `0.841243`
- Mean delta top-3: `0.061036`
- Mean backdoor spectral entropy: `0.461946`
- Mean clean spectral entropy: `0.628052`
- Mean delta entropy: `-0.166106`

Module types where the backdoor adapter is more concentrated than the clean
structural reference under the conservative criterion
`delta_top1 > 0` and `delta_entropy < 0`:

- `down_proj`
- `gate_proj`
- `o_proj`
- `q_proj`
- `up_proj`
- `v_proj`

GO/NO-GO spectral sanity decision:

- `spectral_sanity_check_supports_continuing`: `True`
- Reason: all expected modules matched, and at least one target module shows
  higher backdoor top-1 energy with lower entropy than the clean structural
  reference.

Important caveats:

- This is a structural spectral sanity check only.
- FlagAlpha is not a perfect task/distribution-matched clean reference because
  it is Chinese/English QA/chat.
- Raw Frobenius norms and max singular values are not primary evidence because
  BackdoorLLM uses `lora_alpha=16` while FlagAlpha uses `lora_alpha=32`.
- Do not write final research claims from this comparison alone.

Decision:

- Week-2 spectral GO/NO-GO sanity check passes.
- It is reasonable to proceed to adapter-only sanitisation design and
  implementation next.
- The next implementation step should still avoid full model loading and should
  focus on producing edited LoRA adapter files from spectral rules before any
  inference/evaluation scripts.

## 2026-05-17T16:10:46Z Spectral-only Sanitised Adapter Generation Added

Scope of this step: implement adapter-only CPU sanitisation utilities and a
first spectral-only sanitised adapter generation script. No full Llama-2 model
was loaded. No inference was run. No GPU code was run. No BackdoorLLM
evaluation was run. Clean-prompt sensitivity was not implemented yet.

Files created/modified:

- Modified `src/lora_sanitise/attenuation.py`.
- Modified `src/lora_sanitise/svd_tools.py`.
- Created `scripts/07_generate_spectral_sanitised_adapters.py`.
- Appended this section to `status.md`.

Implemented in `src/lora_sanitise/attenuation.py`:

- `select_top_spectral_components(S, k)`
- `attenuate_singular_values(S, selected_indices, gamma)`
- `refactor_svd_to_lora_A_B(U, S_new, Vh)`
- `reconstruction_error(A_new, B_new, delta_w_target)`
- `validate_lora_factor_shapes(A, B)`
- Finite-value checks for NaN/Inf and shape/rank consistency.

Implemented in `src/lora_sanitise/svd_tools.py`:

- `compact_svd_full_for_pair(A, B)`
- This computes compact factors `U, S, Vh` for `Delta W = B @ A` using QR plus
  an `r x r` SVD, without forming dense `Delta W`.

Implemented in `scripts/07_generate_spectral_sanitised_adapters.py`:

- Loads only the cached BackdoorLLM adapter:
  `BackdoorLLM/Jailbreak_Llama2-7B_BadNets`.
- Reads `adapter_model.safetensors` and `adapter_config.json`.
- Groups complete LoRA A/B pairs.
- For each pair, computes compact SVD without dense `Delta W`.
- Generates six spectral-only variants:
  - `top1_gamma_0.0`
  - `top1_gamma_0.25`
  - `top1_gamma_0.50`
  - `top3_gamma_0.0`
  - `top3_gamma_0.25`
  - `top3_gamma_0.50`
- Refactors edited singular spectra back into PEFT-compatible LoRA A/B tensors
  with the original tensor shapes and dtypes.
- Preserves `adapter_config.json` unchanged.
- Saves each variant under:
  `outputs/sanitised_adapters/<variant_name>/`
- Writes each variant's:
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `sanitisation_report.json`
- Writes:
  - `logs/sanitised_adapter_generation_<timestamp>.json`
  - `outputs/sanitised_adapter_generation_summary.csv`
- Backs up existing fixed variant folders and summary CSV paths with
  timestamped backups before writing replacements.
- Dense reconstruction checks are optional and disabled by default via
  `--max-dense-check-elements 0` to avoid large allocations.

Validation performed:

- Syntax-only AST parse passed for:
  - `src/lora_sanitise/attenuation.py`
  - `src/lora_sanitise/svd_tools.py`
  - `scripts/07_generate_spectral_sanitised_adapters.py`
- Static safety scan found no `torch.load`, no `AutoModel`, and no destructive
  deletion command in the new sanitisation script.
- The script was not run locally because the authoritative adapter cache path
  is on the server.

Exact command to run next on the server:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python scripts/07_generate_spectral_sanitised_adapters.py
```

Expected terminal output:

- Adapter ID and snapshot path.
- A/B pairs processed per variant, expected `224`.
- Variants generated, expected `6`.
- For each variant:
  - selected component count
  - gamma
  - modules edited
  - mean top1 before and after
  - mean entropy before and after
- Pair warnings should be `none`.
- JSON log path and summary CSV path.

Expected output folders/files:

- `outputs/sanitised_adapters/top1_gamma_0.0/adapter_config.json`
- `outputs/sanitised_adapters/top1_gamma_0.0/adapter_model.safetensors`
- `outputs/sanitised_adapters/top1_gamma_0.0/sanitisation_report.json`
- `outputs/sanitised_adapters/top1_gamma_0.25/adapter_config.json`
- `outputs/sanitised_adapters/top1_gamma_0.25/adapter_model.safetensors`
- `outputs/sanitised_adapters/top1_gamma_0.25/sanitisation_report.json`
- `outputs/sanitised_adapters/top1_gamma_0.50/adapter_config.json`
- `outputs/sanitised_adapters/top1_gamma_0.50/adapter_model.safetensors`
- `outputs/sanitised_adapters/top1_gamma_0.50/sanitisation_report.json`
- `outputs/sanitised_adapters/top3_gamma_0.0/adapter_config.json`
- `outputs/sanitised_adapters/top3_gamma_0.0/adapter_model.safetensors`
- `outputs/sanitised_adapters/top3_gamma_0.0/sanitisation_report.json`
- `outputs/sanitised_adapters/top3_gamma_0.25/adapter_config.json`
- `outputs/sanitised_adapters/top3_gamma_0.25/adapter_model.safetensors`
- `outputs/sanitised_adapters/top3_gamma_0.25/sanitisation_report.json`
- `outputs/sanitised_adapters/top3_gamma_0.50/adapter_config.json`
- `outputs/sanitised_adapters/top3_gamma_0.50/adapter_model.safetensors`
- `outputs/sanitised_adapters/top3_gamma_0.50/sanitisation_report.json`
- `logs/sanitised_adapter_generation_<timestamp>.json`
- `outputs/sanitised_adapter_generation_summary.csv`

Current limitations:

- These are spectral-only sanitised adapter variants.
- Clean-prompt sensitivity is not included yet.
- No ASR, clean utility, generation, or model-loading evaluation has been run.
- Dense reconstruction checks are skipped by default for memory safety.

Next step after this script succeeds:

- Verify the generated `sanitisation_report.json` files and summary CSV.
- Inspect that each variant has `224` edited modules, no errors, and expected
  before/after spectral metric changes.
- Then implement a lightweight adapter-load smoke test that loads adapter files
  through PEFT metadata only or a carefully controlled low-memory model-loading
  smoke test, before any ASR or clean-utility evaluation.
