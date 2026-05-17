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
