# Known Issues

This file records practical and scientific caveats that matter for future runs and report writing.

## Copied Hugging Face Cache Symlink And 0-Byte File Issue

Problem:

- A Hugging Face cache copied from `ki-016` to `ki-010` produced broken/corrupted snapshot files.
- Snapshot files such as `config.json`, `tokenizer_config.json`, and safetensors entries appeared as 0-byte regular files.
- Transformers then failed with errors such as:
  `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

Likely cause:

- Hugging Face cache snapshots normally use symlinks into `blobs/`.
- Copying the cache between systems did not preserve the cache structure correctly.

Fix used:

```bash
mkdir -p /home/43e3/hf-cache-aisp
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
python -c "from huggingface_hub import snapshot_download; snapshot_download('NousResearch/Llama-2-7b-chat-hf'); snapshot_download('BackdoorLLM/Jailbreak_Llama2-7B_BadNets')"
```

Current rule:

- On `ki-010`, use:

```bash
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
```

- Do not rely on the corrupted project-root copied cache for model loading.

## GPU Memory Issue On ki-016

Problem:

- Initial 4-bit base attach failed because only about 437 MB of VRAM was free.
- `nvidia-smi` showed most VRAM was occupied by other Python processes.

Fix used:

- Added `scripts/10_gpu_memory_diagnosis.py`.
- Freed GPU memory before retrying.
- Base attach then passed with about 7.2 GB free before loading.

Current rule:

- Before model-loading tests, run:

```bash
python scripts/10_gpu_memory_diagnosis.py
```

- Do not run tiny forward or generation if free VRAM is too low.

## Fresh Hugging Face Cache Required On ki-010

Problem:

- `ki-010` did not share the working cache state from `ki-016`.
- Copied cache was not reliable.

Fix used:

- Created fresh cache:
  `/home/43e3/hf-cache-aisp`

Current rule:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/home/43e3/hf-cache-aisp
source .venv/bin/activate
```

## FlagAlpha Clean Reference Caveat

Problem:

- No perfect task/distribution-matched clean LoRA reference was found.
- `FlagAlpha/Llama2-Chinese-7b-Chat-LoRA` is structurally useful but not a perfect clean reference:
  - Chinese chat adapter
  - uses `adapter_model.bin`
  - LoRA alpha differs from BackdoorLLM adapter

Fix used:

- Loaded `.bin` only with:
  `torch.load(..., map_location="cpu", weights_only=True)`
- Used FlagAlpha only for structural spectral sanity checks.
- Focused on normalized spectral metrics instead of raw magnitude.

Current rule:

- Do not describe FlagAlpha as a matched clean baseline.
- Describe it as a provisional structural clean reference.

## Trigger Prompts Are Currently Unverified

Problem:

- No official BackdoorLLM trigger examples were found in local project files.
- `configs/experiment.yaml` still has `attack_eval_prompts: null`.
- Current trigger-pilot prompts are placeholders.
- `data/eval_prompts/trigger_probe_small_unverified.jsonl` is also a placeholder probe file, not an official trigger set.

Fix used:

- Labeled pilot prompts as `trigger_pilot` and `unverified_trigger_prompt`.
- Added explicit caveats to logs/status/report notes.

Current rule:

- Do not call trigger-pilot results ASR.
- Do not call `trigger_probe_unverified` results ASR.
- Before real ASR, verify trigger format from BackdoorLLM materials or explicitly label the evaluation as unverified-trigger probing.

## Pilot Results Are Not ASR

Problem:

- The small baseline pilot used 5 clean prompts and 5 unverified trigger-pilot prompts per adapter.
- It checked generation plumbing and simple indicators, not final attack behavior.

Current rule:

- Do not claim:
  - ASR reduction
  - clean utility preservation
  - defence success
  - superiority over baselines

Allowed interpretation:

- The pilot confirms that low-memory generation and structured logging work for the original and selected sanitised adapters.

## Simple Keyword Flags Are Crude

Problem:

- The pilot unsafe keyword flag counted the word `harm` in benign safe-AI outputs.

Current rule:

- Treat keyword flags as rough diagnostics only.
- Do not equate keyword hits with unsafe output.
- Real ASR/utility scoring needs explicit rubrics or judge logic.

## Do Not Overwrite Or Delete Caches/Adapters

Current rule:

- Do not delete `.venv`, cache folders, adapter folders, `outputs/`, or `logs/`.
- Do not modify original cached adapters.
- Do not use unsafe `torch.load` for `.bin` adapters.
- Prefer new timestamped output files and CSV backups.
