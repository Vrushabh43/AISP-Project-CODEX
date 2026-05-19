# Experiment Summary Template

Use this after every major experiment or smoke test. Keep it concise, factual, and suitable for later report writing.

## Experiment Title

Short descriptive name.

## Date/Time

- Date:
- Start time:
- End time:
- Machine:
- GPU:
- Git state, if relevant:

## Script

- Script path:
- Script version or notes:

## Purpose

What question did this run answer?

What it did not try to answer:

## Inputs

- Base model:
- Adapter(s):
- Prompt file(s) or prompt source:
- Cache path:
- Config:
- Seeds:

## Command

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/path/to/cache
source .venv/bin/activate
python scripts/<script_name>.py
```

## Outputs

- JSON log:
- Details CSV:
- Summary CSV:
- Figures:
- Child logs:

## Key Numbers

- Adapters tested:
- Prompt count:
- Clean prompt count:
- Trigger/attack prompt count:
- Failures:
- OOM count:
- Refusals:
- Main metric(s):

## Errors/Issues

- What failed:
- Error text:
- Root cause:
- Fix or workaround:
- Follow-up needed:

## Decision

Proceed, rerun, debug, or stop?

Rationale:

## Next Step

Immediate next action:

Blocking dependencies:

## Report-Writing Note

How this result should be described in the final report:

What not to claim:
