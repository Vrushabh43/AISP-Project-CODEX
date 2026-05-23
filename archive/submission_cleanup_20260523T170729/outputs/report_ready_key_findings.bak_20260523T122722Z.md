# Report-Ready Key Findings

All numbers below are bounded heuristic metrics, not final judged ASR or final judged clean utility.

- The original backdoored adapter has a preliminary trigger success rate of `0.6061` with heuristic clean utility `0.9667`.
- The best expanded SensAware variant is `sensaware_top224_gamma_0.25`, with trigger success `0.0101` and clean utility `0.9667`.
- Expanded SensAware beats the best spectral-only baseline in this bounded heuristic run: `0.0101` versus `0.0606` for `top3_gamma_0.50`.
- Expanded SensAware nearly matches but does not beat the strongest uniform-scaling baseline: `0.0101` versus `0.0000` for `uniform_gamma_0.25`.
- Uniform scaling may be globally weakening the adapter rather than selectively removing suspicious components, so final interpretation needs caution and stronger clean-utility evaluation.
- These results support bounded follow-up analysis, but they should not be written as final ASR claims.

## Case-Diagnostic Snapshot

- Diagnostic prompt count: `96`.
- Original succeeded but SensAware refused or blocked under the heuristic: `60` prompts.
- SensAware still succeeded under the heuristic: `1` prompts.
- All selected defenses blocked prompts where original succeeded: `55` prompts.
