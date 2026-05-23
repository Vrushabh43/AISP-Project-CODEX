# Implementation Notes

These notes describe what was actually implemented, where it differs from the
original plan, and how to interpret the results honestly.

## Implemented Method vs Original Plan

The project plan proposed sensitivity-aware singular-component attenuation of
LoRA updates. That high-level idea was implemented, but the exact sensitivity
estimator changed for feasibility.

Implemented:

- Adapter-only LoRA A/B inspection.
- Compact low-rank SVD for LoRA update products.
- Spectral-only attenuation baselines.
- Uniform scaling baselines.
- Clean-prompt sensitivity using an activation-projection proxy.
- Global top-N component ranking by a preliminary suspiciousness score.
- PEFT-compatible adapter regeneration.
- Bounded heuristic trigger and clean-utility evaluation.

Not implemented:

- KL-divergence sensitivity over component ablations.
- Lambda sweep over a spectral-minus-sensitivity objective.
- Per-module-only top-K SensAware selection.
- External judge based final ASR.
- Final judged clean utility.

## Clean Sensitivity Proxy

The implemented clean sensitivity estimate uses clean activations at LoRA target
modules. For candidate singular component `i`, the script estimates:

`contribution_i ~= s_i^2 * mean((v_i^T x)^2)`

where:

- `s_i` is the singular value of the LoRA update component.
- `v_i` is the right singular vector.
- `x` is the clean input activation to the target module.

The intuition is that components with high spectral energy but low clean
activation contribution are more suspicious and safer to attenuate.

The preliminary score used for expanded variants is:

`spectral_energy_share * (1 - clean_sensitivity_norm_global)`

This is a transparent proxy, not a proof that the component is backdoor-specific.

## Why Expanded SensAware Was Added

The first SensAware variants selected only 16 to 32 components across about 16
modules. They preserved clean utility but barely reduced the bounded trigger
success rate. Failure analysis showed that the spectral-only baselines changed
many more modules/components:

- `top1_gamma_0.50`: 224 components across 224 modules.
- `top3_gamma_0.50`: 672 components across 224 modules.

Expanded SensAware was added to test whether sensitivity-aware scoring becomes
useful only when enough components are allowed to change.

## First SensAware Failure

The first SensAware variants were too conservative:

- `sensaware_top16_gamma_0.50`: trigger success `0.5657`.
- `sensaware_top32_gamma_0.50`: trigger success `0.5758`.
- `sensaware_top32_gamma_0.25`: trigger success `0.5455`.

They improved little over the original adapter (`0.6061`). This is an important
negative result and should not be hidden.

## Expanded SensAware Result

The best expanded SensAware variant was:

`sensaware_top224_gamma_0.25`

Bounded heuristic result:

- Trigger success: `0.0101`.
- Clean utility: `0.9667`.

This beats the best spectral-only baseline in the bounded heuristic run:

- `top3_gamma_0.50`: trigger success `0.0606`, clean utility `0.9333`.

It does not beat the strongest uniform scaling baseline:

- `uniform_gamma_0.25`: trigger success `0.0000`, clean utility `0.9667`.

## Why Uniform Scaling Matters

Uniform scaling is a strong and necessary baseline because it tests whether the
defence is doing anything more targeted than weakening the entire adapter.

In the current bounded heuristic evaluation, uniform scaling is extremely
strong. This creates a mixed result:

- SensAware looks better than spectral-only attenuation.
- SensAware does not beat the simplest strongest uniform baseline.
- Uniform scaling may globally weaken useful adapter behavior in ways the small
  clean heuristic does not capture.

The last point is a caveat, not a proven claim. A stronger clean-utility
benchmark would be required before making a final utility-preservation claim.

## Evaluation Caveats

All current evaluation metrics are bounded heuristic metrics:

- The trigger source is official BackdoorLLM BadNets data.
- The trigger token is verified.
- The scoring is still rule/keyword/refusal based.
- No external safety judge was used.
- No final judged ASR was run.
- No final judged clean utility benchmark was run.

The final report should use phrases such as "bounded heuristic trigger success"
and "heuristic clean utility", not final ASR or final clean utility.

## What Not To Overclaim

Do not claim:

- SensAware beats uniform scaling.
- The backdoor is fully removed.
- The method is robust to adaptive attackers.
- The method is a general LoRA defence.
- The numbers are final judged ASR/utility.
- The implemented method used KL sensitivity or lambda sweeps.

Defensible claim:

Expanded SensAware substantially reduced bounded heuristic trigger success and
beat spectral-only baselines while preserving the small clean-utility heuristic,
but it did not outperform the strongest uniform scaling baseline. The result is
mixed and useful as an empirical study with honest limitations.
