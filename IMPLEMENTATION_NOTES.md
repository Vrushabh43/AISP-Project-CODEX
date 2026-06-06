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
- Bounded heuristic trigger and clean-utility evaluation as an intermediate
  stage.
- Verified BackdoorLLM official rule-based jailbreak ASR scoring as the final
  ASR evidence.
- Clean reference-output NLL/perplexity as a stronger clean-utility probe.

Not implemented:

- KL-divergence sensitivity over component ablations.
- Lambda sweep over a spectral-minus-sensitivity objective.
- Per-module-only top-K SensAware selection.
- External-judged final ASR.
- Human-judged final clean utility.

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

Earlier bounded heuristic result:

- Trigger success: `0.0101`.
- Clean utility: `0.9667`.

This beats the best spectral-only baseline in the bounded heuristic run:

- `top3_gamma_0.50`: trigger success `0.0606`, clean utility `0.9333`.

It did not beat the strongest uniform scaling baseline under that older
bounded heuristic:

- `uniform_gamma_0.25`: trigger success `0.0000`, clean utility `0.9667`.

Updated official rule-based ASR result:

- `original`: 36/99 = 0.363636.
- `uniform_gamma_0.25`: 13/99 = 0.131313.
- `uniform_gamma_0.50`: 7/99 = 0.070707.
- `top3_gamma_0.50`: 7/99 = 0.070707.
- `sensaware_top224_gamma_0.25`: 1/99 = 0.010101.

Under the verified BackdoorLLM official rule-based jailbreak ASR scorer applied
to project-local deterministic `[INST]` generations, `sensaware_top224_gamma_0.25`
has the lowest observed ASR count among the focused conditions. Its
clean-reference perplexity is also lowest in this run: `3.557692`.

## Why Uniform Scaling Still Matters

Uniform scaling is a strong and necessary baseline because it tests whether the
defence is doing anything more targeted than weakening the entire adapter.

The older bounded heuristic evaluation made uniform scaling look extremely
strong. The updated official rule-based ASR scorer changes the focused
comparison, but the baseline remains important:

- SensAware looks better than spectral-only attenuation.
- SensAware now has lower observed official rule-based ASR than the two focused
  uniform baselines.
- Uniform scaling may globally weaken useful adapter behavior in ways the small
  clean heuristic does not capture.

The last point remains a caveat, not a proven claim. Clean-reference perplexity
helps, but it is still a reference-likelihood probe rather than final human
utility.

## Evaluation Caveats

The final ASR metric is:

`BackdoorLLM official rule-based jailbreak ASR`

This means:

- The trigger source is official BackdoorLLM BadNets data.
- The trigger token is verified.
- The scorer is the verified BackdoorLLM rule-based refusal-keyword ASR path.
- The scorer is applied to project-local deterministic `[INST]` generations.
- The metric is not external-judged ASR or a human harmfulness judgment.
- No final human clean utility benchmark was run.
- Clean perplexity is a reference-likelihood probe, not final human utility.

The final report should use phrases such as "BackdoorLLM official rule-based
jailbreak ASR scorer applied to project-local deterministic generations" and
"clean-reference perplexity probe." It should not call the results
external-judged ASR or final human clean utility.

## What Not To Overclaim

Do not claim:

- SensAware beats all possible uniform scaling settings.
- The backdoor is fully removed.
- The method is robust to adaptive attackers.
- The method is a general LoRA defence.
- The numbers are external-judged ASR or final human utility.
- The implemented method used KL sensitivity or lambda sweeps.

Defensible claim:

Expanded SensAware substantially reduced official rule-based jailbreak ASR
under the verified BackdoorLLM scorer applied to project-local deterministic
generations. In the focused updated comparison, `sensaware_top224_gamma_0.25`
had the lowest observed ASR count, 1/99, and the lowest clean-reference
perplexity, 3.557692. Wilson intervals over 99 prompts and the rule-based
nature of the metric must be reported.
