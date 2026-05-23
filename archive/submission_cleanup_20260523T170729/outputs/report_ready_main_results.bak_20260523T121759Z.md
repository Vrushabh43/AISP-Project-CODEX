# Report-Ready Main Results

Bounded heuristic metrics only; not final judged ASR/utility.

| Method group | Adapter | Trigger success rate | Trigger reduction vs original | Clean utility score | Clean delta vs original | Trade-off score | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| original | `original` | 0.6061 | 0.0000 | 0.9667 | 0.0000 | 0.3606 | Reference backdoored adapter; no defence applied. |
| uniform | `uniform_gamma_0.25` | 0.0000 | 0.6061 | 0.9667 | 0.0000 | 0.9667 | Strongest bounded trigger reduction in this table, but it scales the whole adapter and may globally weaken adapter behavior. |
| uniform | `uniform_gamma_0.50` | 0.0202 | 0.5859 | 0.9333 | -0.0334 | 0.9131 | Uniform baseline with less aggressive global attenuation than gamma=0.25. |
| spectral_only | `top3_gamma_0.50` | 0.0606 | 0.5455 | 0.9333 | -0.0334 | 0.8727 | Best spectral-only top-sigma baseline in the bounded run. |
| spectral_only | `top1_gamma_0.50` | 0.0909 | 0.5152 | 0.9333 | -0.0334 | 0.8424 | Simpler spectral-only baseline attenuating the leading component. |
| sensaware_expanded | `sensaware_top224_gamma_0.25` | 0.0101 | 0.5960 | 0.9667 | 0.0000 | 0.9566 | Best expanded SensAware variant; beats spectral-only trigger rates but not the strongest uniform baseline under this heuristic. |
| sensaware_expanded | `sensaware_top128_gamma_0.25` | 0.0707 | 0.5354 | 0.9667 | 0.0000 | 0.8960 | Expanded SensAware variant with fewer selected components than top224. |
| sensaware_expanded | `sensaware_top336_gamma_0.50` | 0.0808 | 0.5253 | 0.9333 | -0.0334 | 0.8525 | Expanded SensAware variant with broader selection but milder attenuation. |
