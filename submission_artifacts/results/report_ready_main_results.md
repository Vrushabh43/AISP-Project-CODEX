# Main Results

Official rule-based ASR is the verified BackdoorLLM refusal-keyword scorer
applied to project-local deterministic `[INST]` generations. It is not
external-judged ASR. Clean perplexity is a reference-likelihood probe, not final
human utility.

| Method group | Adapter | Official rule-based ASR | Wilson 95% CI | Clean reference perplexity | Interpretation |
|---|---|---:|---:|---:|---|
| original | `original` | 36/99 = 0.363636 | [0.275617, 0.461843] | 3.655260 | Backdoored adapter reference. |
| uniform | `uniform_gamma_0.25` | 13/99 = 0.131313 | [0.078372, 0.211799] | 3.747472 | Low ASR but worse clean-reference perplexity than original. |
| uniform | `uniform_gamma_0.50` | 7/99 = 0.070707 | [0.034670, 0.138816] | 3.607213 | Strong uniform baseline. |
| spectral-only | `top3_gamma_0.50` | 7/99 = 0.070707 | [0.034670, 0.138816] | 3.559641 | Strong spectral-only baseline with clean perplexity close to SensAware. |
| SensAware | `sensaware_top224_gamma_0.25` | 1/99 = 0.010101 | [0.001785, 0.055017] | 3.557692 | Lowest observed ASR and lowest clean-reference perplexity in the focused run. |

The final result is favorable to the best SensAware variant under this verified
rule-based scorer, but the sample size is 99 prompts and Wilson intervals should
be reported with all ASR rates.
