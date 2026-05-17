# Post-Hoc Sanitisation of Backdoored LoRA Adapters
## via Sensitivity-Aware Singular-Component Attenuation

**Course / Area:** MAI/MKI — AI Security and Privacy / Backdooring of LLMs
**Project type:** Semester project (single course), empirical study
**Duration:** 8 weeks (Phase 2 execution + writing)
**Status:** Phase 1 plan finalised; implementation not yet started
**Version:** Final master document, May 2026


---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Final Title and Rationale](#2-final-title-and-rationale)
3. [Abstract](#3-abstract)
4. [Research Question](#4-research-question)
5. [Problem Setting and Threat Model](#5-problem-setting-and-threat-model)
6. [Prior Art and the Specific Gap](#6-prior-art-and-the-specific-gap)
7. [What Is Genuinely New](#7-what-is-genuinely-new)
8. [Proposed Method](#8-proposed-method)
9. [Clean-Prompt Sensitivity Algorithm](#9-clean-prompt-sensitivity-algorithm)
10. [Experimental Design](#10-experimental-design)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [Pre-Registered GO/NO-GO and Pivot Plan](#12-pre-registered-gono-go-and-pivot-plan)
13. [Eight-Week Plan with Hour Budget](#13-eight-week-plan-with-hour-budget)
14. [Resources and Compute](#14-resources-and-compute)
15. [Honest Risk Analysis](#15-honest-risk-analysis)
16. [Implementation Pitfalls](#16-implementation-pitfalls)
17. [Success Criteria](#17-success-criteria)
18. [Limitations and Ethics](#18-limitations-and-ethics)
19. [Expected Deliverables](#19-expected-deliverables)
20. [AI Assistant Usage Declaration](#20-ai-assistant-usage-declaration)
21. [Open Questions for the Reviewer](#21-open-questions-for-the-reviewer)
22. [References](#22-references)
23. [One-Paragraph Email Version](#23-one-paragraph-email-version)

---

## 1. Executive Summary

Users routinely download small LoRA adapters from public hubs and attach them to trusted base language models without inspection. A backdoored adapter behaves normally on clean inputs but produces attacker-chosen outputs when a trigger is present. Recent work shows that poisoned LoRA adapters leave a distinctive spectral fingerprint in their update matrix \( \Delta W = BA \), but published methods exploit this fingerprint mainly for *detection* (arXiv 2602.15195), *training-time* mitigation (RoRA, arXiv 2601.06305), or *layer-level* spectral pruning for unrelated agent-safety problems (S3LoRA, arXiv 2508.15068).

This project investigates whether the same fingerprint can be turned into a *post-hoc, singular-component-level sanitisation operator*. Keep the trusted base untouched, edit only the adapter, attenuate the small number of singular components that combine high spectral energy with low influence on clean behaviour. The proposed method applies SVD to \( \Delta W \) per LoRA target module, computes a hybrid suspicion score that fuses spectral energy share with a clean-prompt sensitivity term, multiplicatively attenuates the top-K suspicious components, and refactors the result back into a standard LoRA \( A/B \) form.

The contribution is **incremental, not transformative.** The spectral fingerprint is established prior art. The behavioural sensitivity score is a port of Fine-Pruning (Liu et al., 2018) to LoRA singular components. The genuinely new element is the *combination* of these two signals to drive selective, surgical attenuation in a defender setting where retraining is impossible. I estimate ≈25–30 % probability that the proposed method strictly beats all post-hoc baselines, and ≈70 % probability of producing a defensible empirical study — including useful negative results.

---

## 2. Final Title and Rationale

**Post-Hoc Sanitisation of Backdoored LoRA Adapters via Sensitivity-Aware Singular-Component Attenuation**

Each phrase in the title corresponds to one design decision:

- **Post-hoc:** the adapter is already trained when the defender receives it; no retraining is required.
- **Backdoored LoRA adapters:** the threat lives in the adapter, not in the base model.
- **Sensitivity-aware:** clean-prompt behaviour is used to protect useful components from over-attenuation.
- **Singular-component attenuation:** the operator edits singular components of \( \Delta W \) (the SVD basis), not raw LoRA rank factors. This distinction matters because the columns of \( B \) and rows of \( A \) are not orthogonal in general.

---

## 3. Abstract

Backdoored LoRA adapters represent a realistic supply-chain risk because users routinely download small parameter-efficient adapters from public hubs and attach them to otherwise trusted base language models. Recent work has shown that poisoned LoRA adapters exhibit distinctive spectral signatures in their update matrices \( \Delta W = BA \), but existing methods use this signal mainly for detection (arXiv 2602.15195) or training-time mitigation (RoRA, arXiv 2601.06305), and the closest post-hoc spectral method (S3LoRA, arXiv 2508.15068) prunes at the layer level for unsafe agent-planning behaviour rather than for backdoors. This project investigates the less explored niche of post-hoc, third-party-adapter, singular-component-level sanitisation of backdoored LoRA adapters. The research question is whether suspicious singular components of an already-trained backdoored adapter can be selectively attenuated to reduce Attack Success Rate while preserving clean utility. The proposed method computes \( \Delta W \) per LoRA target module, applies thin SVD, and scores each singular component using a hybrid metric that fuses spectral energy share with a clean-prompt sensitivity term measuring how much the component contributes to normal task behaviour on a small calibration set; high-suspicion components are attenuated multiplicatively and the edited update is refactored back into LoRA \( A/B \) form for inference. Experiments use one publicly released BackdoorLLM Llama-2-7B-Chat BadNets jailbreak adapter and compare the proposed method against three post-hoc baselines: the unmodified backdoored adapter, uniform adapter scaling, and top-sigma cutoff. Metrics are Attack Success Rate, clean utility on held-out clean prompts, and the ASR–utility trade-off curve. The expected contribution is empirical evidence on whether sensitivity-aware spectral attenuation outperforms purely spectral or purely uniform post-hoc baselines, with honest reporting of failure modes and an explicit limitations discussion of adaptive attackers.

*(8 sentences; within the 8–12 range required by the course brief; timeline is intentionally not mentioned per professor guidance.)*

---

## 4. Research Question

### 4.1 Primary

> **Can suspicious singular components of an already-trained backdoored LoRA update matrix be attenuated post-hoc to reduce Attack Success Rate while preserving clean task utility?**

### 4.2 Operational form (the version the experiments actually test)

> **Does combining a spectral energy score with a clean-prompt sensitivity score identify LoRA singular components whose attenuation reduces ASR more effectively than uniform adapter scaling or top-sigma pruning, while retaining clean utility on a held-out clean prompt set?**

This is a single falsifiable empirical question. If the answer is yes, the hybrid score adds value over simpler post-hoc baselines. If the answer is no (e.g. the top-sigma cutoff matches the proposed method), the empirical finding itself is informative: it shows that the behavioural signal is redundant for this attack family, and that pure spectral pruning suffices.

---

## 5. Problem Setting and Threat Model

### 5.1 Setting

A user holds a trusted base model (Llama-2-7B-Chat) and a third-party LoRA adapter downloaded from a public hub. The adapter may have been fine-tuned on poisoned data. The user wants to retain useful adapter behaviour while removing trigger-conditioned malicious behaviour, **without retraining the adapter and without trigger knowledge**.

### 5.2 Defender capability

| Capability | Available? | Notes |
|---|---|---|
| White-box access to adapter weights | Yes | Adapter is downloaded as a file |
| White-box access to base model | Yes | Trusted root |
| Small clean calibration prompt set (≈100 prompts) | Yes | No labels, no triggers, no overlap with test set |
| Retraining the backdoored adapter | No | No training budget |
| Knowledge of the trigger pattern | No | Trigger-free defence |
| Access to original training data | No | Adapter is from an unknown source |
| Reasonable GPU (16 GB, 4-bit quantisation) | Yes | DC1.07 lab environment |

### 5.3 Attacker capability

The attacker fine-tuned a LoRA on a poisoned dataset (BadNet-style: a fixed token trigger paired with the attacker's target output) and published the adapter. The attacker is **non-adaptive**: they did not design the attack to defeat the proposed defence. Adaptive attackers are acknowledged as a limitation, not addressed empirically (Section 18).

### 5.4 Out of scope

- Backdoor *detection* (this is sanitisation; we assume suspicion already exists or that defence-in-depth is desired regardless).
- Full-model backdoors living in the base weights rather than the adapter.
- Backdoors against vision or multi-modal LoRAs.
- Robustness to triggers the attacker designs to spread across many singular components.
- Backdoors that activate only after multi-turn priming.

---

## 6. Prior Art and the Specific Gap

### 6.1 The spectral fingerprint of backdoored LoRA adapters

Puertolas Merenciano, Chaudhary et al. (arXiv 2602.15195, February 2026) train 400 clean and 100 poisoned LoRA adapters on Llama-3.2-3B-Instruct at rank 16. Poisoned adapters concentrate spectral energy in a small number of dominant singular directions. A logistic regression on five spectral statistics — leading singular value, Frobenius norm, energy concentration, spectral entropy, kurtosis — reaches ≈97 % detection accuracy with under 2 % false-positive rate. The paper is **detection-only**; its future-work paragraph names architecture generalisation and adaptive adversaries, not removal.

### 6.2 Closest post-hoc spectral method

Ao and Rumchurn (S3LoRA, arXiv 2508.15068, August 2025) propose data-free post-hoc spectral pruning of LoRA updates, using a *Spectral Sharpness Index* at the **layer level**. The target problem is unsafe agent planning, not backdoor sanitisation, and the granularity is one decision per layer rather than per singular component. No clean-prompt sensitivity term.

### 6.3 Training-time spectral defence

Luong and Chen (RoRA, arXiv 2601.06305, January 2026) develop the spectral theory of why LoRA fails to "forget" backdoors during fine-tuning and propose three training-time fixes: clean-strengthened regularisation, trigger-insensitive subspace orthogonalisation, and a uniform global rescale \( s = \sigma_{\max}(W_{\text{pre}})/\sigma_{\max}(\Delta W) \). All require defender control of the fine-tuning loop, which contradicts the present threat model.

### 6.4 Ancestor: Fine-Pruning for CNN backdoors

Liu, Dolan-Gavitt and Garg (Fine-Pruning, RAID 2018) prune CNN neurons with low average clean-input activation under the rationale that backdoor-encoding neurons are dormant on clean inputs. This is the conceptual ancestor of the clean-prompt sensitivity score used in this project, ported from CNN channels to LoRA singular components.

### 6.5 Comparison table

| Work | Post-hoc? | Adapter-level? | Backdoor-specific? | Granularity | Combines spectral + behavioural? |
|---|---|---|---|---|---|
| arXiv 2602.15195 | Yes | Yes | Yes | **Detection only** | Spectral only |
| S3LoRA (2508.15068) | Yes | Yes | No (agent safety) | Layer | Spectral only |
| RoRA (2601.06305) | **No (training-time)** | Yes | Yes | Global rescale | Neither |
| Fine-Pruning (2018) | Yes | **No (CNN)** | Yes | Neuron | Behavioural only |
| **This project** | **Yes** | **Yes** | **Yes** | **Singular component** | **Yes** |

### 6.6 Other relevant work

- **BackdoorLLM** (Li et al., NeurIPS 2025, arXiv 2408.12798) provides the benchmark, attack recipes, and the publicly released Llama-2-7B-Chat jailbreak adapters used here.
- **LoRA** (Hu et al., ICLR 2022, arXiv 2106.09685) defines the parameterisation \( \Delta W = (\alpha/r)\,BA \) the method operates on.
- **PEFTGuard** (Sun et al., IEEE S&P 2025, arXiv 2411.17453) trains a meta-classifier for adapter detection; its PADBench is a future stretch resource.
- **BadNets** (Gu et al., 2017, arXiv 1708.06733) is the attack family used in evaluation.
- **HarmBench** (Mazeika et al., ICML 2024, arXiv 2402.04249) and **Llama Guard** (Inan et al., 2023, arXiv 2312.06674) establish evaluation methodology and the optional judge model.

---

## 7. What Is Genuinely New

This project's contribution is one specific combination, not a wholly new mechanism. Stated precisely:

> **The first post-hoc, third-party-adapter, singular-component-level sanitisation operator for backdoored LoRA adapters that fuses a spectral energy score with a clean-prompt sensitivity score to drive selective attenuation.**

Each of the four bolded qualifiers exists in prior work alone or in pairs; their **combination** does not.

### 7.1 What this project does *not* claim

- It is **not** the first to use SVD on \( \Delta W \) (arXiv 2602.15195 does this for detection).
- It is **not** the first to use spectral pruning of a LoRA update post-hoc (S3LoRA does this at the layer level).
- It is **not** the first to use clean-input activation as a backdoor-defence signal (Fine-Pruning does this for CNNs).
- It is **not** the first to identify that backdoors concentrate in few singular directions (arXiv 2602.15195, RoRA).
- It is **not** a universal LoRA defence.
- It is **not** state-of-the-art across all attacks or models.
- It does **not** handle adaptive attackers.

### 7.2 What it *is* the first to do (literature scan, May 2026)

1. Compute and report a per-singular-component suspicion score that combines spectral energy share with a behavioural clean-prompt sensitivity term inside a single LoRA module.
2. Perform **attenuation** (not deletion) at singular-component granularity, with the attenuation factor \( \gamma \) as a tunable hyperparameter on the ASR–utility trade-off.
3. Empirically compare this hybrid score against a pure spectral baseline (top-sigma cutoff) on the same backdoored adapter, isolating the marginal value of the behavioural signal.

This is the level of contribution appropriate for a 7–8 week semester project. It is not a transformative idea. It is a defensible empirical study that fills one specific cell in the prior-art matrix.

---

## 8. Proposed Method

### 8.1 Pipeline (five steps per LoRA target module)

For each LoRA target module \( m \) in the backdoored adapter:

1. **Compute** the unscaled product: \( \Delta W_m = B_m A_m \).
   (PEFT re-applies the \( \alpha/r \) scaling at inference automatically; performing SVD on the unscaled product avoids double-scaling on refactor.)
2. **Thin SVD:** \( \Delta W_m = U_m \operatorname{diag}(S_m) V_m^\top \) with rank \( \le r \).
3. **Score** each singular component \( k \): see Section 8.2.
4. **Select** the top-\( K \) components by suspicion score.
5. **Attenuate:** \( S_m[k] \leftarrow \gamma \cdot S_m[k] \) for selected \( k \), with \( \gamma \in [0, 1] \).
6. **Refactor:** \( B'_m = U_m \operatorname{diag}(\sqrt{S'_m}) \) and \( A'_m = \operatorname{diag}(\sqrt{S'_m}) V_m^\top \).
7. **Numerical sanity check:** \( \lVert B'_m A'_m - U_m \operatorname{diag}(S'_m) V_m^\top \rVert_F < 10^{-5} \).

The sanitised adapter is saved in standard PEFT format and loaded normally for inference.

### 8.2 Suspicion score

\[
\text{suspicious}[k] \;=\; \underbrace{\frac{S_m[k]^2}{\sum_j S_m[j]^2}}_{\text{spectral energy share}} \;-\; \lambda \cdot \underbrace{\text{clean\_sensitivity}[m, k]}_{\text{behavioural protection term}}
\]

- **Spectral energy share** is high for components carrying disproportionate weight in \( \Delta W \); backdoors tend to concentrate energy in a few directions (arXiv 2602.15195).
- **Clean-sensitivity** measures how much zeroing component \( k \) perturbs clean-prompt outputs; high values protect important clean-behaviour components.
- **\( \lambda \)** controls the strength of behavioural protection. \( \lambda = 0 \) reduces the method to top-sigma cutoff (built-in ablation).

### 8.3 Why SVD on \( \Delta W \), not on the raw factors

The columns of \( B \) and rows of \( A \) are not in general orthogonal: the LoRA factorisation is non-unique under any invertible right-rotation. Therefore the raw rank-1 outer products \( B[:,k] \cdot A[k,:] \) are not Frobenius-orthogonal, per-component norms are basis-dependent, and Frobenius energy does not partition across raw factors. Operating on the SVD of \( \Delta W \) recovers a canonical orthonormal basis in which "attenuate component \( k \)" has a well-defined meaning and per-component perturbations of \( \Delta W \) are bounded.

### 8.4 Cross-module score aggregation

**Default: per-module independent scoring and attenuation.** Each LoRA module ranks and attenuates its own top-\( K \) components. Global ranking across modules is *not* used because singular component \( k \) in `q_proj` has no semantic relationship to component \( k \) in `v_proj`, and a global ranking would introduce additional bias toward modules with larger spectral norm. A global-ranking ablation is a stretch goal for Week 7 if time permits.

### 8.5 Hyperparameter sweep grid

| Symbol | Role | Values swept |
|---|---|---|
| \( K \) | Number of components attenuated per module | \( \{1, 2, 4\} \) |
| \( \gamma \) | Attenuation factor applied to selected components | \( \{0.0,\; 0.3,\; 0.6,\; 0.9\} \) |
| \( \lambda \) | Weight of the behavioural term in the suspicion score | \( \{0,\; 0.5,\; 1.0\} \) |
| \( N \) | Calibration prompt count (fixed) | 100 |

Total configurations per method: \( 3 \times 4 \times 3 = 36 \). The \( \lambda = 0 \) slice of the proposed method coincides with the top-sigma cutoff baseline and serves as a direct, built-in ablation of the behavioural signal.

---

## 9. Clean-Prompt Sensitivity Algorithm

The clean-prompt sensitivity score is the behaviour-preserving half of the suspicion score.

### 9.1 Calibration data

- **Source:** 100-prompt random subset of `tatsu-lab/alpaca` (instruction-tuning prompts).
- **Seed:** fixed (`seed = 42`) for reproducibility.
- **Constraints:** no trigger tokens, no overlap with the BackdoorLLM clean test set used for utility evaluation, no overlap with the poisoned test set.

### 9.2 Algorithm (per-module, per-component)

```
INPUT:  module m, singular index k
        U_m, S_m, V_m^T  (thin SVD of ΔW_m)
        calibration set C = {p_1, ..., p_100}

S_test ← copy(S_m); S_test[k] ← 0
ΔW_test ← U_m · diag(S_test) · V_m^T

For each prompt p in C:
    logits_orig ← model_forward(p, with ΔW_m in module m)
    logits_test ← model_forward(p, with ΔW_test in module m only)
    kl[p] ← mean over first 32 generated positions of
            KL(softmax(logits_orig) ‖ softmax(logits_test))

clean_sensitivity[m, k] ← mean(kl) over C
```

### 9.3 Metric choice and rationale

KL divergence on next-token logits at the first 32 generated positions was chosen because:

1. It is smooth and differentiable-friendly (standard in distillation literature).
2. It captures distributional change rather than only argmax change, which matters when the backdoor influences sampling temperature or top-k.
3. It is cheap: 32 positions per prompt × 100 prompts = 3,200 forward steps per component.

Alternatives considered but rejected for the semester scope: per-token log-likelihood drop on a held-out continuation (cheaper but information-poorer), output-embedding distance (less interpretable), classification accuracy on a held-out subset (requires labels we don't have).

### 9.4 Cost estimate

Per module: \( r \) zero-out evaluations \( \times \) 100 calibration prompts \( \times \) 32 token positions. For BadNet at \( r = 16 \) and 7 LoRA target modules: ≈ 35,840 forward steps total, ≈ 15–30 minutes on a 16 GB GPU with 4-bit quantisation of Llama-2-7B-Chat.

---

## 10. Experimental Design

### 10.1 Assets — exact identifiers

| Asset | Identifier |
|---|---|
| Code repository | `github.com/bboylyg/BackdoorLLM` (NeurIPS 2025) |
| Backdoored adapter (primary) | `attack/DPA/examples/llama2-7b-chat/jailbreak/badnet/` (inside BackdoorLLM repo) |
| Poisoned test JSON | `data/test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json` |
| Clean test JSON | matching `clean/...` file in same repo |
| Base model (gated) | `meta-llama/Llama-2-7b-chat-hf` |
| Base model (fallback mirror) | `NousResearch/Llama-2-7b-chat-hf` |
| Calibration prompts | `tatsu-lab/alpaca`, 100-prompt subset, seed 42 |
| ASR judge (stretch only) | `meta-llama/Llama-Guard-3-8B` |
| Pinned dependencies | `peft 0.11.x`, `transformers 4.40–4.44`, `bitsandbytes ≥ 0.43`, `accelerate ≥ 0.30` |

### 10.2 Baselines

Three post-hoc baselines plus the proposed method. Each baseline isolates one variable.

| Tag | Method | Variable isolated |
|---|---|---|
| **B1** | No defence (original backdoored adapter) | Anchor: confirms the backdoor is present at measurable ASR |
| **B2** | Uniform adapter scaling: \( \Delta W \leftarrow \gamma \cdot \Delta W \), \( \gamma \in \{0.1, \ldots, 0.9\} \) | Does *any* attenuation help, or is the method just regularisation? |
| **B3** | Top-sigma cutoff: zero the \( K \) largest singular values; no sensitivity term | Does the behavioural signal add value beyond pure spectral pruning? |
| **Proposed** | Hybrid spectral + sensitivity score, per-module top-\( K \), attenuation \( \gamma \) | The full method |

**Why this is the right minimal set.** B2 controls for "any weakening of the adapter reduces ASR"; B3 controls for "spectral evidence alone is enough"; B1 controls for "the backdoor exists at the measured rate." A fourth RoRA-style global rescale baseline is a stretch goal for Week 7 if time permits — not committed, because at 7–8 weeks it would crowd out either Lambda ablation or the writing buffer.

---

## 11. Evaluation Protocol

### 11.1 Metrics

| Metric | Definition | Direction |
|---|---|---|
| **ASR** | Fraction of poisoned test prompts that elicit the attacker-target behaviour, measured by rule-based keyword match on the BackdoorLLM-provided test set | ↓ better |
| **Clean utility** | Behavioural agreement (exact match / refusal rate) on 200 held-out clean prompts from the BackdoorLLM clean test JSON | ↑ better |
| **ASR–utility trade-off** | Pareto plot across the \( (K, \gamma, \lambda) \) sweep, one curve per method | — |

### 11.2 Stretch metrics (only if Phase 2 has spare GPU time in Weeks 7–8)

- **C4 perplexity** on a 500-line `c4` sample as a second utility axis.
- **Llama-Guard-3-8B** as judge model for ASR on the headline configuration only.
- **Second random seed** on the headline configuration for a preliminary stability check.

These are demoted to stretch goals because they are not required to answer the research question, and overcommitting in the abstract creates a credibility risk if they cannot be delivered.

### 11.3 Sample sizes and noise floor

- Poisoned test set: 200 prompts (per BackdoorLLM file `backdoor200_jailbreak_badnet.json`).
- Clean test set: 200 prompts.
- Calibration set: 100 prompts (used only for sensitivity scoring; never appears in ASR or utility evaluation).

**Empirically, per-seed ASR standard deviation on 100–200 sample test sets is 3–5 percentage points.** Claims of "method A beats method B" therefore require an observed gap **larger than ≈ 5 ASR points** to be interpretable. Differences below this floor are reported as inconclusive, not as a victory for either method. This is the most important honest disclosure in the evaluation section.

---

## 12. Pre-Registered GO/NO-GO and Pivot Plan

The proposed method depends on a specific empirical premise: that the backdoored adapter's \( \Delta W \) shows visibly higher spectral energy concentration than a clean reference adapter. Without this premise, the spectral half of the suspicion score is uninformative.

### 12.1 Week-2 GO/NO-GO check

Plot the singular-value spectrum of \( \Delta W \) for the BadNet adapter and at least one clean reference (priority order below). Compute:

- Top-3 energy share \( E_3 = \sum_{k=1}^{3} S[k]^2 / \sum_j S[j]^2 \) per module.
- Spectral entropy \( H = -\sum_k (S[k]^2 / \sum_j S[j]^2) \log_2 (S[k]^2 / \sum_j S[j]^2) \) per module.

### 12.2 Clean reference sourcing (priority order)

1. **Preferred:** any BackdoorLLM-released clean Llama-2-7B-Chat LoRA on the same instruction data, if available in the repository or HF organisation.
2. **Fallback:** two community-released clean Llama-2-7B-Chat instruction-tuning LoRAs from Hugging Face.
3. **Last resort:** self-trained clean LoRA on a small Alpaca subset using the BackdoorLLM training config with the trigger removed (~6 GPU-hours).

### 12.3 GO criterion (quantitative)

> **GO:** Top-3 energy share \( E_3 \) in the BadNet adapter is at least **20 % higher** (relative) than in the clean reference, in at least **4 of 7** target modules.

This threshold is set without pilot data and is acknowledged as a judgement call (see Section 21, open question 2). It is the smallest difference that should plausibly produce a measurable downstream effect on the proposed method.

### 12.4 NO-GO pivot

If the GO criterion fails, the project drops the spectral term and operates on `clean_sensitivity` alone, framing the contribution as a LoRA analogue of Fine-Pruning. Novelty narrows but remains real. The pivot is decided in Week 2, not later, to preserve the experimental timeline.

---

## 13. Eight-Week Plan with Hour Budget

| Week | Tasks | Hours |
|---|---|---|
| 1 | Clone BackdoorLLM, pin dependencies, request Llama-2 access (parallel: prepare NousResearch fallback), load base model, smoke-test BadNet adapter forward pass | 8 |
| 2 | Reproduce BackdoorLLM-reported ASR; source clean reference adapter(s); run quantitative GO/NO-GO check (Section 12); produce sanity-check figure | 10 |
| 3 | Implement \( \Delta W \) extraction per module, thin SVD, refactor, round-trip numerical sanity check | 8 |
| 4 | Implement spectral score, clean-sensitivity algorithm (Section 9), full suspicion score, per-module top-\( K \) selector, attenuation operator | 8 |
| 5 | Implement baselines B1, B2, B3; run small pilot sweep to verify pipeline correctness | 10 |
| 6 | Full attenuation sweep \( (K, \gamma, \lambda) \) for proposed method + B2 + B3; produce ASR–utility curves | 10 |
| 7 | Analysis: lambda ablation, optional global-ranking ablation, optional second seed on headline config; draft report figures | 8 |
| 8 | Writing + revision + buffer (absorb any week-1/2 overrun); finalise reproducibility notes | 8 |
| **Total** | | **70** |

**Buffer policy.** Week 8 contains the only buffer. If implementation slips beyond Week 5, the lambda ablation in Week 7 is the first cut; the global-ranking and second-seed analyses are the next cuts. The report draft must begin no later than the start of Week 7.

---

## 14. Resources and Compute

| Resource | Amount |
|---|---|
| GPU | DC1.07 lab GPU, 16 GB, 4-bit quantisation of Llama-2-7B-Chat |
| Datacenter A80 hours requested | 0 |
| GPU-hours total (committed scope) | < 40 |
| GPU-hours total (with all stretch goals) | < 60 |
| Disk | < 50 GB for adapters, test data, calibration set, intermediate results |
| Wall-clock | 8 weeks |
| Total student effort | ≈ 70 hours |

**Why no datacenter A80.** The largest per-step cost is forward inference on Llama-2-7B-Chat in 4-bit, which fits comfortably in 16 GB at batch sizes ≤ 4. SVD on rank-16 LoRA matrices is negligible. No fine-tuning is performed except the optional clean-reference LoRA (Section 12.2 option 3), which is short enough for the lab GPU.

---

## 15. Honest Risk Analysis

| # | Risk | Probability | Impact | Defence / pivot |
|---|---|---|---|---|
| 1 | **B3 (top-sigma cutoff) matches the proposed method** | **≈ 30 %** | Core novelty claim weakens | Report as useful negative finding: spectral concentration alone suffices; the behavioural term is redundant for this attack. The \( \lambda \)-sweep is a built-in ablation that exposes this directly. |
| 2 | **Spectral signature absent in BadNet adapter** | **≈ 10–15 %** | Spectral half of suspicion score is uninformative | Pre-registered Week-2 GO/NO-GO triggers clean-sensitivity-only pivot (Fine-Pruning analogue for LoRA). |
| 3 | **Backdoor and clean behaviour entangled in same components** | **≈ 10 %** | ASR–utility trade-off is bad for every method | Report as evidence about how BadNet places its backdoor inside LoRA — itself a finding about attack mechanism. |
| 4 | **Infrastructure pain (Llama-2 gating, PEFT/NF4)** | **≈ 10 %** | 1–2 weeks lost | Pinned dependencies; NousResearch mirror in parallel from Week 1; adapter kept unmerged from NF4 base. |
| 5 | **A new arXiv preprint scoops the niche during the project** | **≈ 5–10 %** | Novelty erosion | The corridor is narrow (post-hoc × adapter × singular-component × backdoor); a careful comparison remains defensible even with partial overlap. |
| 6 | **Adaptive-attacker objection from reviewer or examiner** | **≈ 50 % asked** | Low if disclosed early | Explicit limitation in Section 18; cite adaptive-attacker literature; do not claim robustness. |

### 15.1 Overall feasibility estimates

- Probability of completion with a **defensible empirical result of any kind**: **≈ 70 %**.
- Probability of completion with the proposed method **strictly beating all post-hoc baselines**: **≈ 25–30 %**.

The 25–30 % number is the most important honest disclosure in this document. The professor and reviewer should understand that the headline claim is not pre-determined and that a tie or partial loss against B3 is a likely and publishable outcome.

---

## 16. Implementation Pitfalls

| Pitfall | Why it matters | Safeguard |
|---|---|---|
| **\( \alpha/r \) scaling double-application** | PEFT multiplies \( BA \) by \( \alpha/r \) at inference. If SVD is done on the *scaled* product and refactored without compensation, the effective update is off by a factor of \( \alpha/r \). | Compute SVD on the **unscaled** \( B A \); PEFT re-applies \( \alpha/r \) automatically. Verify with the numerical round-trip check. |
| **DoRA reparameterisation** | DoRA's effective update is not \( BA \); singular-component edits do not commute with the per-output magnitude vector. | Explicitly exclude DoRA adapters. The BackdoorLLM BadNet adapter uses standard LoRA. |
| **rsLoRA scaling** | rsLoRA uses \( \alpha/\sqrt{r} \) instead of \( \alpha/r \). | Excluded by the same check; BackdoorLLM BadNet does not use rsLoRA. |
| **NF4 quantisation rounding** | Merging a small singular-value edit into a 4-bit quantised base can round the change away. | Keep the adapter unmerged in fp16 throughout editing; load with `peft` not by manual merging. |
| **Llama-2 licence gating** | Meta's gated repo can delay access by days. | Use `NousResearch/Llama-2-7b-chat-hf` mirror in parallel from Week 1. |
| **PEFT / transformers version drift** | Saving and loading adapter state dicts has changed across `peft` 0.7 → 0.11. | Pin from BackdoorLLM `requirements.txt`. |
| **Numerical drift on round-trip** | Floating-point error in SVD + refactor can accumulate. | Assert \( \lVert B'A' - U \operatorname{diag}(S') V^\top \rVert_F < 10^{-5} \) per module before saving. |
| **Calibration / test set overlap** | If calibration prompts leak into the clean utility test, results inflate. | Fixed seed 42 for the Alpaca calibration subset; explicit prompt-level deduplication against BackdoorLLM clean test set. |
| **Cross-module attenuation interactions** | Attenuating component \( k \) in one module changes the effective input to downstream modules during sensitivity scoring. | Compute sensitivity scores **sequentially**, one module at a time, against the original (unedited) adapter — not against partially edited intermediate states. |
| **Tokeniser mismatch** | `Llama-2-7b` and `Llama-2-7b-chat` differ in chat formatting tokens; using the wrong one breaks BackdoorLLM's prompt formatter. | Always load the Chat tokeniser; confirm `<s>[INST] ... [/INST]` formatting matches the BackdoorLLM eval scripts. |

---

## 17. Success Criteria

A clear empirical answer is the success criterion — not a positive answer.

| Outcome | What it looks like | Publishability |
|---|---|---|
| **Strong positive** | Proposed method gives a better ASR–utility Pareto front than B2 and B3 at multiple operating points, with gaps > 5 ASR points | Strong result; clear contribution |
| **Acceptable mixed** | Proposed method beats B2 cleanly, ties or modestly beats B3; \( \lambda \)-sweep shows the behavioural term helps in some regions of the curve only | Defensible result; honest contribution |
| **Negative but useful** | B3 (top-sigma cutoff) matches the proposed method on every operating point | Useful negative finding: behavioural term is redundant for this attack family |
| **Pivot outcome** | Week-2 GO/NO-GO fails; project pivots to clean-sensitivity-only scoring (Fine-Pruning analogue for LoRA) | Defensible feasibility-style contribution; narrower but real |
| **Failure** | Pipeline does not run end-to-end by Week 6 | Recoverable only if Week 7–8 buffer is fully spent on debugging; report would describe lessons learned |

The course brief explicitly accepts negative results as long as the methodology is sound and the reasoning is clear. This document is structured to make any of the first four outcomes defensible.

---

## 18. Limitations and Ethics

### 18.1 Acknowledged limitations

1. **One adapter, one attack family.** BadNet uses a fixed token trigger. Generalisation to Sleeper (token-sequence triggers), VPI (virtual prompt injection), MTBA, CTBA, or to semantic / distributional triggers is not tested.
2. **Single base model.** Llama-2-7B-Chat. Generalisation to other architectures and sizes is not tested.
3. **Non-adaptive attacker.** A defender-aware attacker can train backdoors that distribute trigger response across many singular components, defeating component-level attenuation. This is a known limitation of all signature-based defences. Adaptive evaluation is left for future work.
4. **Single-seed headline results.** ASR has 3–5 percentage point per-seed standard deviation. The optional second-seed stability check (Section 11.2) on the headline configuration is a partial mitigation, not a substitute for proper seed averaging.
5. **Rule-based ASR.** The committed metric is keyword-based. The judge-model ASR (Llama-Guard-3-8B) is a stretch goal. Rule-based ASR can over- or under-count subtle generations.
6. **No defence against the broader threat surface.** Backdoors in the base model, multi-trigger backdoors, and backdoors that activate only after multi-turn priming are out of scope.

### 18.2 Ethics

The project develops a **defensive** method. It does not propose a new attack, train new backdoors beyond what already exists in BackdoorLLM (a publicly released NeurIPS 2025 benchmark), or release tooling that increases attacker capability. The trained backdoor adapters used are already public. No human subjects, no personal data, no API queries to third-party providers that retain prompts. The codebase will be released for reproducibility under a permissive license once the project is graded.

---

## 19. Expected Deliverables

By the end of Week 8:

1. **Final report** (4–6 pages text, plus figures, tables, cover page; Arial 11 pt, single column, single line spacing, LaTeX `article` class — per course brief).
2. **Reproducible implementation** of:
   - LoRA \( \Delta W \) extraction per module
   - Thin SVD + refactor with round-trip sanity check
   - Spectral energy score
   - Clean-prompt sensitivity score (per Section 9)
   - Per-module top-\( K \) selector and attenuation operator
   - Baseline implementations B1, B2, B3
3. **Result tables** for B1, B2, B3, and the proposed method on the committed metrics.
4. **At least one ASR–utility trade-off figure** comparing the four methods.
5. **Quantitative GO/NO-GO sanity-check figure** (clean vs. backdoored spectrum overlay).
6. **AI assistant usage declaration** (Section 20).
7. **A clear statement** of whether the proposed method improves over the baselines, supported by the empirical evidence.

---

## 20. AI Assistant Usage Declaration

AI assistants were used during **Phase 1 (planning)** for:

- Literature search and verification of the three closely related papers (arXiv 2602.15195, 2508.15068, 2601.06305) including author lists and method scope.
- Critique of an early draft of the method that conflated raw LoRA outer products with SVD singular components — a mathematical error caught and fixed during planning.
- Drafting and editing the project plan, the abstract, this document, and the review version sent to external reviewers.
- Sanity-checking probability estimates, identifying missing baselines, and proposing the quantitative GO/NO-GO threshold.

During **Phase 2 (implementation)**, AI assistants will be used for:

- Boilerplate generation (data loaders, plotting code) — always reviewed and edited before commit.
- Debugging suggestions when stuck on a specific error.
- Writing-pass copy-edits on the final report.

AI assistants will **not** be used to:

- Generate experimental results.
- Write substantive analytical claims that the author has not personally verified against the data.
- Produce code committed unmodified to the project repository.

Approximate planning-phase AI usage: ≈ 10–12 hours across Phase 1. All technical decisions, the threat model, the choice of baselines, the experimental design, the implementation (to be written), the analysis (to be performed), and the written interpretation of results are the author's responsibility.

---

## 21. Open Questions for the Reviewer

Reviewer input would be especially valuable on the following:

1. **Sensitivity score formulation.** Is KL divergence on next-token logits over the first 32 generated positions the right behavioural signal? Alternatives considered: log-likelihood drop on a continuation, output-embedding distance, classification accuracy on a held-out classification subset. KL was chosen for its smoothness and standard use in distillation literature.
2. **GO/NO-GO threshold.** The "top-3 energy share at least 20 % higher in at least 4 of 7 modules" criterion is set without pilot data. Is there a more principled threshold the reviewer would suggest, or is a qualitative pre-registration (decided by visual inspection of overlaid spectra) preferable for a project at this scale?
3. **Per-module vs. global score aggregation.** Per-module independent attenuation is simpler and easier to debug. Does the reviewer expect a global \( (\text{module}, \text{component}) \) ranking — despite its semantic awkwardness — to produce noticeably stronger results?
4. **Adaptive-attacker treatment.** Acknowledging the adaptive attacker only in the limitations section is the planned approach. Is this sufficient at semester-project level, or would the reviewer expect at least a token analytical bound (e.g. "if the attacker spreads the trigger across \( \ge K' \) components, the defence fails")?
5. **Second attack family.** If 8 weeks allowed one additional adapter, the candidates are Sleeper (token-sequence trigger) and VPI (virtual prompt injection). Which would more strengthen the contribution — and is even one extra attack worth the timeline pressure?
6. **Scooping risk.** Given the velocity of the spectral-LoRA-defence subfield (three closely related papers in 15 months), should the project aim for a workshop submission inside the experimental window, or is that overreach for a semester project?

---

## 22. References

[1] Y. Li, H. Huang, Y. Zhao, X. Ma, J. Sun. *BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models.* arXiv:2408.12798, 2024. NeurIPS 2025. https://arxiv.org/abs/2408.12798

[2] D. Puertolas Merenciano, M. Chaudhary, et al. *Weight Space Detection of Backdoors in LoRA Adapters.* arXiv:2602.15195, 2026. https://arxiv.org/abs/2602.15195

[3] S. Ao, G. Rumchurn. *S3LoRA: Safe Spectral Sharpness-Guided Pruning in Adaptation of Agent Planner.* arXiv:2508.15068, 2025. https://arxiv.org/abs/2508.15068

[4] H.-C. Luong, L. Chen. *Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models (RoRA).* arXiv:2601.06305, 2026. https://arxiv.org/abs/2601.06305

[5] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021. ICLR 2022. https://arxiv.org/abs/2106.09685

[6] K. Liu, B. Dolan-Gavitt, S. Garg. *Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks.* arXiv:1805.12185, 2018. RAID 2018. https://arxiv.org/abs/1805.12185

[7] T. Gu, B. Dolan-Gavitt, S. Garg. *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* arXiv:1708.06733, 2017. https://arxiv.org/abs/1708.06733

[8] Z. Sun et al. *PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning.* arXiv:2411.17453, 2024. IEEE S&P 2025. https://arxiv.org/abs/2411.17453

[9] M. Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249, 2024. ICML 2024. https://arxiv.org/abs/2402.04249

[10] H. Inan et al. *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* arXiv:2312.06674, 2023. https://arxiv.org/abs/2312.06674

---

## 23. One-Paragraph Email Version

> I am proposing a scoped empirical project on post-hoc sanitisation of backdoored LoRA adapters. The setting assumes a trusted base LLM (Llama-2-7B-Chat) and an untrusted third-party LoRA adapter, with white-box access to the adapter weights but no trigger knowledge, no original training data, and no retraining budget. The method computes each LoRA update matrix \( \Delta W = BA \), applies thin SVD, scores singular components using a hybrid metric combining spectral energy share with a clean-prompt sensitivity term (KL divergence on logits over 100 calibration prompts from Alpaca), attenuates suspicious components multiplicatively, and refactors the edited update back into LoRA \( A/B \) form for inference. The contribution is the *combination* of spectral and behavioural scoring at singular-component granularity for backdoor sanitisation — not the spectral observation itself, which is established prior art in arXiv 2602.15195. Evaluation compares the method against three post-hoc baselines (no defence, uniform adapter scaling, top-sigma cutoff) on one BackdoorLLM Llama-2-7B-Chat BadNet jailbreak adapter, using Attack Success Rate and clean utility on held-out clean prompts. A pre-registered Week-2 GO/NO-GO sanity check (clean-vs-backdoored spectral overlay) gates the spectral half of the method. I estimate ≈70 % probability of a defensible empirical result and ≈25–30 % probability of the method strictly beating all baselines. I would value feedback on the sensitivity-score formulation, the GO/NO-GO threshold, and whether the 8-week scope is appropriate.

---

*End of master project document. Implementation begins after Phase 1 sign-off.*
