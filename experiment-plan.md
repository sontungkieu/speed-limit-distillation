# Speed Limit of Distillation: Experiment Plan

Nov 13, 2026 · @ihb

## Purpose and hypotheses

The first 3 weeks decide one thing: is slow absorption in on-policy distillation (OPD) a real, conditioning-driven limit that state selection can lift, or just a tuning or noise problem? Every phase ends in a written go/no-go call.

**Question.** Coverage is cheap but absorption is slow. [One Training Example](https://arxiv.org/abs/2609.04172) reports 16 queries reach 98.9% state coverage, yet a fixed set of states still takes hundreds of steps to absorb. We ask what sets that speed, and whether choosing the supervised states can raise it.

**Working model.** On a fixed state set S, linearize the student around its current weights. The teacher–student residual r then evolves as

```latex
r_{t+1} \approx (I - \eta K_S)\, r_t
```

where K\_S is the kernel of per-state gradients on S. Each eigendirection decays at rate (1 − ηλ), stability needs η < 2/λ\_max, so absorption time grows with the condition number on the directions the residual occupies.

**Hypotheses.**

- **H1 (predictability):** the spectrum of K\_S, weighted by where the residual lies, predicts measured absorption speed on S.
- **H2 (cause):** slow absorption comes mainly from conditioning, not from sampled-token gradient noise or a too-small learning rate.
- **H3 (control):** state sets chosen for conditioning absorb faster than random or top-score sets of the same size, and reach a target accuracy sooner in wall-clock time.

**What counts as failure (and is still useful).**

- Raising the learning rate 10× absorbs about 10× faster with no harm: the limit is tuning, not conditioning.
- Exact top-K gradients absorb far faster than sampled-token ones: the limit is noise, so the paper becomes about variance reduction.
- The spectrum does not rank absorption speed across sets: the linear model is wrong here, so drop H1 and H3.

## Setup

Match [Semi-OPD](https://arxiv.org/pdf/2610.11291)'s published recipe exactly, so its numbers serve as our sanity check and its method as our main baseline.

| Item | Value |
| --- | --- |
| Primary pair | Qwen3-1.7B (non-thinking) ← Qwen3-4B-Instruct-2507 (top-64 overlap 62.5%) |
| Second pair (Phase 2 only) | DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-1.5B (thinking, overlap 79.7%) |
| Training prompts | DAPO-Math-17K (14,116 prompts after dedup) |
| Objective | Sampled-token reverse KL; top-K reverse KL as the Phase 1 control |
| Optimizer | AdamW, β = (0.9, 0.999), weight decay 0.01, grad clip 1.0, bf16 |
| Learning rate | 1×10⁻⁶ constant (baseline; swept in Phase 1) |
| Batch | 128 prompts × 1 rollout per step |
| Rollout cap | 4,096 tokens (primary); 16,384 (second pair) |
| Teacher cache | Sampled-token log-prob + top-64 log-probs per position |
| Dev eval | AIME24 + AIME25, avg@8, 16K cap |
| Final eval | AIME24/25/26 + HMMT Feb26, avg@32, 32K cap |
| Hardware | 1× B200 |
| Codebase | An existing OPD trainer (NeMo-RL, as Semi-OPD used, or verl) |

Use exactly the student checkpoints in Semi-OPD's Table 3. Semi-OPD underperformed OPD when started from a base student, so a base checkpoint changes the results.

Dev eval is for decisions only. Never pick hyperparameters on the final benchmarks.

## Definitions and metrics

Every experiment reports the same four numbers, so results compare across phases.

**Pool P.** One rollout per prompt from the initial student, cached with teacher log-probs. Start with 2,048 prompts; this is the source every S is drawn from.

**State set S.** A subset of m cached rollouts; each token position in them is one state. Train on S repeatedly, with no new generation. Default m = 64 rollouts (roughly 64 × average length states). Report m in rollouts and in tokens.

**Absorption curve.** A(t) = mean top-64 reverse KL over all states in S at step t, divided by A(0). Use top-64 KL here even when training on sampled tokens, so curves compare across objectives.

**Absorption time.** τ₅₀ and τ₉₀ = steps until A(t) first falls below 0.5 and 0.1. If a run never reaches 0.1 within the step budget, report the budget and the final A.

**Generalization checks.** Absorbing S is only useful if it transfers. Track:

- **Held-out KL:** top-64 reverse KL on 256 cached rollouts from prompts not in S (fixed set, reused everywhere).
- **Fresh-rollout KL:** every 50 steps, sample 128 new student rollouts on held-out prompts, score with the teacher, and report mean KL. This catches drift that cached states hide.
- **Dev accuracy:** AIME24/25 avg@8 at the end of each run.

**Kernel K\_S.** K\_S = G Gᵀ, with G the per-state gradients of the training loss with respect to the trainable parameters. Never materialize G. A matrix–vector product costs one backward pass plus one JVP:

```latex
K_S u = G\,(G^\top u), \quad G^\top u = \nabla_\theta \textstyle\sum_i u_i \ell_i, \quad G v = \text{JVP of } (\ell_1,\dots,\ell_n) \text{ along } v
```

Run 40–60 Lanczos iterations to get the top eigenvalues and eigenvectors. Also compute an Adam-preconditioned version, G P Gᵀ with P = diag(1/(√v + ε)) from the current optimizer state, since we train with Adam.

**Spectral statistics per S.** λ\_max; effective rank (Σλ)²/Σλ²; and the residual-weighted prediction

```latex
\hat{A}(t) = \frac{\sum_k (u_k^\top r)^2 (1-\eta\lambda_k)^{2t}}{\sum_k (u_k^\top r)^2}
```

which gives a predicted τ̂₅₀. The part of r outside the top eigenvectors is reported as "unresolved mass". A large value means the sketch missed the slow directions.

Scope note: with 1.7B parameters, full gradients are costly. Start with LoRA (rank 16, all linear layers) for the spectral runs, and check on one S that LoRA and full fine-tuning give similar absorption curves before trusting it.

## Phase 0: reproduce and calibrate (week 1)

Goal: a pipeline that reproduces Semi-OPD on the primary pair, plus measured costs on our B200.

- [ ] Generate one rollout per prompt from the initial student for all 14,116 prompts (4K cap); log total generation time.
- [ ] Score them with the teacher; cache sampled-token and top-64 log-probs. Log scoring time and cache size on disk.
- [ ] Run Semi-OPD for 100 steps with the Setup hyperparameters. Evaluate at steps 60, 80 and 100.
- [ ] Run standard OPD for 10 steps only, and time generation, scoring and the update separately.
- [ ] Build the fixed pool P (2,048 prompts) and the held-out sets from the same cache.
- [ ] Implement the Lanczos kernel routine; test it on a tiny model where G can be built explicitly, and check eigenvalues agree to within 1%.

**Pass criteria**

- Semi-OPD final-eval avg@32, averaged over steps 60/80/100, lands near the paper's 39.8 (Table 1). Within about 3 points is acceptable for a first run; investigate a larger gap before going on.
- Measured cost written down per 100 steps, split into generation, scoring and training. The paper reports 5.0 H100 GPU-hours for Semi-OPD and 56.8 for OPD on this pair (Table 4). Our B200 number replaces every earlier estimate.

If full final eval is too slow, run it once here to check the number, then use dev eval until Phase 2 finals.

## Phase 1: kill tests (weeks 2–3)

Three cheap tests, each able to end the conditioning story. All use the same three random state sets S₁, S₂, S₃ (m = 64 rollouts each, fixed seeds), train only on S with no new generation, and run 300 steps or until A(t) < 0.1. Run them in this order.

### K1: is it just the learning rate?

Train on each S with lr ∈ {1×10⁻⁶, 3×10⁻⁶, 1×10⁻⁵, 3×10⁻⁵, 1×10⁻⁴}, sampled-token loss, Adam. Repeat the best two with plain SGD (momentum 0.9) as a clean check of the linear theory.

Record τ₅₀, τ₉₀, held-out KL, fresh-rollout KL and fresh-rollout length. Mark the largest lr at which held-out KL still improves and rollouts don't inflate; call it lr\*.

- **Kill:** τ₅₀ falls roughly in proportion to lr up to 1×10⁻⁴ with no harm to held-out KL or dev accuracy. Then the limit is tuning; report that and switch to the replay plan.
- **Continue:** absorption speeds up but saturates, or lr\* is reached while A(t) is still above 0.5. That saturation is the speed limit we want to explain. Use lr\* for everything after this.

### K2: is it gradient noise?

At lr\*, train on the same S with three objectives: (a) sampled-token reverse KL; (b) top-64 renormalized reverse KL (exact over the candidate set); (c) sampled-token with a control-variate baseline, as in vOPD.

- **Kill:** (b) or (c) reaches τ₅₀ in under half the steps of (a), consistently across all three sets. Then noise dominates, and the paper becomes about variance reduction.
- **Continue:** (b) is faster by less than about 1.5×. From here on, train with (b), so later results aren't confounded by sampling noise.

### K3: does the spectrum predict the curve?

At step 0 for each S, compute the top 50 eigenpairs of K\_S (and the Adam-preconditioned version) on the LoRA parameters. Build Â(t) and τ̂₅₀ using lr\* converted to an effective step size: for SGD η = lr, for Adam use the preconditioned kernel.

Compare Â(t) with the measured A(t) over the first 50 steps, then recompute the spectrum at steps 50 and 100 to see how fast it drifts.

- **Kill:** prediction misses by more than 2× in τ₅₀ under SGD, even early. The linear picture doesn't describe this setting.
- **Continue:** prediction is within 2× under SGD and gets the shape of the curve, even if Adam is worse. Note how much residual mass sits in the slow eigendirections; that is the quantity selection will try to change.

### Gate 1 (end of week 3)

Write a one-page memo with the K1–K3 tables and a call: continue to Phase 2, pivot (tuning or noise story), or switch back to the replay plan. Discuss it with the mentor before spending Phase 2 compute.

## Phase 2: H1 and H3 (weeks 4–7)

Only start if Gate 1 says continue. Use lr\*, the top-64 objective, and the primary pair unless stated.

### E1: does the spectrum rank state sets? (H1, weeks 4–5)

Draw 20 sets S from pool P, all m = 64 rollouts, and make them deliberately varied: 8 random, 4 from one topic, 4 of the longest rollouts, 4 of the highest-KL rollouts. Compute spectral statistics at step 0, then train each for 300 steps and measure τ₅₀.

Report the Spearman correlation between measured τ₅₀ and each predictor:

| Predictor | Kind |
| --- | --- |
| Predicted τ̂₅₀ from the spectrum | Ours |
| λ\_max | Ours |
| Effective rank | Ours |
| Mean initial KL on S | Baseline |
| Mean student entropy on S | Baseline |
| Top-64 teacher–student overlap on S | Baseline (Semi-OPD's indicator) |
| Total tokens in S | Baseline |

Pass: τ̂₅₀ correlates at ρ ≥ 0.6 and beats every baseline predictor. Partial pass: it helps when combined with a baseline in a two-feature regression.

### E2: can we pick faster-absorbing sets? (H3, weeks 5–6)

Compare selectors at the same budget, choosing m = 64 rollouts from P. Repeat at m = 16 and 256 for one seed to see how the effect scales.

1. **Random** (3 draws).
2. **Top score:** highest TIP-style score (student entropy + KL).
3. **Diversity:** k-center on 1,024-dimensional random projections of per-rollout gradients.
4. **Log-det (ours, v1):** greedily maximize log det(K\_S + δI) on the gradient sketches. This spreads the spectrum and penalizes redundant, high-leverage rollouts.
5. **Predicted gain (ours, v2):** greedily pick the rollout that most reduces predicted held-out KL after T steps under the linear model, with η capped at 1/λ\_max(S). Add a coverage constraint so selection can't drop late reasoning stages; Semi-OPD found that training on only the first quarter of rollouts hurt.

Measure absorption on S, held-out KL, fresh-rollout KL and dev accuracy. Then measure total selection cost: sketch time plus Lanczos time per selection.

Pass: v1 or v2 reaches τ₅₀ at least 1.5× faster than random and top-score, and ends with equal or better held-out KL.

### E3: does it accelerate real training? (weeks 6–7)

Turn the best selector into a training loop: every R steps, select S from the cached pool, train on it, refresh a slice of the pool when absorption on S stalls. Compare on wall-clock time to reach Semi-OPD's dev accuracy at step 100.

| Method | Seeds | Purpose |
| --- | --- | --- |
| Semi-OPD, single pass | 3 | Main baseline |
| Standard OPD | 1 | Reference (expensive) |
| Random small sets, same schedule | 3 | Isolates selection |
| Best selector from E2 | 3 | Ours |

Repeat Semi-OPD, random small sets and our method on the second pair with 2 seeds if time allows. That pair is where OPD beats Semi-OPD, so it tests whether refresh closes the gap.

The acceleration claim needs our method to beat Semi-OPD on time-to-target, counting selection cost. If it only beats random small sets, the paper is a scientific result about absorption, not an acceleration method; say so in the framing.

## Timeline, compute and gates

The plan fits about 10 weeks on 1 B200, ending at the projected ICML 2027 deadline of 22 January (abstracts 16 January). These dates are a third-party projection; confirm them once the [official call](https://icml.cc) is posted.

| Week | Dates (2026–27) | Work | Gate |
| --- | --- | --- | --- |
| 1 | 16–22 Nov | Phase 0: reproduce Semi-OPD, cache pool, test Lanczos | Reproduction within about 3 points |
| 2 | 23–29 Nov | K1 learning rate, K2 noise |  |
| 3 | 30 Nov–6 Dec | K3 spectrum, Gate 1 memo | **Gate 1:** continue, pivot or switch |
| 4–5 | 7–20 Dec | E1 predictability | **Gate 2:** ρ ≥ 0.6, beats baselines |
| 5–6 | 14–27 Dec | E2 selectors | **Gate 3:** ≥1.5× faster than random |
| 6–7 | 21 Dec–3 Jan | E3 training loop, second pair |  |
| 8 | 4–10 Jan | Extra seeds, final evals (avg@32) | Freeze results 10 Jan |
| 9 | 11–16 Jan | Write paper; submit abstract | Abstract by 16 Jan |
| 10 | 17–22 Jan | Finish and submit | Paper by 22 Jan |

**Compute budget (rough; replace with Phase 0 measurements).**

| Phase | Runs | Estimated B200-hours |
| --- | --- | --- |
| Phase 0 | 1 Semi-OPD + 10 OPD steps + caching | 10–20 |
| K1 | 3 sets × 5 lrs + 6 SGD runs = 21 | 40–80 |
| K2 | 3 sets × 3 objectives = 9 | 15–30 |
| K3 | Spectra for 3 sets × 3 checkpoints | 5–15 |
| E1 | 20 sets + spectra | 40–80 |
| E2 | 5 selectors, \~13 runs + sketches | 30–60 |
| E3 | \~11 primary + \~6 second-pair runs | 120–250 |
| Final evals | avg@32 on finals | 40–80 |
| **Total** |  | **About 300–600** |

One B200 gives about 940 usable hours over 8 experimental weeks, so this leaves some slack for failures. Fixed-set runs are cheap because they skip generation; E3 and final evals dominate.

**If behind schedule:** drop the second pair first, then m = 16/256 sweeps, then OPD seeds. Never drop the Semi-OPD and random baselines.

## Logging, pitfalls and reading

**Log every step:** loss on S, A(t), held-out KL, gradient norm, update norm, learning rate, and wall-clock time split into generation, scoring, selection and training. Every 50 steps add fresh-rollout KL and fresh-rollout length. Save the IDs of the rollouts in every S, so any set can be rebuilt exactly.

**Weekly report, one row per run:**

| Run | Pair | S (rollouts / tokens) | Objective | lr | τ₅₀ | τ₉₀ | Held-out KL | Dev acc | B200-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

**Pitfalls to avoid**

- **EOS and tokenizer:** check the student and teacher score the same token IDs, including EOS. A mismatch silently breaks every KL number.
- **Precision:** run Lanczos and JVPs in fp32; bf16 gives unreliable small eigenvalues.
- **Same parameters everywhere:** the kernel must use exactly the parameters being trained, LoRA or full.
- **Consistent KL:** always measure A(t) with the same top-64 renormalization, whatever the training objective.
- **Overfitting S:** falling loss on S with rising held-out KL is overfitting, not absorption. Always report both.
- **Fair baselines:** tune lr separately for each method, and compare at matched tokens or matched wall-clock time, never matched steps alone.
- **Seeds:** fix seeds for set sampling and rollouts; three seeds for any headline number.

**Read before week 2**

- [Rethinking OPD II: One Training Example](https://arxiv.org/abs/2609.04172): the coverage-vs-absorption finding this plan starts from. Read in full, as only the abstract has been checked.
- [Semi-OPD](https://arxiv.org/pdf/2610.11291): recipe, costs (Table 4) and the main baseline.
- [MetaOPD](https://arxiv.org/pdf/2610.11989): one-step VJP + JVP method for per-token gradient scores.
- [Token Utility / FUW](https://arxiv.org/pdf/2610.18883): Fisher view of per-token utility; differs by ignoring cross-token coupling.
- [IER](https://arxiv.org/pdf/2609.24432): gradient noise in sampled-token OPD, relevant to K2.
- [Awesome On-Policy Distillation](https://github.com/chrisliu298/awesome-on-policy-distillation): check the geometry papers and GeoSD for overlap with our framing.
