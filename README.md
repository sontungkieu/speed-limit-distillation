# Speed Limit of Distillation

Phase 0 implementation of [the experiment plan](experiment-plan.md), using
Qwen3-1.7B non-thinking, Qwen3-4B-Instruct-2507 and the pinned 14,116-row DAPO input.
This is an independent Transformers/vLLM reconstruction using the existing
NeMo-RL v0.7 runtime dependencies. Equivalence to the authors' training backend
has not been established. A running job is not a reproduction pass.

The three single-GPU queues run:

1. Numerical tests and a two-step real-model canary, then 10 OPD calibration
   steps with separate generation, teacher-scoring and optimizer timings.
2. Two disjoint cache shards, covering every prompt once with a genuine initial
   student rollout (4,096-token cap), sampled teacher logp and FP32 teacher
   top-64 logp plus token IDs at every response position.
3. Semi-OPD, 100 steps, global batch 128, constant LR 1e-6, AdamW
   (0.9, 0.999), weight decay .01 and gradient clipping 1. FP32 master weights
   and optimizer state, BF16 computation; one sequence per gradient accumulation.
4. Independently queued full final evaluations at checkpoints 60/80/100,
   AIME24/25/26 + HMMT26, avg@32, 32,768-token cap. Training need not wait for
   final evaluation. The aggregate pass gate uses the equal-benchmark average
   and then the equal-checkpoint average, compared with .398 +/- .03.

The cache creates a fixed 2,048-prompt pool, a disjoint 256-prompt held-out set
and three seeded 64-rollout subsets. These held-out prompts are disjoint from
pool P for later repeated-S experiments; Phase 0's global Semi reproduction
still trains on all training prompts. Diagnostic KL uses teacher top-64
candidate sets, independently renormalized teacher/student distributions,
weighted equally over token states. Log sampled-token training loss separately.
The diagnostic global-training curve must not be called a repeated-S curve.
Measurements include step zero, every 10 updates, held-out KL and fresh 128
held-out rollouts at steps 50/100. Missing threshold crossings are censored;
never replace a missing crossing with a measured time.

`speed_limit.lanczos` implements a matrix-free VJP/JVP loss-gradient Gram
operator, Adam second-moment preconditioning and reorthogonalized Lanczos.
The numerical test compares with explicit tiny-model gradients and eigenvalues
at 1% tolerance. A LoRA/full-training equivalence check and a validated spectral
prediction are future research gates; a passing toy test does not establish them.

## Running with the existing remote runtime

No local model downloads or environment builds are needed. On each selected
node, use its existing runtime interpreter and set `PYTHONPATH` to this repo's
`src`. Each role receives one physical GPU UUID; the queue refuses a GPU with
compute processes or more than 1 GiB allocated. The controller resolves that
UUID to a numeric index for vLLM and each model process verifies the CUDA-visible
UUID against the requested physical GPU before loading any model. Provide fresh run/source paths.

```bash
PYTHONPATH="$SOURCE/src" "$INPUTS/runtime/envs/vllm/bin/python" \
  -m speed_limit.queue --source "$SOURCE" --inputs "$INPUTS" \
  --run "$RUN" --role 0 --gpu "$GPU_UUID" --revision "$GIT_COMMIT" --detach
```

Start roles 0, 1, 2, exactly once each, sharing the same run directory. Role 0
owns calibration and training; roles 1/2 own cache shards 0/1 and final evals
(60/100 and 80). Controllers detach from Jupyter, preserve task logs, record
child PIDs and heartbeat state, and stop their queue on failures. Unknown Bridge
commands are never retried. No Phase 1 work launches automatically.

Offline numerical checks (inside the existing runtime):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The source bundle contains no models, datasets, runtime, credentials or run
results. Publish the verified Git commit first, archive it, upload the archive
and its SHA-256 manifest to Hugging Face, then fetch the pinned HF revision on
nodes through the supplied proxy. Source and runtime have separate provenance.

A read-only evidence aggregator is available as `src/speed_limit/report.py RUN`.
It reports partial progress explicitly and separates cache generation, teacher
scoring, refit and updates. `--watch --output PATH --detach` maintains an atomic
summary on shared storage until the queue completes or a failure marker appears.
A newer reporting overlay can inspect a run frozen to an older training commit;
both source revisions remain in their respective receipts.

When the proxy cannot reach the HF binary CDN, `source-bundle.json` transports
the same Git archive as base64 through HF's regular text-file endpoint. The
deployer validates both Git revision and decoded archive SHA-256 before use.

Validate actual saved cache shards without loading a GPU model:

```bash
PYTHONPATH=src python -m speed_limit.cache --file /path/to/batch-00000.pt --expected-rows 32
```

This checks weights-only tensor deserialization, token alignment, candidate
probability mass and agreement between sampled logp and matching top-64 IDs.

Move a checkpoint evaluation to another allocated GPU with the detached launcher:

```bash
python scripts/run_checkpoint.py --inputs "$INPUTS" --runner-source "$RUNNER_SOURCE" \
  --training-run "$RUN" --output "$ATTEMPT" --step 100 --gpu "$GPU_UUID" \
  --revision "$LAUNCHER_COMMIT" --detach
```

The launcher reuses the qualified runner's environment, checks the checkpoint
ready marker and GPU availability, and records process, progress and completion
receipts in the fresh attempt directory. Add `--dev` for AIME24/25 avg@8.
When moving a job previously scheduled by `watch-and-launch-step100.sh`, use
`--retire-watch-run RUN --retirement-receipt FRESH_JSON` on that watcher's host.
The retirement checks the PID's command line before stopping the owned watcher.
To continue a stopped evaluator, add `--prior-result PRIOR_RESULT_DIR`.
The launcher validates the previous run settings and reuses complete four-prompt
batches. A partially written batch is generated again with the original seed.

## Concurrent exploratory K1 training

`speed_limit.fixed` repeats updates only on one of the three cached 64-rollout
sets from `splits.json`. The initial student, AdamW settings and sampled-token
sequence-sum loss match Setup. It measures token-weighted top64 KL on S at
step zero and every update, held-out KL every 10 steps, and 128 fresh held-out
rollouts every 50 steps and at termination. Fresh rollouts are diagnostics.
The run ends at 300 updates or the first measured A below .1.

`scripts/stack_k1.py` runs one set on each of roles 1/2/3, sweeping LR
1e-6, 3e-6, 1e-5, 3e-5 and 1e-4. Each run starts from the initial student.
Role 0 waits for the relocated Phase0 step100 evaluation, evaluates its dev
set, then serves the dev evaluation requests from completed K1 runs serially.
The user authorized this overlap while the Phase0 final benchmark gate is
pending. Run receipts retain that pending status. K1 interpretation follows
verification of the Phase0 model, objective and evaluation protocol.

Fixed-set checkpoints at steps 50, 100, 200 and 300, or early termination,
include model, optimizer and Python/NumPy/Torch/CUDA RNG state. Resume into a
fresh output directory with `python -m speed_limit.fixed ... --resume CHECKPOINT`.
The runner compares the saved set IDs, LR, optimizer and cache manifests,
restores optimizer/RNG state, and links the earlier absorption history.
The optimizer option also supports SGD with momentum .9 for the subsequent
best-two-LR comparison after the AdamW sweep is analyzed.

K1 dev outputs and logs live under the campaign `dev/` directory. The GPU-0
queue state records completion paths for every attempted dev evaluation.
To relocate a sweep, start a fresh campaign with `--previous-run PRIOR_CAMPAIGN`.
The training controller reuses completed runs and resumes incomplete runs from
their latest ready checkpoint. Pass `--training-roles` to the evaluation controller
when the destination campaign contains only a subset of the original roles.

## Bounded multi-node campaign

`scripts/campaign.py` prepares 15 K1 AdamW runs and five exploratory S0 top64
LR runs. Each GPU worker owns an isolated output tree and claims its assigned
jobs exclusively. Recovery workers can select a prior configuration by index,
then run deferred pilot configurations. `--pilot-set` extends a top64 pilot
to another fixed state set while preserving its LR and objective.
Each completed training run queues AIME24/25 dev evaluation. After all 15 K1
dev results, mean dev accuracy across S0/S1/S2 selects two LRs for six SGD runs.
Ties prefer the smaller LR. The top dev LR also schedules top64 on S1/S2.
This ranking is provisional until held-out KL and fresh-rollout length are
reviewed. The vOPD control-variate arm and K3 diagnostics use that provisional
rate for matched comparisons.
Workers accept one physical GPU UUID and an absolute UTC deadline. Each payload
runs in its own session under GNU timeout with TERM and a five-second KILL grace,
so payloads remain bounded if the controller disappears. Controllers stop
launching ten seconds before the deadline. Resume uses prior ready checkpoints
with the saved model, optimizer and RNG state. Source and output paths use the
relocated `/workspace/storage-shared/nlp/users/tungks` root.

`scripts/finish_sgd.py` completes the two unfinished controls from the 20-GPU
campaign. Use a fresh output directory and a verified host and GPU UUID per
role. Role 0 resumes S1 at LR 1e-6 and evaluates its final checkpoint. Role 1
evaluates the completed S0 checkpoint at the same LR, reusing complete dev
batches if present. Each role records its launch, current stage and results.

`scripts/run_control_variate.py` runs K2 at LR 1e-6, the best K1 dev rate and
a rate with existing sampled-token and top-64 comparisons on all three sets.
Roles 0–2 use the exact full-vocabulary vOPD KL baseline on S0–S2. Role 3 uses
the vOPD student top-20 baseline on S0 to compare the approximation against
the exact baseline. The comparison uses the same 64 cached rollouts per set,
AdamW, 300-step ceiling, stopping criterion, diagnostics and dev evaluation.
Like the other K2 arms, these updates reuse fixed sampled tokens. The on-policy
unbiasedness argument from arXiv:2605.07865 does not carry over to this replay
regime. The sampled-token reward uses cached teacher log probabilities, while
the baseline uses a frozen teacher forward pass on the same states.

`scripts/run_k3.py` runs a first real-model K3 diagnostic. Roles 0–2 each use
one B200 to measure the step-zero FP32 LoRA spectrum on one fixed state set.
Each role checks sparse candidate logits against the full output head and
checks Gram symmetry before starting Lanczos. The operator streams one
rollout at a time and uses per-token top-64 KL gradients with the same
rollout-count scaling as training. The predicted curve uses sqrt(2 KL) as
a scalar residual proxy and holds unresolved mass fixed. At step zero no
measured Adam second moments exist, so these spectra are unpreconditioned.
Role 3 fits rank-16 LoRA on S0 for 100 updates at LR 1e-6 and compares its
curve with the campaign's existing full-training top-64 curve. Adapter and
optimizer state are saved at steps 50 and 100. All roles reuse the campaign
cache. This diagnostic leaves checkpoint spectra and prediction validation
for subsequent runs.

Pass `--lora-lr` to `scripts/run_k3.py` to run an independent S0 LoRA
candidate at a different learning rate. The full fine-tuning reference stays
fixed at LR 1e-6 and must match the cache, state set, objective, and optimizer.
Rank candidates by absorption-curve RMSE on steps 1–50. Report agreement on
steps 51–100 separately, with absorption times and the step-100 gap. Use a
separate output directory for each candidate and retain the same rank, scale,
seed, and training recipe across candidates.

For checkpoint spectra, use `run_k3.py --schedule` with a JSON mapping GPU role
numbers to ordered lists of `lora` and `spectrum` stages. Every stage specifies
`task`, `set_index`, and `lr`. A `lora` stage also specifies `reference`. A
`spectrum` stage specifies `checkpoint`, `step`, and `preconditioner` (`none`
or `adam`). The GPU worker waits for the exact ready marker, checks checkpoint
lineage and parameter order, then measures the spectrum in FP32. Adam uses
the bias-corrected second moments saved at that checkpoint. Residual predictions
are local to the checkpoint and reuse the same KL proxy as at step zero.

`scripts/run_residual_k3.py` runs a bounded K3 diagnostic with a different
linearization. It differentiates the per-token residual sqrt(2 KL + epsilon)
and starts Lanczos at that residual. The tridiagonal projection therefore
acts on the full initial residual, including components outside the largest
50 modes. Step-zero forecasts use the same clipping, momentum, weight decay,
and sequence-sum batch normalization as the measured FP32 LoRA SGD controls.
They use no future gradient norms. The 40/60-iteration comparison diagnoses
projection error. Step-50/100 spectra show how the residual-weighted geometry
changes after SGD. The launcher takes an ordered per-GPU schedule, records its
deadline, and keeps each worker and its subprocesses under a 12-hour GNU timeout.
On nodes with an MPS daemon, pass its verified PID and control pipe with
`--mps-server` and `--mps-pipe`. Admission requires NVML visibility of all MPS
clients and a selected GPU with no client. The controller checks this before
detach and before every stage.
`--after-role` allows a queued worker to wait for a specific predecessor on
the same host and physical GPU. The wait counts toward the watchdog deadline,
and the GPU is checked again when the predecessor releases it.
An SGD stage may set `forecast_from` to a completed, matching step-zero
projection directory. The launcher writes the pre-run forecast before the
first SGD update, including for the `1e-3` follow-up rate.
Set `save_final_adapter: true` on an SGD stage to save the adapter at the
last measured step, with a ready marker for downstream evaluation.

### K3 learning-rate evaluation

Run a second, detached `run_residual_k3.py` queue with `--after-role` on each
of the same physical GPUs. Each `gate-eval` schedule entry names a replayed
`checkpoint`, its `trained_role` receipt and `source_stage`, and the `reference`
directory for the original SGD curve. A base-model entry omits these fields.
The replay must match the original training contract and differ by at most
0.001 in normalized KL across the entire 300-step trajectory.

The evaluator scores the 256 fixed held-out rollouts with the FP32 student.
It merges the replayed adapter into the student for BF16 vLLM generation and
uses the existing 480-sample AIME24/25 dev protocol. It also generates on the
same 128 held-out prompts per model and scores fresh top-64 KL and response
length. A final `gate-summary` stage collects all six matched runs and the
base model, and writes per-case and equal-set LR averages to `complete.json`.
