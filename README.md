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
