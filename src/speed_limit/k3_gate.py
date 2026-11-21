"""Evaluate a replayed residual-K3 adapter on cached and fresh held-out states."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from .k3 import cached_rows, load_lora_checkpoint, make_batch
from .legacy import write_json
from .phase0 import Phase0, distribution, sha256
from .spectral import CandidateLoss, enable_lora


FINAL_STEP = 300
REPLAY_TOLERANCE = 1e-3
FRESH_COUNT = 128


def read(path):
    return json.loads(path.read_text())


def checked_replay(checkpoint, reference):
    replay = read(checkpoint.parent / 'complete.json')
    original = read(reference / 'complete.json')
    contract = read(checkpoint.parent / 'contract.json')
    original_contract = read(reference / 'contract.json')
    for field in original_contract:
        if field != 'revision' and original_contract[field] != contract[field]:
            raise ValueError(f'Replay contract differs on {field}')
    if any(len(run['curve']) != FINAL_STEP + 1 for run in (replay, original)):
        raise ValueError('Both SGD trajectories must reach step 300')
    gap = max(abs(a - b) for a, b in zip(replay['curve'], original['curve'], strict=True))
    initial_gap = abs(replay['initial_mean_kl'] / original['initial_mean_kl'] - 1)
    if not math.isfinite(gap) or gap > REPLAY_TOLERANCE or initial_gap > REPLAY_TOLERANCE:
        raise ValueError(f'SGD replay diverged from original: curve gap {gap}, initial KL gap {initial_gap}')
    return contract, {'reference': str(reference), 'max_abs_A_gap': gap,
                      'initial_kl_relative_gap': initial_gap, 'A300': replay['curve'][-1],
                      'times': replay['times'], 'lr': contract['lr'], 'set_index': contract['set_index']}


def load_student(args):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    path = args.inputs / 'models/student'
    student = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                   attn_implementation='eager').to('cuda').eval()
    if not args.checkpoint:
        return student, [], None
    contract, replay = checked_replay(args.checkpoint, args.reference)
    layers = enable_lora(student)
    if str(path) != contract['model_path'] or layers != contract['adapter_layers']:
        raise ValueError('Evaluation student differs from replay configuration')
    replay['checkpoint'], _ = load_lora_checkpoint(student, args.checkpoint, FINAL_STEP, contract, 'none')
    return student, layers, replay


@torch.no_grad()
def measure(student, records):
    objective = CandidateLoss(student)
    total, tokens, per_rollout = 0., 0, []
    for record in records:
        if not record['response']:
            per_rollout.append(None)
            continue
        kl = objective(make_batch(record))
        if not torch.isfinite(kl).all() or kl.min() < -2e-5:
            raise RuntimeError('Invalid top-64 KL')
        total += float(kl.sum())
        tokens += kl.numel()
        per_rollout.append(float(kl.mean()))
    if not tokens:
        raise RuntimeError('No response tokens to score')
    return {'mean_kl_per_token': total / tokens, 'rollouts': len(records), 'tokens': tokens,
            'row_ids': [record['row_id'] for record in records], 'per_rollout_kl': per_rollout}


@torch.no_grad()
def merge_adapter(student, layers):
    for path in layers:
        layer = student.model.get_submodule(path)
        layer.base.weight.addmm_(layer.b, layer.a, alpha=layer.scale)
        student.model.set_submodule(path, layer.base)


def prepare(args):
    student, layers, replay = load_student(args)
    splits_path = args.cache / 'splits.json'
    splits = read(splits_path)
    rows = splits['heldout']
    if len(rows) != 256 or set(rows).intersection(splits['pool']):
        raise ValueError('Held-out split must have 256 rows disjoint from the training pool')
    records, manifests = cached_rows(args.cache, rows)
    split_sha = sha256(splits_path)
    if args.checkpoint:
        contract = read(args.checkpoint.parent / 'contract.json')
        if contract['splits_sha256'] != split_sha or contract['cache_manifests'] != manifests:
            raise ValueError('Evaluation cache differs from the training contract')
    result = measure(student, records)
    if layers:
        probe = measure(student, records[:2])['mean_kl_per_token']
        merge_adapter(student, layers)
        merged = measure(student, records[:2])['mean_kl_per_token']
        if abs(probe - merged) > 1e-3:
            raise ValueError(f'Merged LoRA KL probe differs: {probe} versus {merged}')
        result['fp32_merge_probe_gap'] = abs(probe - merged)
        student.to(device='cpu', dtype=torch.bfloat16).save_pretrained(args.output / 'merged', safe_serialization=True)
    write_json(args.output / 'heldout.json', {**result, 'replay': replay,
                                             'splits_sha256': split_sha, 'cache_manifests': manifests})
    print('K3_HELDOUT_COMPLETE', result['mean_kl_per_token'], flush=True)


def generate(args):
    result = read(args.output / 'heldout.json')
    step = FINAL_STEP if args.checkpoint else 0
    args.output = args.output / 'dev'
    args.checkpoint = args.output.parent / 'merged' if result['replay'] else None
    args.checkpoint_step, args.dev, args.seed, args.canary = step, True, 0, False
    args.batch_size, args.cap, args.steps, args.task = 128, 4096, 1, 'evaluate'
    runner = Phase0(args)
    runner.run_evaluation()
    rows = result['row_ids'][:FRESH_COUNT]
    prompts = [runner.render(runner.data[row]['prompt'], training=True) for row in rows]
    outputs = runner.generate(prompts, args.cap, 100000)
    records = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        answer, = output.outputs
        records.append({'row_id': row, 'prompt': prompt, 'response': list(answer.token_ids),
                        'finish_reason': answer.finish_reason})
    write_json(args.output.parent / 'fresh-rollouts.json', records)
    print('K3_FRESH_GENERATED', len(records), flush=True)


def score(args):
    from transformers import AutoModelForCausalLM

    heldout = read(args.output / 'heldout.json')
    records = read(args.output / 'fresh-rollouts.json')
    if [record['row_id'] for record in records] != heldout['row_ids'][:FRESH_COUNT]:
        raise ValueError('Fresh rollout IDs differ from the held-out subset')
    step = FINAL_STEP if args.checkpoint else 0
    dev = read(args.output / 'dev' / f'eval-step-{step:04d}.json')
    if (set(dev['benchmarks']) != {'aime24', 'aime25'} or dev['samples_per_problem'] != 8
            or any(item['total'] != 240 for item in dev['benchmarks'].values())):
        raise ValueError('Expected 480 matched AIME24/25 completions')
    student, _, replay = load_student(args)
    if replay != heldout['replay']:
        raise ValueError('Adapter changed between eval stages')
    teacher = AutoModelForCausalLM.from_pretrained(args.inputs / 'models/teacher', dtype=torch.bfloat16,
                                                  attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    for record in records:
        if record['response']:
            _, record['top64_ids'], record['top64_logprobs'] = distribution(teacher, record, teacher=True)
    del teacher
    lengths = [len(record['response']) for record in records]
    fresh = measure(student, records)
    fresh.update(mean_response_tokens=sum(lengths) / len(lengths),
                 capped=sum(record['finish_reason'] == 'length' for record in records),
                 empty=sum(length == 0 for length in lengths))
    write_json(args.output / 'complete.json', {'heldout': heldout, 'dev': dev, 'fresh': fresh})
    print('K3_GATE_CASE_COMPLETE', dev['mean_accuracy'], fresh['mean_response_tokens'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'generate', 'score'))
    for name in ('inputs', 'cache', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.reference):
        parser.error('Supply both the replayed adapter checkpoint and the original training run')
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
    if uuid != os.environ.get('SPEED_LIMIT_GPU_UUID', '').lower().removeprefix('gpu-'):
        raise RuntimeError('CUDA GPU UUID differs from requested device')
    if args.stage == 'prepare':
        args.output.mkdir(parents=True, exist_ok=False)
    {'prepare': prepare, 'generate': generate, 'score': score}[args.stage](args)


if __name__ == '__main__':
    main()
