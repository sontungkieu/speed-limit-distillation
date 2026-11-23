"""Project clipped SGD dynamics onto a Krylov space started at the KL residual."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from .k3 import cached_set, check_gram, load_lora_checkpoint, make_batch
from .lanczos import lanczos_top_eigenpairs
from .legacy import append_json, write_json
from .metrics import absorption_times
from .spectral import CandidateLoss, StreamedGram, enable_lora


RATES = (3e-5, 1e-4, 3e-4)
MOMENTUM = .9
WEIGHT_DECAY = .01
CLIP = 1.
EPS = 1e-8
STEPS = 300


class KLResidual(nn.Module):
    def __init__(self, kl):
        super().__init__()
        self.kl = kl

    def forward(self, batch):
        return (2 * self.kl(batch).clamp_min(0.) + EPS).sqrt()


def project_sgd(projection, initial_norm, rollout_count, state_count, lr, steps):
    """Clipped momentum SGD with the residual Jacobian held fixed at step zero.

    G is dr/dtheta, and projection represents G Gᵀ / rollout_count. At LoRA
    initialization, B=0 makes G theta_0=0. Decay therefore contributes
    weight_decay * (r_t-r_0) to the linearized change in residual space.
    """
    eigenvalues, directions = torch.linalg.eigh(projection.to(device='cpu', dtype=torch.float64))
    tolerance = 1e-5 * max(float(eigenvalues[-1]), 1.)
    if float(eigenvalues[0]) < -tolerance or eigenvalues[-1] <= 0:
        raise ValueError('Projected residual Gram must be positive semidefinite and nonzero')
    eigenvalues = eigenvalues.clamp_min(0.)
    initial = directions[0].clone()
    state = initial.clone()
    velocity = torch.zeros_like(initial)
    normalized_epsilon = state_count * EPS / initial_norm**2
    if normalized_epsilon >= 1.:
        raise ValueError('Initial KL mass is nonpositive')
    curve, clips = [1.], []
    for _ in range(steps):
        gradient_norm = initial_norm * math.sqrt(float((eigenvalues * state.square()).sum()) / rollout_count)
        factor = min(1., CLIP / (gradient_norm + 1e-6))
        velocity = MOMENTUM * velocity + factor * eigenvalues * state + WEIGHT_DECAY * (state - initial)
        state = state - lr * velocity
        a = (float(state.square().sum()) - normalized_epsilon) / (1. - normalized_epsilon)
        if not math.isfinite(a):
            break
        curve.append(max(0., a))
        clips.append(factor)
    return {'curve': curve, 'clip_factors': clips, 'times': absorption_times(torch.tensor(curve, dtype=torch.float64)),
            'steps_predicted': len(curve) - 1, 'lambda_max': float(eigenvalues[-1]),
            'residual_weights': directions[0].square().tolist()}


def spectrum(args, student, kl, records, batches):
    check = check_gram(kl, student, records)
    write_json(args.output / 'canary.json', check)
    residual = KLResidual(kl)
    gram = StreamedGram(residual, batches)
    with torch.no_grad():
        initial = torch.cat([residual(batch) for batch in batches]).detach()
    initial_norm = float(initial.norm())
    iteration = 0

    def measured(vector):
        nonlocal iteration
        started = time.monotonic()
        result = gram.matvec(vector)
        torch.cuda.synchronize()
        iteration += 1
        row = {'iteration': iteration, 'seconds': time.monotonic() - started,
               'max_memory_bytes': torch.cuda.max_memory_allocated()}
        append_json(args.output / 'iterations.jsonl', row)
        print('RESIDUAL_GRAM', args.set_index, args.step, iteration, row['seconds'], flush=True)
        return result

    projection = lanczos_top_eigenpairs(measured, initial.numel(), iterations=60, device='cuda',
                                        initial_vector=initial, return_projection=True).cpu()
    torch.save({'tridiagonal': projection, 'initial_residual': initial.cpu()}, args.output / 'projection.pt')
    values, directions = torch.linalg.eigh(projection.double())
    result = {'step': args.step, 'set_index': args.set_index, 'matvecs': iteration, 'initial_norm': initial_norm,
              'rollout_count': len(records), 'state_count': initial.numel(), 'lambda_max': float(values[-1]),
              'ritz_values': values.tolist(), 'residual_weights': directions[0].square().tolist()}
    if args.step == 0:
        forecasts = {}
        for lr in RATES:
            full = project_sgd(projection, initial_norm, len(records), initial.numel(), lr, STEPS)
            small = project_sgd(projection[:40, :40], initial_norm, len(records), initial.numel(), lr, STEPS)
            pairs = zip(full['curve'][:51], small['curve'][:51])
            full['projection_40_vs_full_max_gap_first_50'] = max(abs(a-b) for a,b in pairs)
            forecasts[f'{lr:.0e}'] = full
        result['forecasts'] = forecasts
    return result


def train(args, model, batches, parameters, contract):
    optimizer = torch.optim.SGD(parameters.values(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    count = sum(batch[2].shape[0] for batch in batches)

    @torch.no_grad()
    def mean_kl():
        value = sum(float(model(batch).sum()) for batch in batches) / count
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid measured fixed-set KL')
        return value

    initial = mean_kl()
    curve = [1.]
    checkpoints = set(args.checkpoint_steps)
    if args.save_checkpoints:
        checkpoints.update((50, 100))
    earliest_stop = max(50, *checkpoints) if checkpoints else 50

    def save_adapter(step):
        folder = args.output / f'checkpoint-{step:04d}'
        folder.mkdir()
        torch.save({name: value.detach().cpu() for name, value in parameters.items()}, folder / 'adapter.pt')
        write_json(folder / 'ready.json', {'step': step, 'revision': contract['revision']})

    append_json(args.output / 'absorption.jsonl', {'step': 0, 'A': 1., 'mean_kl': initial})
    for step in range(1, STEPS + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            (model(batch).sum() / len(batches)).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters.values(), CLIP, error_if_nonfinite=True)
        optimizer.step()
        kl = mean_kl()
        curve.append(kl / initial)
        row = {'step': step, 'A': curve[-1], 'mean_kl': kl, 'gradient_norm': float(norm),
               'clip_factor': min(1., CLIP / (float(norm) + 1e-6)), 'seconds': time.monotonic() - started}
        append_json(args.output / 'absorption.jsonl', row)
        print('RESIDUAL_SGD', args.set_index, args.lr, step, row['A'], row['seconds'], flush=True)
        if step in checkpoints:
            save_adapter(step)
        if curve[-1] < .1 and step >= earliest_stop:
            break
    if args.save_final_adapter and step not in checkpoints:
        save_adapter(step)
    return {'set_index': args.set_index, 'lr': args.lr, 'curve': curve, 'initial_mean_kl': initial,
            'times': absorption_times(torch.tensor(curve, dtype=torch.float64))}


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', choices=('spectrum', 'sgd'), required=True)
    parser.add_argument('--set-index', type=int, choices=range(3), required=True)
    parser.add_argument('--lr', type=float, choices=(*RATES, 1e-3), default=1e-4)
    parser.add_argument('--forecast-from', type=Path)
    parser.add_argument('--step', type=int, choices=(0, 50, 100), default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--save-checkpoints', action='store_true')
    parser.add_argument('--save-final-adapter', action='store_true')
    parser.add_argument('--checkpoint-steps', type=int, nargs='+', choices=range(1, STEPS + 1), default=())
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.step) or (args.checkpoint and args.task != 'spectrum'):
        parser.error('Provide a checkpoint path with a nonzero spectrum step')
    if args.forecast_from and args.task != 'sgd':
        parser.error('A saved forecast projection is only used for an SGD comparison')
    if (args.save_final_adapter or args.checkpoint_steps) and args.task != 'sgd':
        parser.error('Only SGD saves adapters')
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
    if uuid != os.environ.get('SPEED_LIMIT_GPU_UUID', '').lower().removeprefix('gpu-'):
        raise RuntimeError('CUDA GPU UUID differs from requested device')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    print('RESIDUAL_GPU', uuid, args.task, args.set_index, flush=True)
    records, metadata = cached_set(args.cache, args.set_index)
    from transformers import AutoModelForCausalLM
    path = args.inputs / 'models/student'
    student = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                   attn_implementation='eager').to('cuda').eval()
    layers = enable_lora(student)
    contract = {**metadata, 'task': args.task, 'set_index': args.set_index, 'lr': args.lr,
                'revision': args.revision, 'model_path': str(path), 'seed': 0, 'adapter_rank': 16,
                'adapter_alpha': 32, 'adapter_layers': layers, 'objective': 'teacher_top64_reverse_kl',
                'precision': 'FP32', 'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY, 'clip': CLIP}
    if args.checkpoint:
        contract['checkpoint'], _ = load_lora_checkpoint(student, args.checkpoint, args.step, contract, 'none')
    write_json(args.output / 'contract.json', contract)
    if args.forecast_from:
        reference = json.loads((args.forecast_from / 'contract.json').read_text())
        receipt = json.loads((args.forecast_from / 'complete.json').read_text())
        if reference['task'] != 'spectrum' or receipt['step'] != 0 or receipt['matvecs'] != 60:
            raise ValueError('Expected a completed step-zero projection')
        for field in ('set_index', 'row_ids', 'cache_manifests', 'model_path', 'seed', 'precision',
                      'objective', 'adapter_rank', 'adapter_alpha', 'adapter_layers',
                      'momentum', 'weight_decay', 'clip'):
            if reference[field] != contract[field]:
                raise ValueError(f'Forecast projection differs on {field}')
        if receipt['rollout_count'] != len(records) or receipt['state_count'] != sum(len(r['response']) for r in records):
            raise ValueError('Projection state count differs from the training set')
        projection = torch.load(args.forecast_from / 'projection.pt', map_location='cpu', weights_only=True)
        full = project_sgd(projection['tridiagonal'], receipt['initial_norm'], receipt['rollout_count'],
                           receipt['state_count'], args.lr, STEPS)
        short = project_sgd(projection['tridiagonal'][:40, :40], receipt['initial_norm'],
                            receipt['rollout_count'], receipt['state_count'], args.lr, STEPS)
        full['projection_40_vs_full_max_gap_first_50'] = max(
            abs(a - b) for a, b in zip(full['curve'][:51], short['curve'][:51]))
        write_json(args.output / 'forecast.json', {
            **full, 'source': str(args.forecast_from), 'source_revision': reference['revision'],
            'set_index': args.set_index, 'lr': args.lr,
        })
    print('RESIDUAL_READY', args.task, args.set_index, args.step, flush=True)
    model = CandidateLoss(student)
    batches = [make_batch(record) for record in records]
    params = {name: value for name,value in student.named_parameters() if value.requires_grad}
    if args.task == 'sgd':
        result = train(args, model, batches, params, contract)
    else:
        result = spectrum(args, student, model, records, batches)
    write_json(args.output / 'complete.json', result)
    print('RESIDUAL_COMPLETE', args.task, args.set_index, args.step, flush=True)


if __name__ == '__main__':
    main()
