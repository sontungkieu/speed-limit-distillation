"""K3 LoRA curves and checkpoint spectra using cached teacher support."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from .fixed import capture_training_state, read_curve
from .lanczos import adam_preconditioner, lanczos_top_eigenpairs
from .legacy import append_json, write_json
from .metrics import absorption_times
from .phase0 import distribution, load_rows, sha256, top64_kl
from .spectral import CandidateLoss, StreamedGram, enable_lora, residual_prediction


def cached_set(cache, set_index):
    splits_path = cache / 'splits.json'
    splits = json.loads(splits_path.read_text())
    pool = splits['pool']
    rows = splits['state_sets'][set_index]
    if (len(pool) != 2048 or len(set(pool)) != 2048 or set(pool) & set(splits['heldout'])
            or rows != random.Random(set_index).sample(pool, 64)):
        raise ValueError('Fixed state-set split differs from the campaign')
    records, fingerprints = cached_rows(cache, rows)
    return records, {'row_ids': rows, 'cache_manifests': fingerprints, 'splits_sha256': sha256(splits_path)}


def cached_rows(cache, rows):
    if len(rows) != len(set(rows)) or not rows:
        raise ValueError('Expected distinct cached rollout row IDs')
    selection = set(rows)
    index = {}
    fingerprints = []
    for shard in range(2):
        path = cache / f'shard-{shard}' / 'complete.json'
        manifest = json.loads(path.read_text())
        if (manifest['shard'], manifest['shards'], manifest['seed'], manifest['cap']) != (shard, 2, 0, 4096):
            raise ValueError('Cache manifest differs from the campaign')
        fingerprints.append(sha256(path))
        for entry in manifest['entries']:
            included = selection.intersection(entry['rows'])
            if not included:
                continue
            source = cache / entry['file']
            if included.intersection(index) or sha256(source) != entry['sha256']:
                raise ValueError(f'Duplicate row or cache hash mismatch in {source}')
            index.update((row, source) for row in included)
    if len(index) != len(rows):
        raise ValueError('Incomplete cached rollout selection')
    return load_rows(index, rows), fingerprints


def make_batch(record, limit=None):
    response = record['response'][:limit]
    return (torch.tensor([record['prompt'] + response], dtype=torch.long, device='cuda'),
            len(record['prompt']) - 1, record['top64_ids'][:limit].to('cuda', dtype=torch.long),
            record['top64_logprobs'][:limit].to('cuda'))


def check_gram(model, student, records, preconditioner=None):
    batches = [make_batch(record, 32) for record in records[:2]]
    with torch.no_grad():
        for batch in batches:
            tokens, start, ids, teacher = batch
            logits = student(input_ids=tokens, use_cache=False).logits[0, start:start + len(ids)]
            reference = top64_kl(logits.gather(-1, ids), teacher)
            torch.testing.assert_close(model(batch), reference, atol=5e-5, rtol=1e-4)
    gram = StreamedGram(model, batches, preconditioner)
    u, v = torch.randn(2, sum(gram.sizes), device='cuda')
    started = time.monotonic()
    ku, kv = gram.matvec(u), gram.matvec(v)
    left, right = u @ kv, v @ ku
    torch.testing.assert_close(left, right, atol=1e-4, rtol=2e-3)
    if not torch.isfinite(ku).all() or u @ ku < 0:
        raise RuntimeError('Nonfinite or non-PSD real-model Gram canary')
    return {'seconds': time.monotonic() - started, 'tokens': sum(gram.sizes),
            'symmetry_absolute_error': float((left - right).abs())}


def checkpoint_preconditioner(parameters, optimizer, step, lr):
    groups = optimizer['param_groups']
    if len(groups) != 1 or groups[0]['lr'] != lr:
        raise ValueError('Checkpoint optimizer layout or learning rate differs')
    group = groups[0]
    ids = group['params']
    if len(set(ids)) != len(parameters) or set(ids) != set(optimizer['state']):
        raise ValueError('Checkpoint optimizer parameter IDs differ')
    moments = {}
    for (name, parameter), state_id in zip(parameters.items(), ids, strict=True):
        state = optimizer['state'][state_id]
        moment = state['exp_avg_sq']
        if (float(state['step']) != step or moment.shape != parameter.shape
                or moment.dtype != torch.float32 or not torch.isfinite(moment).all()
                or (moment < 0).any()):
            raise ValueError(f'Invalid Adam second moment for {name}')
        moments[name] = moment.to(parameter.device)
    return adam_preconditioner(parameters, moments, step, beta2=group['betas'][1], eps=group['eps'])


def load_lora_checkpoint(student, checkpoint, step, contract, mode):
    ready = json.loads((checkpoint / 'ready.json').read_text())
    previous = json.loads((checkpoint.parent / 'contract.json').read_text())
    if (ready['step'] != step or ready['revision'] != previous['revision']
            or previous['task'] not in ('lora', 'sgd')):
        raise ValueError('Checkpoint receipt does not match the requested training step')
    for field in ('row_ids', 'cache_manifests', 'splits_sha256', 'set_index', 'lr', 'model_path',
                  'adapter_rank', 'adapter_alpha', 'adapter_layers', 'seed', 'objective'):
        if previous[field] != contract[field]:
            raise ValueError(f'Checkpoint training contract differs on {field}')
    parameters = {name: value for name, value in student.named_parameters() if value.requires_grad}
    adapter_path = checkpoint / 'adapter.pt'
    adapter = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if tuple(adapter) != tuple(parameters):
        raise ValueError('Checkpoint adapter parameter ordering differs')
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = adapter[name]
            if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f'Invalid adapter parameter {name}')
            parameter.copy_(value)
    receipt = {'path': str(checkpoint), 'training_revision': previous['revision'],
               'adapter_sha256': sha256(adapter_path), 'step': step}
    if mode == 'none':
        return receipt, None
    state_path = checkpoint / 'training_state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    if state['step'] != step:
        raise ValueError('Optimizer checkpoint step differs')
    preconditioner = checkpoint_preconditioner(parameters, state['optimizer'], step, contract['lr'])
    receipt['optimizer_sha256'] = sha256(state_path)
    return receipt, {'student.' + name: value for name, value in preconditioner.items()}


def spectrum(args, student, records, preconditioner=None):
    student.eval()
    model = CandidateLoss(student)
    check = check_gram(model, student, records, preconditioner)
    write_json(args.output / 'canary.json', check)
    print('K3_CANARY', json.dumps(check), flush=True)
    batches = [make_batch(record) for record in records]
    gram = StreamedGram(model, batches, preconditioner)
    with torch.no_grad():
        losses = torch.cat([model(batch) for batch in batches])
    if not torch.isfinite(losses).all() or losses.min() < -5e-5:
        raise ValueError('Invalid per-state KL')
    iteration = 0

    def measured_matvec(u):
        nonlocal iteration
        started = time.monotonic()
        value = gram.matvec(u)
        torch.cuda.synchronize()
        iteration += 1
        result = {'iteration': iteration, 'seconds': time.monotonic() - started,
                  'max_memory_allocated_bytes': torch.cuda.max_memory_allocated()}
        append_json(args.output / 'iterations.jsonl', result)
        print('K3_MATVEC', json.dumps(result), flush=True)
        return value

    eigenvalues, eigenvectors = lanczos_top_eigenpairs(
        measured_matvec, losses.numel(), iterations=60, device='cuda', seed=args.set_index)
    eigenvalues, eigenvectors = eigenvalues[:50], eigenvectors[:, :50]
    weights, unresolved, predicted = residual_prediction(losses, eigenvalues, eigenvectors, args.lr, 300)
    if float(eigenvalues[0]) <= 0:
        raise RuntimeError('Spectrum has no positive measured eigenvalue')
    positive = eigenvalues.double().clamp_min(0)
    torch.save({'eigenvalues': eigenvalues.cpu(), 'eigenvectors': eigenvectors.cpu(),
                'state_kl': losses.cpu()}, args.output / 'spectrum.pt')
    curve = predicted.tolist() if predicted is not None else None
    return {'matvecs': iteration, 'eigenvalues': eigenvalues.tolist(),
            'effective_rank_top_subspace': float(positive.sum().square() / positive.square().sum()),
            'initial_mean_kl': float(losses.mean()), 'residual_proxy': 'sqrt(2 * top64 reverse KL)',
            'kernel_normalization': 'G G^T / rollout_count for unscaled per-token gradients',
            'residual_weights': weights.tolist(), 'unresolved_residual_mass': float(unresolved),
            'predicted_A_unresolved_mass_fixed': curve,
            'predicted_times': absorption_times(curve) if curve is not None else None,
            'prediction_learning_rate': args.lr,
            'preconditioning': 'adam_second_moment' if preconditioner is not None else 'none',
            'parameter_step': args.checkpoint_step or 0}


def curve_agreement(curve, reference):
    differences = [observed - target for observed, target in zip(curve, reference, strict=True)]
    if len(differences) != 101 or not all(math.isfinite(value) for value in differences):
        raise ValueError('Expected finite curves on steps 0–100')
    return {
        'calibration_rmse_1_50': math.sqrt(sum(value**2 for value in differences[1:51]) / 50),
        'validation_rmse_51_100': math.sqrt(sum(value**2 for value in differences[51:101]) / 50),
        'difference_at_100': differences[100],
    }


def fit_lora(args, student, records, contract):
    reference_contract = json.loads((args.reference / 'fixed-contract.json').read_text())
    for field in ('row_ids', 'cache_manifests', 'set_index'):
        if reference_contract[field] != contract[field]:
            raise ValueError(f'Reference uses a different {field}')
    reference_lr = args.reference_lr if args.reference_lr is not None else args.lr
    if reference_contract['lr'] != reference_lr:
        raise ValueError('Reference learning rate differs from the declared rate')
    if (reference_contract['objective'] != 'top64_renormalized_reverse_kl_sequence_sum'
            or reference_contract['optimizer'] != 'adamw'):
        raise ValueError('Reference objective or optimizer differs')
    reference = {row['step']: row['A'] for row in read_curve(args.reference / 'absorption.jsonl', 100)}
    if set(reference) != set(range(101)):
        raise ValueError('Full fine-tuning reference must cover steps 0–100')
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = {name: value for name, value in student.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, betas=(.9, .999),
                                  weight_decay=.01, fused=True)
    tokens = sum(len(record['response']) for record in records)

    @torch.no_grad()
    def measure():
        student.eval()
        total = 0.
        for record in records:
            _, candidates = distribution(student, record)
            total += float(top64_kl(candidates, record['top64_logprobs']).sum())
        student.train()
        if not 0 < total < float('inf'):
            raise ValueError('Invalid measured fixed-set KL')
        return total / tokens

    initial = measure()
    curve = [1.]
    append_json(args.output / 'absorption.jsonl', {'step': 0, 'A': 1., 'mean_kl': initial})
    started = time.monotonic()
    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        for record in records:
            _, candidates = distribution(student, record, gradients=True)
            loss = top64_kl(candidates, record['top64_logprobs']).sum() / len(records)
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
        optimizer.step()
        kl = measure()
        curve.append(kl / initial)
        row = {'step': step, 'A': curve[-1], 'mean_kl': kl, 'grad_norm': float(grad_norm),
               'full_finetuning_A': reference.get(step), 'elapsed_seconds': time.monotonic() - started}
        append_json(args.output / 'absorption.jsonl', row)
        print('K3_LORA_STEP', json.dumps(row), flush=True)
        if step in (50, 100):
            folder = args.output / f'checkpoint-{step:04d}'
            folder.mkdir()
            torch.save({name: value.detach().cpu() for name, value in params.items()}, folder / 'adapter.pt')
            torch.save(capture_training_state(optimizer, step), folder / 'training_state.pt')
            write_json(folder / 'ready.json', {'step': step, 'revision': args.revision})
    target_curve = [reference[step] for step in range(len(curve))]
    return {'full_finetuning_reference': str(args.reference), 'reference_lr': reference_lr,
            'lora_lr': args.lr, **curve_agreement(curve, target_curve),
            'comparison': [{'step': step, 'lora_A': curve[step], 'full_finetuning_A': reference[step]}
                           for step in (0, 50, 100)], 'lora_times': absorption_times(curve),
            'reference_times': absorption_times(target_curve)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-lr', type=float)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-step', type=int, choices=(50, 100))
    parser.add_argument('--preconditioner', choices=('none', 'adam'), default='none')
    parser.add_argument('--task', choices=('spectrum', 'lora'), required=True)
    parser.add_argument('--set-index', type=int, choices=range(3), required=True)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if args.task == 'lora' and args.reference is None:
        parser.error('LoRA comparison requires the full fine-tuning result path')
    if bool(args.checkpoint) != bool(args.checkpoint_step) or (args.checkpoint and args.task != 'spectrum'):
        parser.error('A spectrum checkpoint requires both its path and step')
    if args.preconditioner == 'adam' and args.checkpoint is None:
        parser.error('Adam preconditioning requires saved optimizer moments')
    if any(rate is not None and (not math.isfinite(rate) or rate <= 0)
           for rate in (args.lr, args.reference_lr)):
        parser.error('Learning rates must be finite and positive')
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
    if uuid != os.environ.get('SPEED_LIMIT_GPU_UUID', '').lower().removeprefix('gpu-'):
        raise RuntimeError('CUDA GPU UUID differs from requested device')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    print('K3_GPU', uuid, args.task, args.set_index, flush=True)
    records, metadata = cached_set(args.cache, args.set_index)
    print('K3_CACHE', len(records), sum(len(record['response']) for record in records), flush=True)
    from transformers import AutoModelForCausalLM
    model_path = args.inputs / 'models/student'
    student = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.float32,
        attn_implementation='eager' if args.task == 'spectrum' else 'sdpa').to('cuda')
    layers = enable_lora(student)
    contract = {**metadata, 'revision': args.revision, 'set_index': args.set_index, 'task': args.task,
                'lr': args.lr, 'reference_lr': args.reference_lr,
                'objective': 'teacher_top64_reverse_kl', 'model_path': str(model_path),
                'seed': 0, 'adapter_rank': 16, 'adapter_alpha': 32, 'adapter_layers': layers,
                'rollouts': len(records), 'tokens': sum(len(record['response']) for record in records),
                'precision': 'FP32' if args.task == 'spectrum' else 'FP32 weights and BF16 forward'}
    preconditioner = None
    if args.checkpoint is not None:
        checkpoint, preconditioner = load_lora_checkpoint(
            student, args.checkpoint, args.checkpoint_step, contract, args.preconditioner)
        contract.update(checkpoint=checkpoint, preconditioner=args.preconditioner)
        print('K3_CHECKPOINT', json.dumps(checkpoint), flush=True)
    write_json(args.output / 'contract.json', contract)
    start = time.monotonic()
    results = (spectrum(args, student, records, preconditioner) if args.task == 'spectrum'
               else fit_lora(args, student, records, contract))
    write_json(args.output / 'complete.json', {**results, 'elapsed_seconds': time.monotonic() - start})
    print('K3_COMPLETE', args.task, args.set_index, flush=True)


if __name__ == '__main__':
    main()
