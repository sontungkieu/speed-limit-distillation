"""K1 repeated-state training with token-weighted absorption and resumable state."""
import argparse
import json
import random
from pathlib import Path
import time

import numpy as np
import torch

from .legacy import append_json, write_json
from .metrics import absorption_times
from .phase0 import Phase0, load_cache, load_rows, sha256


def capture_training_state(optimizer, step):
    numpy_state = np.random.get_state()
    return {
        'optimizer': optimizer.state_dict(), 'step': step,
        'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
        'cuda_rng_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'numpy_rng': [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])],
    }


def restore_training_state(optimizer, state):
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    if state['cuda_rng_all']:
        torch.cuda.set_rng_state_all(state['cuda_rng_all'])
    return int(state['step'])


def read_curve(path, through=None):
    if not path.exists():
        return []
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_step = {value['step']: value for value in values if through is None or value['step'] <= through}
    return [by_step[step] for step in sorted(by_step)]


class FixedSet(Phase0):
    def checkpoint(self, step):
        target = self.run / f'checkpoint-{step:04d}'
        target.mkdir()
        self.student.save_pretrained(target, safe_serialization=True)
        self.tokenizer.save_pretrained(target)
        torch.save(capture_training_state(self.optimizer, step), target / 'training_state.pt')
        write_json(target / 'receipt.json', {'step': step, 'run_id': self.args.run_id,
                   'time': time.time(), 'source_revision': self.args.revision})
        write_json(target / 'ready.json', {'step': step, 'time': time.time()})
        return target

    def train_fixed(self):
        args = self.args
        index, splits = load_cache(self.cache_directory, 2, self.order)
        state_ids = splits['state_sets'][args.set_index]
        fixed = load_rows(index, state_ids)
        heldout = load_rows(index, splits['heldout'])
        contract = {
            'schema_version': 1, 'objective': 'sampled_token_reverse_kl_sequence_sum',
            'set_index': args.set_index, 'row_ids': state_ids, 'heldout': splits['heldout'],
            'lr': args.lr, 'optimizer': args.optimizer, 'max_steps': args.steps,
            'cache_manifests': [sha256(self.cache_directory / f'shard-{i}' / 'complete.json')
                                for i in (0, 1)],
            'batch_rollouts': len(fixed), 'tokens': sum(len(r['response']) for r in fixed),
            'seed': args.seed, 'weight_decay': .01, 'clip': 1.,
            'scope': 'exploratory K1 concurrent with pending Phase0 final evaluation',
        }
        write_json(self.run / 'fixed-contract.json', contract)
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=args.lr,
                                             momentum=.9, weight_decay=.01)
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = args.lr
        start_step = 0
        history = []
        if args.resume:
            previous_run = args.resume.parent
            previous_contract = json.loads((previous_run / 'fixed-contract.json').read_text())
            if previous_contract != contract:
                raise ValueError('Resume contract differs from the saved fixed-state experiment')
            if not (args.resume / 'ready.json').exists():
                raise ValueError('Resume checkpoint lacks a ready marker')
            from transformers import AutoModelForCausalLM
            saved_model = AutoModelForCausalLM.from_pretrained(args.resume, dtype=torch.float32,
                attn_implementation='sdpa', local_files_only=True)
            self.student.load_state_dict(saved_model.state_dict())
            del saved_model
            state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=True)
            start_step = restore_training_state(self.optimizer, state)
            self.last_refit = -1
            baseline = json.loads((previous_run / 'baseline.json').read_text())
            history = read_curve(previous_run / 'absorption.jsonl', start_step)
            write_json(self.run / 'resume.json', {'checkpoint': str(args.resume), 'step': start_step})
        else:
            initial = self.measurement(fixed, 0, 'S')
            heldout_initial = self.measurement(heldout, 0, 'heldout')
            baseline = {'mean_kl_per_token': initial['mean_kl_per_token'],
                        'heldout_mean_kl': heldout_initial['mean_kl_per_token'],
                        'initial_cached_heldout_mean_length': sum(len(r['response']) for r in heldout[:128]) / 128}
            history = [{'step': 0, 'A': 1.}]
        if baseline['mean_kl_per_token'] <= 0:
            raise ValueError('Nonpositive initial KL')
        write_json(self.run / 'baseline.json', baseline)
        for observation in history:
            append_json(self.run / 'absorption.jsonl', observation)
        if args.resume:
            current = self.measurement(fixed, start_step, 'S')
            observation = {'step': start_step, 'A': current['mean_kl_per_token'] / baseline['mean_kl_per_token']}
            history = [row for row in history if row['step'] < start_step] + [observation]
            append_json(self.run / 'absorption.jsonl', observation)
        final_step = start_step
        fresh_step = None
        remaining_steps = range(start_step + 1, args.steps + 1) if history[-1]['A'] >= .1 else ()
        for step in remaining_steps:
            update = self.update(fixed)
            update.update(step=step, time=time.time())
            append_json(self.run / 'train.jsonl', update)
            measurement = self.measurement(fixed, step, 'S')
            observation = {'step': step, 'A': measurement['mean_kl_per_token'] / baseline['mean_kl_per_token']}
            history.append(observation)
            append_json(self.run / 'absorption.jsonl', observation)
            print('FIXED_STEP', args.set_index, args.lr, step, observation['A'], flush=True)
            if step % 10 == 0:
                self.measurement(heldout, step, 'heldout')
            if step % 50 == 0:
                self.measure_fresh(splits['heldout'][:128], step, baseline)
                fresh_step = step
            final_step = step
            if observation['A'] < .1:
                break
            if step == 50 or step % 100 == 0:
                self.checkpoint(step)
        if final_step % 10:
            self.measurement(heldout, final_step, 'heldout')
        if fresh_step != final_step:
            self.measure_fresh(splits['heldout'][:128], final_step, baseline)
        checkpoint = self.run / f'checkpoint-{final_step:04d}'
        if not (checkpoint / 'ready.json').exists():
            self.checkpoint(final_step)
        summary = absorption_times([row['A'] for row in history], [row['step'] for row in history])
        summary.update(set_index=args.set_index, lr=args.lr, optimizer=args.optimizer,
                       steps=final_step, source_revision=args.revision, checkpoint=str(checkpoint),
                       dev_evaluation_complete=False, elapsed_seconds=time.time() - self.started)
        write_json(self.run / 'complete.json', summary)
        write_json(self.run / 'dev-request.json', {'checkpoint': str(checkpoint),
            'step': final_step, 'training_run': str(self.run), 'mode': 'dev', 'time': time.time()})
        print('FIXED_COMPLETE', json.dumps(summary), flush=True)

    def measure_fresh(self, row_ids, step, baseline):
        self.refit(step)
        fresh = self.collect(row_ids, 100000 + step)
        self.measurement(fresh, step, 'fresh')
        mean_length = sum(len(row['response']) for row in fresh) / len(fresh)
        append_json(self.run / 'fresh.jsonl', {'step': step, 'generation_seconds': self.generation_seconds,
            'teacher_seconds': self.scoring_seconds, 'mean_response_tokens': mean_length,
            'length_ratio_to_initial_cache': mean_length / baseline['initial_cached_heldout_mean_length'],
            'cap_hit_fraction': sum(row['finish_reason'] == 'length' for row in fresh) / len(fresh)})
        torch.save(fresh, self.run / f'fresh-step-{step:04d}.pt')


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'output', 'cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--set-index', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--optimizer', choices=('adamw', 'sgd'), default='adamw')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if args.lr <= 0 or not 0 < args.steps <= 300:
        parser.error('Positive LR and step budget in 1..300 are required')
    args.task = 'train'
    args.seed = 0
    args.cap = 4096
    args.batch_size = 64
    args.checkpoint = None
    args.canary = False
    args.dev = False
    runner = FixedSet(args)
    runner.train_fixed()


if __name__ == '__main__':
    main()
