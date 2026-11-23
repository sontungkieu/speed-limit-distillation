"""Persist the LR/checkpoint comparison as complete cases become available."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def save(path, result):
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False))
    pending.replace(path)


def metrics(result):
    return {'heldout_kl': result['heldout']['mean_kl_per_token'], 'dev_accuracy': result['dev']['mean_accuracy'],
            'fresh_mean_tokens': result['fresh']['mean_response_tokens'], 'fresh_kl': result['fresh']['mean_kl_per_token']}


def verified_metrics(result, baseline):
    heldout = result['heldout']
    if (heldout['rollouts'] != 256 or result['fresh']['rollouts'] != 128
            or heldout['row_ids'] != baseline['row_ids']
            or heldout['splits_sha256'] != baseline['splits_sha256']
            or heldout['cache_manifests'] != baseline['cache_manifests']
            or result['fresh']['row_ids'] != baseline['row_ids'][:128]
            or set(result['dev']['benchmarks']) != {'aime24', 'aime25'}
            or result['dev']['samples_per_problem'] != 8
            or any(b['total'] != 240 for b in result['dev']['benchmarks'].values())):
        raise ValueError('Checkpoint metrics do not match the baseline evaluation contract')
    return metrics(result)


def report(root, schedule, baseline, gate1):
    baseline_metrics = verified_metrics(baseline, baseline['heldout'])
    cases = {}
    missing, errors, queues = [], {}, {}
    completed = 0
    expected = 0
    for role, stages in schedule.items():
        work = root / f'gpu-{role}'
        state = work / f'role-{role}.json'
        queues[role] = read(state) if state.is_file() else {'phase': 'waiting_to_start'}
        for stage in stages:
            if stage['task'] != 'gate-eval':
                continue
            expected += 1
            path = work / stage['name'] / 'complete.json'
            if not path.is_file():
                missing.append(stage['name'])
                continue
            try:
                result = read(path)
                replay = result['heldout']['replay']
                step = replay['checkpoint']['step']
                if (step != stage['checkpoint_step'] or replay['checkpoint']['path'] != stage['checkpoint']
                        or replay['reference'] != stage['reference']):
                    raise ValueError('Receipt differs from the scheduled checkpoint')
                cases[replay['lr'], step, replay['set_index']] = {
                    **verified_metrics(result, baseline['heldout']), 'receipt': str(path),
                    'A': replay.get('A_at_checkpoint', replay['A300'])}
                completed += 1
            except (KeyError, ValueError) as error:
                errors[stage['name']] = str(error)
    if gate1:
        for item in gate1['cases'].values():
            replay = item['replay']
            if replay is not None:
                cases[replay['lr'], 300, replay['set_index']] = {
                    **{k: item[k] for k in baseline_metrics}, 'receipt': item['receipt'], 'A': replay['A300']}
    comparisons = []
    for lr, step in sorted({(lr, step) for lr, step, _ in cases}):
        available = {index: cases[lr, step, index] for index in range(3) if (lr, step, index) in cases}
        entry = {'lr': lr, 'step': step, 'sets': available, 'matched_sets': len(available)}
        if len(available) == 3:
            averages = {metric: mean(x[metric] for x in available.values()) for metric in (*baseline_metrics, 'A')}
            checks = {'heldout_kl_improves': averages['heldout_kl'] < baseline_metrics['heldout_kl'],
                      'no_dev_accuracy_loss': averages['dev_accuracy'] >= baseline_metrics['dev_accuracy'],
                      'no_fresh_length_increase': averages['fresh_mean_tokens'] <= baseline_metrics['fresh_mean_tokens']}
            entry.update(means=averages, quality_checks=checks, passes_quality=all(checks.values()),
                         absorbed_half_on_all_sets=all(x['A'] < .5 for x in available.values()))
        comparisons.append(entry)
    phases = {role: state['phase'] for role, state in queues.items()}
    return {'completed': completed, 'expected': expected, 'remaining': missing, 'invalid': errors,
            'phases': phases, 'failures': {role: state.get('failures') or {} for role, state in queues.items()},
            'baseline': baseline_metrics, 'original_gate1': gate1['decision'] if gate1 else None,
            'comparisons': comparisons, 'updated_at': time.time()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'schedule', 'baseline', 'gate1', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        args.output.mkdir(exist_ok=False)
        argv = [sys.executable, '-B', str(Path(__file__).resolve()),
                *(arg for arg in sys.argv[1:] if arg != '--detach')]
        with (args.output / 'controller.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': child.pid, 'deadline': args.deadline, 'report': str(args.output / 'progress.json')}
        save(args.output / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
        return
    schedule, baseline = read(args.schedule), read(args.baseline)
    while True:
        decision = read(args.gate1) if args.gate1.is_file() else None
        result = report(args.run, schedule, baseline, decision)
        worker_done = all(phase in ('complete', 'failed') for phase in result['phases'].values())
        expired = time.time() >= args.deadline
        result['status'] = ('complete' if result['completed'] == result['expected'] else
                            'partial' if worker_done or expired else 'running')
        save(args.output / 'progress.json', result)
        if result['status'] != 'running':
            save(args.output / 'final.json', result)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
