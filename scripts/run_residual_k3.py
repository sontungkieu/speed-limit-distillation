"""Run the ordered K3 residual/SGD stages on one verified GPU."""
import argparse
import csv
import json
from statistics import mean
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def gpu_preflight(args):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
    matches = [row for row in csv.reader(inventory.splitlines(), skipinitialspace=True) if row[0] == args.gpu]
    if len(matches) != 1 or int(matches[0][1]) > 1024:
        raise RuntimeError('Selected GPU UUID is absent or has an allocation above 1 GiB')
    raw_apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader'], text=True)
    apps = list(csv.reader(raw_apps.splitlines(), skipinitialspace=True))
    if args.mps_server is not None:
        server = str(args.mps_server)
        process_ids = {pid for _, pid, _ in apps if pid != server}
        commands = f'get_server_list\nget_client_list {server}\n'
        mps = subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input=commands, text=True,
                             capture_output=True, timeout=5, check=True,
                             env={**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': str(args.mps_pipe)})
        reports = mps.stdout.split()
        if mps.stderr.strip() or not reports or reports[0] != server or not set(reports[1:]) <= process_ids:
            raise RuntimeError('MPS daemon or all of its clients could not be identified')
        apps = [row for row in apps if row[1:] != [server, 'nvidia-cuda-mps-server']]
    if any(uuid == args.gpu for uuid, _, _ in apps):
        raise RuntimeError('Selected GPU has a compute client')
    return inventory


def predecessor(args):
    state = json.loads(args.after_role.read_text())
    if state['hostname'] != args.host or state['gpu'] != args.gpu:
        raise RuntimeError('Predecessor is not on the same host and physical GPU')
    return state


def gate_eval(queue, stage, path):
    args = queue.a
    command = [queue.python, '-m', 'speed_limit.k3_gate']
    options = ['--inputs', str(args.inputs), '--cache', str(args.cache), '--output', str(path)]
    if 'checkpoint' in stage:
        checkpoint = Path(stage['checkpoint'])
        trained_role = Path(stage['trained_role'])
        source_stage = stage.get('source_stage', checkpoint.parent.name)
        ready = checkpoint / 'ready.json'
        complete = checkpoint.parent / 'complete.json'
        while not (ready.is_file() and complete.is_file()):
            state = json.loads(trained_role.read_text())
            if source_stage in (state.get('failures') or {}) or state['phase'] in ('complete', 'failed'):
                if ready.is_file() and complete.is_file():
                    break
                raise RuntimeError(f'Training role ended without completed checkpoint: {checkpoint}')
            queue.save(phase=f"wait-{stage['name']}", waiting_for=[str(ready), str(complete)])
            time.sleep(10)
        options += ['--checkpoint', str(checkpoint), '--reference', stage['reference'],
                    '--checkpoint-step', str(stage.get('checkpoint_step', 300))]
    if stage.get('prior'):
        options += ['--prior', stage['prior']]
    if stage.get('temporary_merge'):
        options.append('--temporary-merge')
    for phase in ('prepare', 'generate', 'score'):
        queue.check_gpu()
        queue.invoke(f"{stage['name']}-{phase}", command=[*command, phase, *options])


def gate_summary(queue, stage, path):
    from speed_limit.queue import save

    root = queue.a.run.parent
    schedules = json.loads(queue.a.schedule.read_text())
    cases = {name: (Path(entry['receipt']), Path(entry['state']) if entry.get('state') else None)
             for name, entry in stage.get('external_cases', {}).items()}
    for role, jobs in schedules.items():
        case_root = root / f'gpu-{role}'
        for job in jobs:
            if job['task'] == 'gate-eval':
                name = job['name']
                if name in cases:
                    raise ValueError(f'Duplicate evaluation case: {name}')
                cases[name] = (case_root / name / 'complete.json', case_root / f'role-{role}.json')
    metrics = {}
    heldout_identity = None
    for name, (source, state_file) in cases.items():
        while not source.is_file():
            if state_file is None:
                raise ValueError(f'External evaluation receipt is missing: {source}')
            if state_file.is_file():
                state = json.loads(state_file.read_text())
                if name in (state.get('failures') or {}) or state['phase'] in ('complete', 'failed'):
                    if source.is_file():
                        break
                    raise RuntimeError(f'Evaluation ended without results: {source}')
            queue.save(phase=stage['name'], waiting_for=[str(source)])
            time.sleep(15)
        result = json.loads(source.read_text())
        current_identity = tuple(result['heldout'][key] for key in ('splits_sha256', 'cache_manifests', 'row_ids'))
        if heldout_identity is not None and current_identity != heldout_identity:
            raise ValueError('Gate cases use different held-out splits or cache manifests')
        heldout_identity = current_identity
        if (result['heldout']['rollouts'] != 256 or result['fresh']['rollouts'] != 128
                or result['fresh']['row_ids'] != result['heldout']['row_ids'][:128]
                or set(result['dev']['benchmarks']) != {'aime24', 'aime25'}
                or result['dev']['samples_per_problem'] != 8
                or any(bench['total'] != 240 for bench in result['dev']['benchmarks'].values())):
            raise ValueError(f'Incomplete held-out, fresh or dev evaluation: {source}')
        metrics[name] = {'receipt': str(source), 'replay': result['heldout']['replay'],
                         'heldout_kl': result['heldout']['mean_kl_per_token'],
                         'dev_accuracy': result['dev']['mean_accuracy'], 'benchmarks': result['dev']['benchmarks'],
                         'fresh_kl': result['fresh']['mean_kl_per_token'],
                         'fresh_mean_tokens': result['fresh']['mean_response_tokens'],
                         'fresh_capped': result['fresh']['capped']}
    rates = {}
    for rate in (3e-4, 1e-3):
        runs = [item for item in metrics.values() if item['replay'] and item['replay']['lr'] == rate]
        if {item['replay']['set_index'] for item in runs} != {0, 1, 2} or len(runs) != 3:
            raise ValueError(f'Expected one evaluation per set for LR {rate}')
        rates[f'{rate:.0e}'] = {key: mean(item[key] for item in runs) for key in
                               ('heldout_kl', 'dev_accuracy', 'fresh_kl', 'fresh_mean_tokens', 'fresh_capped')}
    if len(metrics) != 7 or metrics['base']['replay'] is not None:
        raise ValueError('Expected six adapter cases and the initial student')
    path.mkdir()
    save(path / 'complete.json', {'cases': metrics, 'means_by_lr': rates, 'baseline': metrics['base']})


def budget_eval(queue, stage, path):
    args = queue.a
    argv = [queue.python, '-m', 'speed_limit.k3_gate', 'budget', '--inputs', str(args.inputs),
            '--cache', str(args.cache), '--output', str(path), '--dev-max-tokens', str(stage['cap'])]
    if stage.get('checkpoint'):
        argv += ['--checkpoint', stage['checkpoint'], '--reference', stage['reference'],
                 '--checkpoint-step', str(stage['checkpoint_step'])]
    queue.check_gpu()
    queue.invoke(stage['name'], command=argv)


def budget_summary(queue, stage, path):
    from speed_limit.queue import save

    root = queue.a.run.parent
    schedule = json.loads(queue.a.schedule.read_text())
    cases = {}
    for role, jobs in schedule.items():
        directory = root / f'gpu-{role}'
        for item in jobs:
            if item['task'] == 'budget-dev':
                name = item['name']
                if name in cases:
                    raise ValueError(f'Duplicate budget case: {name}')
                cases[name] = (item, directory / name / 'complete.json', directory / f'role-{role}.json')
    if not cases:
        raise ValueError('No budget evaluation cases were scheduled')
    pending = dict(cases)
    checked, errors = {}, {}
    path.mkdir()
    while pending:
        for name, (item, source, state_path) in list(pending.items()):
            if source.is_file():
                result = json.loads(source.read_text())
                replay = result['replay']
                if (result['cap'] != item['cap'] or result['samples'] != 480
                        or (replay['checkpoint']['path'] if replay else None) != item.get('checkpoint')):
                    raise ValueError(f'Budget case differs from schedule: {source}')
                checked[name] = {**result, 'receipt': str(source)}
                del pending[name]
                continue
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                failure = (state.get('failures') or {}).get(name)
                if failure or state['phase'] in ('complete', 'failed'):
                    if source.is_file():
                        continue
                    errors[name] = failure or state['phase']
                    del pending[name]
        save(path / 'progress.json', {'completed': len(checked), 'expected': len(cases),
                                      'pending': sorted(pending), 'failures': errors})
        if pending:
            queue.save(phase=stage['name'], pending_budget_cases=sorted(pending))
            time.sleep(15)
    groups = {}
    for result in checked.values():
        replay = result['replay']
        family = 'base' if replay is None else f"{replay['lr']:.0e}@{replay['checkpoint']['step']}"
        groups.setdefault((result['cap'], family), []).append(result)
    comparison = []
    for (cap, family), members in sorted(groups.items()):
        if family != 'base' and {item['replay']['set_index'] for item in members} != {0, 1, 2}:
            continue
        if len(members) != (1 if family == 'base' else 3):
            raise ValueError(f'Duplicate budget cohort: {cap}, {family}')
        comparison.append({'cap': cap, 'family': family, 'runs': len(members),
                           'dev_accuracy': mean(item['dev']['mean_accuracy'] for item in members),
                           'mean_response_tokens': mean(item['mean_response_tokens'] for item in members),
                           'capped_share': mean(item['capped'] / item['samples'] for item in members)})
    save(path / 'complete.json', {'status': 'partial' if errors else 'complete', 'expected': len(cases),
                                  'cases': checked, 'failures': errors, 'comparison': comparison})


def residual_stage(queue, stage, path):
    args = queue.a
    task = stage['task']
    step = stage.get('step', 0)
    options = ['--inputs', str(args.inputs), '--cache', str(args.cache), '--output', str(path),
               '--set-index', str(stage['set_index']), '--task', task, '--step', str(step),
               '--lr', str(stage.get('lr', 1e-4)), '--revision', args.revision]
    if stage.get('save_checkpoints'):
        options.append('--save-checkpoints')
    if stage.get('save_final_adapter'):
        options.append('--save-final-adapter')
    if stage.get('checkpoint_steps'):
        options.extend(['--checkpoint-steps', *map(str, stage['checkpoint_steps'])])
    if stage.get('forecast_from'):
        options += ['--forecast-from', stage['forecast_from']]
    if step:
        checkpoint = Path(stage['checkpoint'])
        if not (checkpoint / 'ready.json').is_file():
            raise RuntimeError(f'Expected earlier stage to publish {checkpoint}')
        options += ['--checkpoint', str(checkpoint)]
    queue.check_gpu()
    queue.invoke(stage['name'], command=[queue.python, '-m', 'speed_limit.residual_k3', *options])


def run(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import Queue, save

    class WorkerQueue(Queue):
        def check_gpu(self):
            self.save(gpu_preflight=gpu_preflight(self.a))

    if args.after_role:
        original = predecessor(args)
        print('WAITING_FOR_GPU', args.after_role, flush=True)
        while True:
            prior = predecessor(args)
            if (prior['pid'], prior['started_at']) != (original['pid'], original['started_at']):
                raise RuntimeError('Predecessor worker identity changed while waiting')
            if prior['phase'] in ('complete', 'failed') and not prior.get('child_pid'):
                break
            save(args.run / 'waiting.json', {'predecessor': str(args.after_role), 'phase': prior['phase'],
                                            'heartbeat_at': time.time()})
            time.sleep(10)
    queue = WorkerQueue(args)
    queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                     NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
    stages = json.loads(args.schedule.read_text())[str(args.role)]
    results, failures = {}, {}
    for stage in stages:
        name = stage['name']
        path = args.run / name
        try:
            if stage['task'] == 'gate-eval':
                gate_eval(queue, stage, path)
            elif stage['task'] == 'gate-summary':
                gate_summary(queue, stage, path)
            elif stage['task'] == 'budget-dev':
                budget_eval(queue, stage, path)
            elif stage['task'] == 'budget-summary':
                budget_summary(queue, stage, path)
            else:
                residual_stage(queue, stage, path)
            receipt = path / 'complete.json'
            json.loads(receipt.read_text())
            results[name] = str(receipt)
        except Exception:
            failures[name] = traceback.format_exc()
            save(args.run / f'failed-{name}.json', {'error': failures[name]})
        queue.save(results=results, failures=failures)
    queue.save(phase='failed' if failures else 'complete', finished_at=time.time())
    if failures:
        raise RuntimeError(f'K3 stages failed: {", ".join(failures)}')


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'inputs', 'cache', 'run', 'schedule'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('host', 'gpu', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', type=int, choices=range(8), required=True)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--seconds', type=int)
    duration.add_argument('--deadline-epoch', type=int)
    parser.add_argument('--mps-server', type=int)
    parser.add_argument('--mps-pipe', type=Path)
    parser.add_argument('--after-role', type=Path, help='Wait for an earlier worker on this physical GPU')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    now = time.time()
    deadline = args.deadline_epoch if args.deadline_epoch is not None else now + (43200 if args.seconds is None else args.seconds)
    if not 5 < deadline - now <= 108000:
        parser.error('Worker must allow the termination grace and fit within 30 hours')
    if bool(args.mps_server) != bool(args.mps_pipe):
        parser.error('Supply both the verified MPS server PID and its control pipe')
    if args.mps_server is not None and args.mps_server <= 0:
        parser.error('MPS server PID must be positive')
    if not args.detach:
        run(args)
        return

    if args.after_role:
        predecessor(args)
    else:
        gpu_preflight(args)
    args.run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seconds = deadline - started
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(seconds - 5),
            sys.executable, str(Path(__file__).resolve()),
            *(arg for arg in sys.argv[1:] if arg != '--detach')]
    with (args.run / 'controller.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'watchdog_pid': child.pid, 'host': args.host, 'gpu': args.gpu, 'role': args.role,
               'revision': args.revision, 'started_at': started,
               'deadline_epoch': deadline, 'duration_seconds': seconds,
               'after_role': str(args.after_role) if args.after_role else None}
    (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
