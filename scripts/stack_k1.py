"""Three fixed-state LR sweeps and a serial GPU-0 evaluation queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

LEARNING_RATES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def wait_for_step100(path, state, state_path):
    while True:
        if path.exists():
            value = json.loads(path.read_text())
            if value.get('phase') == 'complete':
                return
            if value.get('phase') == 'failed':
                raise RuntimeError('Step100 evaluator failed; inspect its worker log')
        state.update(phase='waiting_for_phase0_step100', heartbeat_at=time.time())
        save(state_path, state)
        time.sleep(15)


def invoke(args, env, state, state_path, name, argv, completion):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == args.gpu for line in raw.splitlines()):
        raise RuntimeError('Reserved GPU has a compute process before next job')
    output_root = args.run / 'dev' if args.role == 0 else args.run
    output_root.mkdir(parents=True, exist_ok=True)
    log = output_root / f'{name}.log'
    state.update(phase='running', current_job=name, log=str(log), argv=argv)
    with log.open('xb') as stream:
        child = subprocess.Popen(argv, env=env, cwd=args.source,
                                 stdout=stream, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        while child.poll() is None:
            state['heartbeat_at'] = time.time()
            save(state_path, state)
            time.sleep(10)
    result = {'name': name, 'exit_code': child.returncode, 'time': time.time(),
              'complete': child.returncode == 0 and completion.exists(),
              'completion_path': str(completion)}
    state['child_pid'] = None
    state.setdefault('jobs', []).append(result)
    save(state_path, state)
    return result['complete']


def worker(args):
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import environment
    env = environment(args)
    python = str(args.inputs / 'runtime/envs/vllm/bin/python')
    state_path = args.run / f'role-{args.role}.json'
    if state_path.exists():
        raise RuntimeError('Role already has state; inspect before restarting')
    state = {'pid': os.getpid(), 'hostname': os.uname().nodename,
             'role': args.role, 'gpu': args.gpu, 'source_revision': args.revision,
             'started_at': time.time(), 'phase': 'starting', 'jobs': [],
             'phase0_gate': 'pending; concurrent K1 authorized by user'}
    save(state_path, state)
    try:
        if args.role:
            set_index = args.role - 1
            for lr in LEARNING_RATES:
                job = f'S{set_index}-lr-{lr:g}'
                output = args.run / f'S{set_index}' / f'lr-{lr:g}'
                argv = [python, '-m', 'speed_limit.fixed', '--inputs', str(args.inputs),
                        '--output', str(output), '--cache', str(args.training_run / 'teacher-cache'),
                        '--set-index', str(set_index), '--lr', str(lr), '--steps', '300',
                        '--revision', args.revision]
                if args.previous_run:
                    previous = args.previous_run / f'S{set_index}' / f'lr-{lr:g}'
                    completion = previous / 'complete.json'
                    if completion.exists() and (previous / 'dev-request.json').exists():
                        state['jobs'].append({'name': job, 'complete': True,
                                              'completion_path': str(completion)})
                        save(state_path, state)
                        continue
                    checkpoints = sorted(previous.glob('checkpoint-*/ready.json'))
                    if checkpoints:
                        argv.extend(['--resume', str(checkpoints[-1].parent)])
                invoke(args, env, state, state_path, job, argv, output / 'complete.json')
            state.update(phase='complete', finished_at=time.time())
            save(state_path, state)
            return
        wait_for_step100(args.step100_attempt / 'state.json', state, state_path)
        baseline_output = args.run / 'dev' / 'phase0-step100'
        baseline_args = [python, '-m', 'speed_limit.phase0', '--task', 'evaluate',
            '--inputs', str(args.inputs), '--output', str(baseline_output),
            '--cache', str(args.training_run / 'teacher-cache'), '--checkpoint',
            str(args.training_run / 'semi/checkpoint-0100'), '--checkpoint-step', '100', '--dev']
        invoke(args, env, state, state_path, 'dev-phase0-step100', baseline_args,
               baseline_output / 'complete.json')
        handled = set()
        def pending_requests():
            requests = {}
            if args.previous_run:
                for role in args.training_roles:
                    for path in args.previous_run.glob(f'S{role - 1}/lr-*/dev-request.json'):
                        requests[(path.parent.parent.name, path.parent.name)] = path
            for path in args.run.glob('S*/lr-*/dev-request.json'):
                requests[(path.parent.parent.name, path.parent.name)] = path
            return sorted(requests.values())

        while True:
            requests = pending_requests()
            for path in requests:
                if str(path) in handled:
                    continue
                request = json.loads(path.read_text())
                name = f'dev-{path.parent.parent.name}-{path.parent.name}'
                output = args.run / 'dev' / name
                argv = [python, '-m', 'speed_limit.phase0', '--task', 'evaluate',
                    '--inputs', str(args.inputs), '--output', str(output),
                    '--cache', str(args.training_run / 'teacher-cache'),
                    '--checkpoint', request['checkpoint'], '--checkpoint-step',
                    str(request['step']), '--dev']
                invoke(args, env, state, state_path, name, argv, output / 'complete.json')
                handled.add(str(path))
            training_states = [args.run / f'role-{role}.json' for role in args.training_roles]
            all_finished = all(path.exists() and json.loads(path.read_text()).get('phase')
                               in ('complete', 'failed') for path in training_states)
            if all_finished and all(str(path) in handled for path in pending_requests()):
                break
            state.update(phase='waiting_for_training_checkpoints', heartbeat_at=time.time())
            save(state_path, state)
            time.sleep(15)
        state.update(phase='complete', finished_at=time.time())
        save(state_path, state)
    except Exception:
        state.update(phase='failed', error=traceback.format_exc(), finished_at=time.time())
        save(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'inputs', 'training-run', 'run', 'step100-attempt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--role', type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--previous-run', type=Path)
    parser.add_argument('--training-roles', type=int, choices=(1, 2, 3),
                        nargs='+', default=(1, 2, 3))
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.detach:
        claim = args.run / f'role-{args.role}-launch.json'
        with claim.open('x') as receipt:
            argv = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
            argv.remove('--detach')
            with (args.run / f'role-{args.role}-controller.log').open('xb') as stream:
                child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            value = {'controller_pid': child.pid, 'role': args.role, 'gpu': args.gpu,
                     'source_revision': args.revision, 'time': time.time()}
            receipt.write(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)
        return
    worker(args)


if __name__ == '__main__':
    main()
