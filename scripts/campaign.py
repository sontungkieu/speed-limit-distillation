"""Six-hour, single-GPU work stealing campaign across explicitly allocated nodes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

RATES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def save(path, value):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def add_job(root, name, kind, **values):
    path = root / 'jobs' / name
    try:
        path.mkdir()
    except FileExistsError:
        return
    save(path / 'spec.json', dict(name=name, kind=kind, **values))


def prepare(args):
    (args.run / 'jobs').mkdir(parents=True, exist_ok=False)
    save(args.run / 'contract.json', dict(deadline=args.deadline, revision=args.revision,
         nodes=['mtp-30b-generation-0', 'mtp-30b-generation-2', 'anhhbt-qad1'],
         gpu_count=20, phase0_gate='pending aggregate audit',
         top64_scope='exploratory S0 LR sweep concurrent with K1'))
    for index in range(3):
        for lr in RATES:
            add_job(args.run, f'k1-S{index}-{lr:g}', 'train', set_index=index,
                    lr=lr, optimizer='adamw', objective='sampled')
    for lr in RATES:
        add_job(args.run, f'k2pilot-S0-{lr:g}', 'train', set_index=0,
                lr=lr, optimizer='adamw', objective='top64')


def result(job):
    path = job / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def claim(args):
    own = args.run / f'worker-{args.worker}' / 'jobs'
    own.mkdir(parents=True, exist_ok=True)
    initial_index = args.worker if args.initial_index is None else args.initial_index
    for index in [initial_index, *args.followup_index]:
        initial = sorted((args.run / 'jobs').iterdir())[index]
        spec = json.loads((initial / 'spec.json').read_text())
        add_job(own.parent, spec.pop('name'), **spec)
    for job in sorted(own.iterdir()):
        done = result(job)
        spec = json.loads((job / 'spec.json').read_text())
        if done and done['complete'] and spec['kind'] == 'train':
            completion = json.loads((job / 'output/complete.json').read_text())
            add_job(own.parent, 'dev-' + spec['name'], 'dev',
                    checkpoint=completion['checkpoint'], step=completion['steps'])
    # Every worker writes only its private subtree. Shared artifacts are read-only.
    scores = []
    for lr in RATES:
        values = []
        for index in range(3):
            paths = list(args.run.glob(f'worker-*/jobs/dev-k1-S{index}-{lr:g}'))
            if not paths or not result(paths[0]) or not result(paths[0])['complete']:
                break
            job = paths[0]
            dev = json.loads((job / 'spec.json').read_text())
            values.append(json.loads((job / 'output' /
                          f"eval-step-{dev['step']:04d}.json").read_text())['mean_accuracy'])
        if len(values) != 3:
            break
        scores.append((sum(values) / 3, lr))
    if len(scores) == 5:
        best = sorted(scores, key=lambda row: (-row[0], row[1]))[:2]
        selection = args.run / f'worker-{args.worker}' / 'selection.json'
        save(selection, dict(metric='mean dev accuracy across three sets', scores=scores,
             best_two=best, interpretation='provisional; inspect heldout KL and fresh length'))
        if initial_index < 6:
            index = initial_index % 3
            lr = best[initial_index // 3][1]
            add_job(own.parent, f'sgd-S{index}-{lr:g}', 'train', set_index=index,
                    lr=lr, optimizer='sgd', objective='sampled')
        elif initial_index in (6, 7):
            index = initial_index - 5
            lr = best[0][1]
            add_job(own.parent, f'k2pilot-S{index}-{lr:g}', 'train', set_index=index,
                    lr=lr, optimizer='adamw', objective='top64')
    for job in sorted(own.iterdir()):
        if result(job) or (job / 'claim.json').exists():
            continue
        with (job / 'claim.json').open('x') as stream:
            json.dump(dict(host=os.uname().nodename, worker=args.worker,
                           gpu=args.gpu, time=time.time()), stream)
        return job
    return None


def gpu_free(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if any(row.split(',')[0].strip() == gpu for row in rows.splitlines()):
        raise RuntimeError('Assigned GPU contains a compute process')


def argv_for(args, job):
    spec = json.loads((job / 'spec.json').read_text())
    output = job / 'output'
    python = str(args.inputs / 'runtime/envs/vllm/bin/python')
    common = ['--inputs', str(args.inputs), '--output', str(output),
              '--cache', str(args.phase0 / 'teacher-cache')]
    if spec['kind'] == 'dev':
        return [python, '-m', 'speed_limit.phase0', '--task', 'evaluate', *common,
                '--checkpoint', spec['checkpoint'], '--checkpoint-step', str(spec['step']), '--dev']
    argv = [python, '-m', 'speed_limit.fixed', *common, '--set-index', str(spec['set_index']),
            '--lr', str(spec['lr']), '--steps', '300', '--optimizer', spec['optimizer'],
            '--objective', spec['objective'], '--revision', args.revision]
    if spec['objective'] == 'sampled' and spec['optimizer'] == 'adamw':
        previous = args.previous / f"S{spec['set_index']}" / f"lr-{spec['lr']:g}"
        checkpoints = sorted(previous.glob('checkpoint-*/ready.json'))
        if checkpoints:
            argv.extend(['--resume', str(checkpoints[-1].parent)])
    return argv


def stop_group(child):
    # The child owns a new session, including all vLLM descendants.
    try:
        os.killpg(child.pid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def worker(args):
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import environment
    args.role = args.worker
    shared_run = args.run
    args.run = shared_run / f'worker-{args.worker}'
    args.run.mkdir(parents=True, exist_ok=True)
    env = environment(args)
    args.run = shared_run
    state_path = args.run / f'worker-{args.worker}' / 'state.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(pid=os.getpid(), host=os.uname().nodename, gpu=args.gpu,
                 deadline=args.deadline, revision=args.revision)
    interrupted = False

    def stopping(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    while not interrupted and time.time() < args.deadline - 10:
        gpu_free(args.gpu)
        job = claim(args)
        if job is None:
            state.update(phase='waiting', heartbeat=time.time())
            save(state_path, state)
            time.sleep(10)
            continue
        argv = argv_for(args, job)
        # GNU timeout remains alive with the payload if this controller disappears.
        argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s',
                str(max(1, int(args.deadline - time.time() - 5))), *argv]
        state.update(phase='running', job=job.name, heartbeat=time.time())
        save(state_path, state)
        with (job / 'payload.log').open('xb') as stream:
            child = subprocess.Popen(argv, cwd=args.source, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            try:
                while child.poll() is None and not interrupted and time.time() < args.deadline - 10:
                    state['heartbeat'] = time.time()
                    save(state_path, state)
                    time.sleep(3)
            finally:
                stop_group(child)
        complete = child.returncode == 0 and (job / 'output/complete.json').exists()
        save(job / 'result.json', dict(complete=complete, exit_code=child.returncode,
             finished_at=time.time(), interrupted=interrupted, deadline=args.deadline))
        state['child_pid'] = None
    state.update(phase='stopped', finished_at=time.time())
    save(state_path, state)


def main():
    parser = argparse.ArgumentParser()
    for key in ('source', 'run', 'inputs', 'phase0', 'previous'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--initial-index', type=int, choices=range(20))
    parser.add_argument('--followup-index', type=int, choices=range(20), nargs='*', default=[])
    parser.add_argument('--gpu')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
        return
    if args.worker is None or not args.gpu or args.deadline <= time.time():
        parser.error('Worker, GPU and future deadline required')
    if args.detach:
        with (args.run / f'worker-{args.worker}-launch.json').open('x') as stream:
            cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
            cmd.remove('--detach')
            with (args.run / f'worker-{args.worker}.log').open('xb') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            receipt = dict(pid=child.pid, host=os.uname().nodename, gpu=args.gpu,
                           deadline=args.deadline, worker=args.worker)
            json.dump(receipt, stream)
        print(json.dumps(receipt), flush=True)
    else:
        try:
            worker(args)
        except Exception:
            save(args.run / f'worker-{args.worker}-failed.json', dict(error=traceback.format_exc(),
                                                                   time=time.time()))
            raise


if __name__ == '__main__':
    main()
