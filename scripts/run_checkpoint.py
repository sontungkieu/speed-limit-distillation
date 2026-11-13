"""Run one checkpoint evaluation detached using the qualified runtime environment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def retire_watch(run, receipt):
    pid = int((run / 'watch-step100.pid').read_text().strip())
    proc = Path('/proc') / str(pid)
    expected = str(run.parent / 'watch-and-launch-step100.sh')
    if proc.exists():
        command = (proc / 'cmdline').read_bytes().split(b'\0')
        if expected.encode() not in command:
            raise RuntimeError('Watch PID identity differs from the authorized script')
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not proc.exists() or (proc / 'stat').read_text().split(') ', 1)[1].startswith('Z'):
                break
            time.sleep(.1)
        else:
            if expected.encode() not in (proc / 'cmdline').read_bytes().split(b'\0'):
                raise RuntimeError('Watch identity changed during retirement')
            os.kill(pid, signal.SIGKILL)
    save(receipt, {'pid': pid, 'script': expected, 'retired_at': time.time(),
                   'hostname': os.uname().nodename})
    print(receipt.read_text(), flush=True)


def evaluate(args):
    sys.path.insert(0, str(args.runner_source / 'src'))
    from speed_limit.queue import environment, resolve_gpu_index
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                         '--format=csv,noheader'], text=True)
    index = resolve_gpu_index(args.gpu, inventory)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    usage = [line.split(',')[1] for line in rows.splitlines()
             if line.split(',')[0].strip() == args.gpu]
    if len(usage) != 1 or int(usage[0]) > 1024:
        raise RuntimeError('Requested GPU is occupied')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == args.gpu for line in processes.splitlines()):
        raise RuntimeError('Requested GPU has a compute process')
    checkpoint = args.training_run / 'semi' / f'checkpoint-{args.step:04d}'
    ready = json.loads((checkpoint / 'ready.json').read_text())
    if ready['step'] != args.step or not any(checkpoint.glob('*.safetensors')):
        raise RuntimeError('Checkpoint is incomplete')
    runner_receipt = json.loads((args.runner_source / 'release-receipt.json').read_text())
    env_args = argparse.Namespace(source=args.runner_source, inputs=args.inputs,
        run=args.output, role=0, gpu=args.gpu, revision=runner_receipt['revision'])
    env = environment(env_args)
    result = args.output / 'result'
    argv = [str(args.inputs / 'runtime/envs/vllm/bin/python'), '-m', 'speed_limit.phase0',
            '--task', 'evaluate', '--inputs', str(args.inputs), '--output', str(result),
            '--cache', str(args.training_run / 'teacher-cache'), '--checkpoint', str(checkpoint),
            '--checkpoint-step', str(args.step)]
    if args.dev:
        argv.append('--dev')
    state = {'pid': os.getpid(), 'hostname': os.uname().nodename, 'gpu': args.gpu,
             'gpu_index': index, 'step': args.step, 'dev': args.dev, 'started_at': time.time(),
             'runner_revision': runner_receipt['revision'], 'launcher_revision': args.revision,
             'argv': argv, 'phase': 'starting'}
    state_file = args.output / 'state.json'
    save(state_file, state)
    try:
        with (args.output / 'worker.log').open('xb') as stream:
            child = subprocess.Popen(argv, env=env, cwd=args.runner_source,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, phase='running')
            while child.poll() is None:
                state['heartbeat_at'] = time.time()
                save(state_file, state)
                time.sleep(10)
        state.update(exit_code=child.returncode, finished_at=time.time())
        if child.returncode:
            raise RuntimeError(f'Evaluator failed with exit {child.returncode}')
        completion = json.loads((result / 'complete.json').read_text())
        if completion['step'] != args.step:
            raise RuntimeError('Completion checkpoint mismatch')
        state.update(phase='complete', summary=json.loads(
            (result / f'eval-step-{args.step:04d}.json').read_text()))
        save(state_file, state)
    except Exception as error:
        state.update(phase='failed', error=str(error), finished_at=time.time())
        save(state_file, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retire-watch-run', type=Path)
    parser.add_argument('--retirement-receipt', type=Path)
    for name in ('inputs', 'runner-source', 'training-run', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--step', type=int, choices=(60, 80, 100))
    parser.add_argument('--gpu')
    parser.add_argument('--revision')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.retire_watch_run:
        if not args.retirement_receipt or args.retirement_receipt.exists():
            raise ValueError('A fresh retirement receipt is required')
        retire_watch(args.retire_watch_run, args.retirement_receipt)
        return
    if any(getattr(args, name) is None for name in
           ('inputs', 'runner_source', 'training_run', 'output', 'step', 'gpu', 'revision')):
        parser.error('Evaluation requires input, source, run, output, step, GPU and revision')
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
        argv.remove('--detach')
        with (args.output / 'controller.log').open('xb') as stream:
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        receipt = {'controller_pid': child.pid, 'output': str(args.output),
                   'launcher_revision': args.revision, 'time': time.time()}
        save(args.output / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
        return
    evaluate(args)


if __name__ == '__main__':
    main()
