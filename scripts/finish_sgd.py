"""Finish the two incomplete SGD controls from the fixed-set campaign."""
import argparse
import json
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path


def run(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import Queue, save

    queue = None
    try:
        queue = Queue(args)
        queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                         NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
        cache = args.phase0 / 'teacher-cache'
        worker = args.previous / f'worker-{21 if args.role == 0 else 20}' / 'jobs'
        set_index = 1 if args.role == 0 else 0
        previous = worker / f'sgd-S{set_index}-1e-06' / 'output'
        common = ['--inputs', str(args.inputs), '--cache', str(cache)]
        if args.role == 0:
            checkpoints = sorted(previous.glob('checkpoint-*/ready.json'))
            if not checkpoints:
                raise RuntimeError('No complete SGD checkpoint is available')
            training = args.run / 'training'
            queue.invoke('training', command=[
                queue.python, '-m', 'speed_limit.fixed', *common, '--output', str(training),
                '--set-index', '1', '--lr', '1e-6', '--steps', '300', '--optimizer', 'sgd',
                '--resume', str(checkpoints[-1].parent), '--revision', args.revision,
            ])
        else:
            training = previous
        result = json.loads((training / 'complete.json').read_text())
        step = result['steps']
        checkpoint = training / f'checkpoint-{step:04d}'
        if (result['set_index'] != set_index or result['optimizer'] != 'sgd'
                or result['lr'] != 1e-6 or not (checkpoint / 'ready.json').is_file()):
            raise RuntimeError('Invalid SGD checkpoint for final evaluation')
        evaluation = [
            queue.python, '-m', 'speed_limit.phase0', '--task', 'evaluate', *common,
            '--output', str(args.run / 'dev'), '--checkpoint', str(checkpoint),
            '--checkpoint-step', str(step), '--dev',
        ]
        prior_output = worker / 'dev-sgd-S0-1e-06' / 'output'
        saved_samples = prior_output / f'eval-step-{step:04d}.jsonl'
        if args.role == 1 and saved_samples.is_file():
            prior = json.loads((prior_output / 'started.json').read_text())
            if (prior['checkpoint'] != str(checkpoint) or prior['checkpoint_step'] != step
                    or prior['inputs'] != str(args.inputs) or prior['seed'] != 0
                    or not prior['dev'] or prior['canary'] or prior['task'] != 'evaluate'):
                raise RuntimeError('Previous evaluation settings differ')
            evaluation.extend(['--resume-eval', str(saved_samples)])
        queue.check_gpu()
        queue.invoke('dev', command=evaluation)
        summary = json.loads((args.run / 'dev' / f'eval-step-{step:04d}.json').read_text())
        if ({name: values['total'] for name, values in summary['benchmarks'].items()}
                != {'aime24': 240, 'aime25': 240}):
            raise RuntimeError('Incomplete dev evaluation')
        queue.save(phase='complete', finished_at=time.time(), summary=summary)
    except Exception:
        save(args.run / f'failed-role-{args.role}.json', {'error': traceback.format_exc()})
        if queue is not None:
            queue.save(phase='failed', finished_at=time.time())
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'run', 'inputs', 'phase0', 'previous'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('host', 'gpu', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', type=int, choices=(0, 1), required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    if args.detach:
        args.run.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve()),
                *(value for value in sys.argv[1:] if value != '--detach')]
        with (args.run / 'controller.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(pid=child.pid, host=args.host, gpu=args.gpu, role=args.role,
                       run=str(args.run), revision=args.revision, started_at=time.time())
        (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
