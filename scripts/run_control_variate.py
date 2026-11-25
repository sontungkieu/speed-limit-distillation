"""Run one matched K2 objective followed by dev evaluation."""
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def run(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import Queue, save

    queue = None
    try:
        queue = Queue(args)
        queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                         NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
        common = ['--inputs', str(args.inputs), '--cache', str(args.cache)]
        training = args.run / 'training'
        queue.save(objective=args.objective, set_index=args.set_index, lr=args.lr)
        queue.invoke('training', command=[
            queue.python, '-m', 'speed_limit.fixed', *common, '--output', str(training),
            '--set-index', str(args.set_index), '--lr', str(args.lr), '--steps', '300',
            '--objective', args.objective, '--revision', args.revision,
        ])
        result = json.loads((training / 'complete.json').read_text())
        step = result['steps']
        checkpoint = training / f'checkpoint-{step:04d}'
        if (result['set_index'] != args.set_index or result['objective'] != args.objective
                or result['lr'] != args.lr or result['optimizer'] != 'adamw'
                or result['checkpoint'] != str(checkpoint)
                or json.loads((checkpoint / 'ready.json').read_text())['step'] != step):
            raise RuntimeError('Training completion does not match the requested K2 arm')
        queue.check_gpu()
        queue.invoke('dev', command=[
            queue.python, '-m', 'speed_limit.phase0', '--task', 'evaluate', *common,
            '--output', str(args.run / 'dev'), '--checkpoint', str(checkpoint),
            '--checkpoint-step', str(step), '--dev',
        ])
        summary = json.loads((args.run / 'dev' / f'eval-step-{step:04d}.json').read_text())
        if ({name: values['total'] for name, values in summary['benchmarks'].items()}
                != {'aime24': 240, 'aime25': 240}):
            raise RuntimeError('Incomplete dev evaluation')
        queue.save(phase='complete', finished_at=time.time(), training=result, summary=summary)
    except Exception:
        save(args.run / f'failed-role-{args.role}.json', {'error': traceback.format_exc()})
        if queue is not None:
            queue.save(phase='failed', finished_at=time.time())
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'inputs', 'cache', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('host', 'gpu', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', type=int, choices=range(8), required=True)
    parser.add_argument('--objective', choices=('top64', 'control-full', 'control-top20'))
    parser.add_argument('--set-index', type=int, choices=range(3))
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('A positive finite learning rate is required')
    if (args.objective is None) != (args.set_index is None) or (args.role >= 4 and args.objective is None):
        parser.error('Specify both objective and set index for an explicit arm')
    if args.objective is None:
        args.objective = 'control-top20' if args.role == 3 else 'control-full'
        args.set_index = args.role % 3
    now = time.time()
    if args.deadline_epoch is not None and not 0 < args.deadline_epoch - now <= 108000:
        parser.error('The watchdog deadline must be within the next 30 hours')
    if args.detach:
        args.run.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve()),
                *(value for value in sys.argv[1:] if value != '--detach')]
        if args.deadline_epoch is not None:
            remaining = max(.001, args.deadline_epoch - time.time())
            argv = ['timeout', '--signal=TERM', '--kill-after=5s', f'{remaining:.3f}s', *argv]
        with (args.run / 'controller.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(pid=child.pid, host=args.host, gpu=args.gpu, role=args.role,
                       run=str(args.run), revision=args.revision, started_at=time.time(),
                       lr=args.lr, objective=args.objective, set_index=args.set_index,
                       deadline_epoch=args.deadline_epoch)
        (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
