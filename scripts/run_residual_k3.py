"""Run the ordered K3 residual/SGD stages on one verified GPU for up to 12 hours."""
import argparse
import json
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

    queue = Queue(args)
    queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                     NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
    stages = json.loads(args.schedule.read_text())[str(args.role)]
    results, failures = {}, {}
    for stage in stages:
        name, task = stage['name'], stage['task']
        path = args.run / name
        step = stage.get('step', 0)
        common = ['--inputs', str(args.inputs), '--cache', str(args.cache), '--output', str(path),
                  '--set-index', str(stage['set_index']), '--task', task, '--step', str(step),
                  '--lr', str(stage.get('lr', 1e-4)), '--revision', args.revision]
        try:
            if stage.get('save_checkpoints'):
                common.append('--save-checkpoints')
            if step:
                checkpoint = Path(stage['checkpoint'])
                if not (checkpoint / 'ready.json').is_file():
                    raise RuntimeError(f'Expected earlier stage to publish {checkpoint}')
                common += ['--checkpoint', str(checkpoint)]
            queue.check_gpu()
            queue.invoke(name, command=[queue.python, '-m', 'speed_limit.residual_k3', *common])
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
    parser.add_argument('--role', type=int, choices=range(4), required=True)
    parser.add_argument('--seconds', type=int, default=43200)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    if args.seconds <= 5 or args.seconds > 43200:
        parser.error('Worker duration must allow the five-second termination grace')
    if not args.detach:
        run(args)
        return

    args.run.mkdir(parents=True, exist_ok=False)
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(args.seconds - 5),
            sys.executable, str(Path(__file__).resolve()),
            *(arg for arg in sys.argv[1:] if arg != '--detach')]
    started = time.time()
    with (args.run / 'controller.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'watchdog_pid': child.pid, 'host': args.host, 'gpu': args.gpu, 'role': args.role,
               'revision': args.revision, 'started_at': started,
               'deadline_epoch': started + args.seconds, 'duration_seconds': args.seconds}
    (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
