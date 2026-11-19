"""Launch one independent K3 diagnostic on an assigned physical GPU."""
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
    sweep = args.lora_lr is not None
    task = 'lora' if sweep or args.role == 3 else 'spectrum'
    set_index = 0 if sweep else args.role % 3
    output = args.run / task
    argv = ['-m', 'speed_limit.k3', '--inputs', str(args.inputs), '--cache', str(args.cache),
            '--output', str(output), '--set-index', str(set_index), '--task', task,
            '--revision', args.revision, '--lr', str(args.lora_lr if sweep else 1e-6)]
    if task == 'lora':
        argv += ['--reference', str(args.reference)]
    if sweep:
        argv += ['--reference-lr', '1e-6']
    try:
        queue = Queue(args)
        queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                         NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
        queue.invoke(task, command=[queue.python, *argv])
        result = json.loads((output / 'complete.json').read_text())
        queue.save(phase='complete', result=str(output / 'complete.json'),
                   elapsed_seconds=result['elapsed_seconds'], finished_at=time.time())
    except Exception:
        save(args.run / f'failed-role-{args.role}.json', {'error': traceback.format_exc()})
        if queue is not None:
            queue.save(phase='failed', finished_at=time.time())
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'inputs', 'cache', 'run', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('host', 'gpu', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', type=int, choices=range(4), required=True)
    parser.add_argument('--lora-lr', type=float, help='Run S0 LoRA at this rate against the 1e-6 reference')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.lora_lr is not None and (not math.isfinite(args.lora_lr) or args.lora_lr <= 0):
        parser.error('--lora-lr must be finite and positive')
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    if args.detach:
        args.run.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve()),
                *(value for value in sys.argv[1:] if value != '--detach')]
        with (args.run / 'controller.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': child.pid, 'host': args.host, 'gpu': args.gpu, 'role': args.role,
                   'run': str(args.run), 'revision': args.revision, 'started_at': time.time(),
                   'lora_lr': args.lora_lr}
        (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
