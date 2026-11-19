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


def run_schedule(queue, args, schedule):
    from speed_limit.queue import save

    results, failures = {}, {}
    for entry in schedule:
        set_index = entry['set_index']
        task = entry['task']
        name = 'lora'
        extra = []
        if task == 'lora':
            extra = ['--reference', entry['reference'], '--reference-lr', '1e-6']
        else:
            checkpoint = Path(entry['checkpoint'])
            step, mode = entry['step'], entry['preconditioner']
            name = f'spectrum-S{set_index}-step-{step:04d}-{mode}'
            extra = ['--checkpoint', str(checkpoint), '--checkpoint-step', str(step), '--preconditioner', mode]
        output = args.run / name
        try:
            if task == 'spectrum' and not (checkpoint / 'ready.json').is_file():
                started = time.monotonic()
                while not (checkpoint / 'ready.json').is_file():
                    if (checkpoint.parents[1] / 'failed-lora.json').exists():
                        raise RuntimeError(f'Producer failed before publishing {checkpoint}')
                    if time.monotonic() - started >= 5400:
                        raise TimeoutError(f'Checkpoint was not published within 90 minutes: {checkpoint}')
                    queue.save(phase='waiting_for_checkpoint', next_stage=name, checkpoint=str(checkpoint))
                    time.sleep(10)
            queue.check_gpu()
            queue.invoke(name, command=[queue.python, '-m', 'speed_limit.k3',
                '--inputs', str(args.inputs), '--cache', str(args.cache), '--output', str(output),
                '--set-index', str(set_index), '--task', task, '--revision', args.revision,
                '--lr', str(entry['lr']), *extra])
            receipt = output / 'complete.json'
            json.loads(receipt.read_text())
            results[name] = str(receipt)
        except Exception:
            failures[name] = traceback.format_exc()
            save(args.run / f'failed-{name}.json', {'error': failures[name]})
        queue.save(results=results, failures=failures)
    if failures:
        raise RuntimeError(f'K3 stages failed: {", ".join(failures)}')
    queue.save(phase='complete', finished_at=time.time())


def run(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source / 'src'))
    from speed_limit.queue import Queue, save

    queue = None
    try:
        queue = Queue(args)
        queue.env.update(PYTHONDONTWRITEBYTECODE='1',
                         NRL_CACHE_ROOT=str(args.run / 'compile-cache' / 'runtime'))
        if args.schedule:
            schedule = json.loads(args.schedule.read_text())
            run_schedule(queue, args, schedule[str(args.role)])
            return
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
    for name in ('source', 'inputs', 'cache', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--schedule', type=Path, help='Per-GPU dependencies and ordered K3 stages')
    for name in ('host', 'gpu', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', type=int, choices=range(4), required=True)
    parser.add_argument('--lora-lr', type=float, help='Run S0 LoRA at this rate against the 1e-6 reference')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.schedule and (args.lora_lr is not None or args.reference is not None):
        parser.error('Use a schedule or individual launch options')
    if args.schedule is None and args.reference is None:
        parser.error('An individual launch requires --reference')
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
                   'lora_lr': args.lora_lr, 'schedule': str(args.schedule) if args.schedule else None}
        (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
