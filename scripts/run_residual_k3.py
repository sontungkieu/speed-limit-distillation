"""Run the ordered K3 residual/SGD stages on one verified GPU for up to 12 hours."""
import argparse
import csv
import json
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
        name, task = stage['name'], stage['task']
        path = args.run / name
        step = stage.get('step', 0)
        common = ['--inputs', str(args.inputs), '--cache', str(args.cache), '--output', str(path),
                  '--set-index', str(stage['set_index']), '--task', task, '--step', str(step),
                  '--lr', str(stage.get('lr', 1e-4)), '--revision', args.revision]
        try:
            if stage.get('save_checkpoints'):
                common.append('--save-checkpoints')
            if stage.get('save_final_adapter'):
                common.append('--save-final-adapter')
            if stage.get('forecast_from'):
                common += ['--forecast-from', stage['forecast_from']]
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
    parser.add_argument('--mps-server', type=int)
    parser.add_argument('--mps-pipe', type=Path)
    parser.add_argument('--after-role', type=Path, help='Wait for an earlier worker on this physical GPU')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        parser.error('Launch host does not match the allocation')
    if args.seconds <= 5 or args.seconds > 43200:
        parser.error('Worker duration must allow the five-second termination grace')
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
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(args.seconds - 5),
            sys.executable, str(Path(__file__).resolve()),
            *(arg for arg in sys.argv[1:] if arg != '--detach')]
    started = time.time()
    with (args.run / 'controller.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'watchdog_pid': child.pid, 'host': args.host, 'gpu': args.gpu, 'role': args.role,
               'revision': args.revision, 'started_at': started,
               'deadline_epoch': started + args.seconds, 'duration_seconds': args.seconds,
               'after_role': str(args.after_role) if args.after_role else None}
    (args.run / 'launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
