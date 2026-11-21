"""Enforce a wall-clock cutoff for already detached K3 workers on one node."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import NamedTuple


PROC = Path('/proc')


class Process(NamedTuple):
    parent: int
    session: int
    started: int


def process(pid):
    try:
        state = (PROC / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if state[0] != 'Z':
        return Process(int(state[1]), int(state[3]), int(state[19]))
    return None


def verified_leader(path, hostname):
    receipt = json.loads(path.read_text())
    pid = receipt.get('watchdog_pid', receipt.get('pid'))
    info = process(pid) if isinstance(pid, int) and pid > 1 else None
    if info is None or info.session != pid:
        raise ValueError(f'Expected a live detached session for {path}')
    argv = (PROC / str(pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if 'watchdog_pid' in receipt:
        valid = (receipt['host'] == hostname and Path(argv[0]).name == 'timeout'
                 and any(Path(arg).name == 'run_residual_k3.py' for arg in argv))
        expected = {'--run': str(path.parent), '--host': hostname, '--gpu': receipt['gpu'],
                    '--role': str(receipt['role']), '--revision': receipt['revision']}
    else:
        valid = any(Path(arg).name == 'close_k3_gate.py' for arg in argv[1:3])
        expected = {'--output': str(path.parent), '--summary': receipt['summary']}
    if not valid or any(following(argv, key) != value for key, value in expected.items()):
        raise ValueError(f'Process arguments differ from the launch receipt: {path}')
    return pid, info.started


def following(argv, flag):
    return next((value for key, value in zip(argv, argv[1:]) if key == flag), None)


def owned_processes(known):
    table = {int(entry.name): process(int(entry.name)) for entry in PROC.iterdir() if entry.name.isdecimal()}
    active = {pid: info for pid, info in table.items() if info and known.get(pid) == info.started}
    while active:
        parents = set(active)
        sessions = {info.session for info in active.values() if known.get(info.session) is not None
                    and (table.get(info.session) is None or table[info.session].started == known[info.session])}
        active.update({pid: info for pid, info in table.items()
                       if info and (info.session in sessions or info.parent in parents)})
        if len(active) == len(parents):
            break
    known.update((pid, info.started) for pid, info in active.items())
    return active


def send(active, sig):
    issued = []
    for pid, expected in active.items():
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            current = process(pid)
            if current and current.started == expected.started:
                signal.pidfd_send_signal(fd, sig)
                issued.append(pid)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)
    return issued


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def watch(args):
    armed = json.loads((args.output / 'scope.json').read_text())
    if armed['boot_id'] != (PROC / 'sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Node boot identity changed')
    known = {item['pid']: item['started'] for item in armed['sessions']}
    active = owned_processes(known)
    probe = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(probe, 0)
    finally:
        os.close(probe)
    save(args.output / 'watching.json', {'active_pids': sorted(active), 'deadline': args.deadline})
    while active and time.time() < args.deadline - 5:
        time.sleep(max(0, min(15, args.deadline - 5 - time.time())))
        active = owned_processes(known)
    terminated = send(active, signal.SIGTERM)
    killed = []
    if active:
        time.sleep(max(0, args.deadline - time.time()))
        active = owned_processes(known)
        for _ in range(10):
            if not active:
                break
            killed.extend(send(active, signal.SIGKILL))
            time.sleep(.2)
            active = owned_processes(known)
    result = {'status': 'survivors' if active else ('cutoff_applied' if terminated or killed else 'finished'),
              'term_pids': terminated, 'kill_pids': sorted(set(killed)),
              'remaining_pids': sorted(active), 'finished_at': time.time(), 'deadline': args.deadline}
    save(args.output / 'complete.json', result)
    if active:
        raise RuntimeError(f'Owned processes still alive at cutoff: {sorted(active)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--state', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.host != os.uname().nodename:
        parser.error('Cutoff must run on the exact workload host')
    if not args.detach:
        watch(args)
        return
    if args.deadline < time.time() + 5:
        parser.error('Set a future deadline with at least five seconds of grace')
    sessions = []
    for path in args.state:
        pid, started = verified_leader(path, args.host)
        sessions.append({'receipt': str(path), 'pid': pid, 'started': started})
    if len({item['pid'] for item in sessions}) != len(sessions):
        parser.error('Duplicate session in requested scope')
    args.output.mkdir(exist_ok=False)
    save(args.output / 'scope.json', {'boot_id': (PROC / 'sys/kernel/random/boot_id').read_text().strip(),
                                      'host': args.host, 'sessions': sessions, 'deadline': args.deadline})
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), *(arg for arg in sys.argv[1:] if arg != '--detach')]
    with (args.output / 'controller.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    receipt = {'pid': child.pid, 'deadline': args.deadline, 'sessions': len(sessions), 'host': args.host}
    save(args.output / 'armed.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
