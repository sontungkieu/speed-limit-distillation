"""Stop only detached job sessions identified by their controller receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        return int(fields[2]), int(fields[3]), fields[0], fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def members(pid):
    found = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            number = int(entry.name)
            state = process(number)
            if state and state[1] == pid and state[2] != 'Z':
                found[number] = state
    return found


def verify(path):
    state = json.loads(path.read_text())
    pid = state['pid']
    if state['hostname'] != os.uname().nodename or pid <= 1:
        raise ValueError(f'Invalid controller identity for {path}')
    if state['phase'] in ('complete', 'failed') and not members(pid):
        return None
    info = process(pid)
    if not info or info[:2] != (pid, pid) or info[2] == 'Z':
        raise ValueError(f'Controller is not a live session leader: {path}')
    cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().split('\0')
    if path.name.startswith('role-'):
        expected = ('stack_k1.py', '--run', str(path.parent), '--role', str(state['role']))
    else:
        expected = ('run_checkpoint.py', '--output', str(path.parent), '--step', str(state['step']))
    script, path_flag, directory, role_flag, role = expected
    if (len(cmd) < 2 or Path(cmd[1]).name != script or
            any(flag not in cmd or cmd[cmd.index(flag) + 1] != value
                for flag, value in ((path_flag, directory), (role_flag, role), ('--gpu', state['gpu'])))):
        raise ValueError(f'Controller command does not match {path}')
    return pid, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, nargs='+', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    jobs = [(path, verify(path)) for path in args.state]
    with args.receipt.open('x') as receipt:
        stopped = []
        for path, job in jobs:
            if job is None:
                stopped.append({'state': str(path), 'status': 'already_finished'})
                receipt.write(json.dumps(stopped[-1]) + '\n')
                receipt.flush()
                continue
            pid, info = job
            verified = verify(path)
            if verified is None or verified[0] != pid or verified[1][3] != info[3]:
                raise RuntimeError(f'Controller changed: {path}')
            # Freeze the controller before stopping its children so it cannot start another run.
            os.kill(pid, signal.SIGSTOP)
            for number, state in members(pid).items():
                current = process(number)
                if number != pid and current and current[1] == pid and current[3] == state[3]:
                    try:
                        os.kill(number, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and set(members(pid)) - {pid}:
                time.sleep(.2)
            # Kill the verified leader last to keep the session identity anchored.
            for number in sorted(members(pid), key=lambda value: value == pid):
                current = process(number)
                if current and current[1] == pid:
                    try:
                        os.kill(number, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for _ in range(50):
                if not members(pid):
                    break
                time.sleep(.2)
            else:
                raise RuntimeError(f'Job session still contains live processes: {pid}')
            stopped.append({'state': str(path), 'session': pid, 'status': 'stopped'})
            receipt.write(json.dumps(stopped[-1]) + '\n')
            receipt.flush()
        print(json.dumps(stopped), flush=True)


if __name__ == '__main__':
    main()
