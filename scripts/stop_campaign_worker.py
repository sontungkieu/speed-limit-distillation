"""Stop one verified campaign worker and its separately owned payload session."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def alive(pid):
    try:
        return not (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].startswith('Z')
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.run / f'worker-{args.worker}/state.json').read_text())
    pid = state['pid']
    if state['host'] != os.uname().nodename or pid <= 1 or not alive(pid):
        raise ValueError('Worker is not alive on this host')
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().split('\0')
    if (Path(command[1]).name != 'campaign.py' or command[command.index('--run') + 1] != str(args.run)
            or command[command.index('--worker') + 1] != str(args.worker)
            or command[command.index('--gpu') + 1] != state['gpu']):
        raise ValueError('Worker command differs from receipt')
    with args.receipt.open('x') as receipt:
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not alive(pid):
                break
            time.sleep(.2)
        else:
            raise RuntimeError('Worker did not acknowledge stop')
        child = state.get('child_pid')
        if child and alive(child):
            raise RuntimeError('Payload is still alive')
        receipt.write(json.dumps(dict(worker=args.worker, pid=pid, gpu=state['gpu'], stopped_at=time.time())))
        print('STOPPED', args.worker, pid, flush=True)


if __name__ == '__main__':
    main()
