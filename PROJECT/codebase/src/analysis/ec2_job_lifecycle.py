"""Durable systemd job finalization; shutdown is an explicit deployment option.

No AWS keys are required on the instance. The caller verifies that Ubuntu's
shutdown behavior is Stop. Services are started explicitly and never enabled
at boot. systemd, rather than an interactive SSH connection, owns the job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def finalize(output, poweroff=False, environment=None, shutdown=None):
    output = Path(output)
    environment = os.environ if environment is None else environment
    service_result = environment.get('SERVICE_RESULT', 'unknown')
    record = dict(at=now(), service_result=service_result,
                  exit_code=environment.get('EXIT_CODE'),
                  exit_status=environment.get('EXIT_STATUS'),
                  poweroff_requested=poweroff, full_run_started=False)
    files = {}
    # Include checkpoints and errors on failure, without calling them complete.
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name not in {'lifecycle.json', 'shutdown_error.json'}:
            files[str(path.relative_to(output))] = dict(
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record['saved_files'] = files
    durable_json(output/'lifecycle.json', record)
    if hasattr(os, 'sync'):
        os.sync()
    if poweroff:
        try:
            # --no-block returns after the shutdown transaction is enqueued.
            if shutdown is not None:
                shutdown()
            else:
                subprocess.run(['/usr/bin/systemctl', '--no-block', 'poweroff'],
                               check=True, timeout=10)
        except Exception as error:
            durable_json(output/'shutdown_error.json',
                         dict(at=now(), error=f'{type(error).__name__}: {error}'))
            raise
    return record


def probe(output, mode):
    output = Path(output)
    durable_json(output/'probe_checkpoint.json', dict(at=now(), mode=mode, saved=True))
    if mode == 'failure':
        raise RuntimeError('Intentional preflight error; not an experiment failure')
    if mode == 'timeout':
        while True:
            time.sleep(1)
    durable_json(output/'probe_complete.json', dict(at=now(), mode=mode, passed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['finalize', 'probe'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poweroff', action='store_true')
    parser.add_argument('--mode', choices=['success', 'failure', 'timeout'], default='success')
    args = parser.parse_args()
    if args.action == 'finalize':
        finalize(args.output, poweroff=args.poweroff)
    else:
        probe(args.output, args.mode)
