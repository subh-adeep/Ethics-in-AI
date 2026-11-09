"""Build a small EC2 test bundle without launching the full experiment."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT/'codebase/results/ec2_phi4_preflight_20261020'
REMOTE = '/home/ubuntu/ethics-ec2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    full = PROJECT/'codebase/results/exact_lookup_full_20261020'
    files = [PROJECT/'codebase/src/analysis'/name for name in (
        'ec2_phi4_preflight.py', 'ec2_job_lifecycle.py', 'exact_lookup_smoke.py',
        'web_search_pilot.py', 'local_gpu_followup.py', 'medqa_web_dual_gpu.py',
        'audit_saved_reply_quality.py')]
    files.extend([PROJECT/'codebase/src/benchmark/answer_parser.py',
                  PROJECT/'kaggle/kernel_tools_v2/parallel_capture_20261014/parallel_assets/runner.py'])
    payload = {f'PROJECT/{p.relative_to(PROJECT).as_posix()}': p.read_bytes() for p in files}
    for package in ('analysis', 'benchmark'):
        payload[f'PROJECT/codebase/src/{package}/__init__.py'] = b''
    for name in ('manifest.json', 'protocol.json'):
        payload[name] = (full/name).read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    payload['source_hashes.json'] = json.dumps(hashes, indent=2).encode()
    # Explicit execution only: none of these services is enabled at boot.
    units = {}
    for case in ('success', 'failure', 'timeout'):
        output = f'{REMOTE}/runs/lifecycle-{case}'
        units[f'ethics-lifecycle-{case}.service'] = service(
            f'/usr/bin/python3 /usr/local/lib/ethics-ec2/lifecycle.py probe --output {output} --mode {case}',
            output, 3 if case == 'timeout' else 30, False)
    output = f'{REMOTE}/runs/phi4-smoke'
    units['ethics-phi4-smoke.service'] = service(
        f'/home/ubuntu/ethics-venv/bin/python {REMOTE}/PROJECT/codebase/src/analysis/ec2_phi4_preflight.py',
        output, 1200, True)
    units['ethics-phi4-deadline.service'] = '\n'.join([
        '[Unit]', 'Description=Backup deadline for EC2 Phi-4 test', '[Service]',
        'Type=oneshot', 'Environment=SERVICE_RESULT=backup-deadline',
        f'ExecStart=/usr/bin/python3 /usr/local/lib/ethics-ec2/lifecycle.py finalize --output {REMOTE}/runs/backup-deadline --poweroff', ''])
    units['ethics-phi4-deadline.timer'] = '\n'.join([
        '[Unit]', 'Description=Stop EC2 if the Phi-4 test exceeds 25 minutes',
        '[Timer]', 'OnActiveSec=25min', 'AccuracySec=1s',
        'Unit=ethics-phi4-deadline.service', ''])
    for name, value in units.items():
        payload[f'deployment/{name}'] = value.encode()
    install = '''#!/bin/bash
set -euo pipefail
cd /home/ubuntu/ethics-ec2
sudo install -d -m 755 /usr/local/lib/ethics-ec2
sudo install -m 644 PROJECT/codebase/src/analysis/ec2_job_lifecycle.py /usr/local/lib/ethics-ec2/lifecycle.py
for case in lifecycle-success lifecycle-failure lifecycle-timeout phi4-smoke backup-deadline; do
  install -d -m 755 "runs/$case"
done
sudo install -m 644 deployment/*.service deployment/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/ethics-*.service /etc/systemd/system/ethics-*.timer
echo EC2_UNITS_INSTALLED_NOT_STARTED
'''
    payload['deployment/install.sh'] = install.encode()
    (OUT/'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    guards = [PROJECT/'codebase/src/benchmark/answer_parser.py',
              PROJECT/'Midsem report/revision_from_rough_20261015/midsem_report.pdf',
              full/'manifest.json', full/'protocol.json', full/'kaggle_bundle/run_full.py']
    (OUT/'source_guards.json').write_text(json.dumps({str(p): sha(p) for p in guards}, indent=2), encoding='utf-8')
    with tarfile.open(OUT/'preflight_bundle.tar.gz', 'w:gz') as archive:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    (OUT/'plan.json').write_text(json.dumps(dict(
        instance_id='i-03f96c0b0edb6326a', region='us-east-1',
        scope='EC2 setup and lifecycle validation only; full run not launched',
        model='phi4', conversations=4, medqa=2, math=2, answer_turns=12,
        answer_tokens=160, optional_tool_selection=True, exact_id_one_entry=True,
        runtime_limit_seconds=1200, independent_backup_deadline_seconds=1500,
        auto_start_on_boot=False, shutdown_behavior_verified='stop'), indent=2), encoding='utf-8')
    print(OUT/'preflight_bundle.tar.gz')


def service(command, output, timeout, poweroff):
    suffix = ' --poweroff' if poweroff else ''
    return '\n'.join([
        '[Unit]', 'Description=Isolated EC2 experiment preflight',
        'Wants=network-online.target', 'After=network-online.target',
        '[Service]', 'Type=exec', 'User=ubuntu', 'Group=ubuntu',
        f'WorkingDirectory={REMOTE}', 'Environment=PYTHONUNBUFFERED=1',
        'Environment=CUDA_VISIBLE_DEVICES=0', 'Environment=TOKENIZERS_PARALLELISM=false',
        f'ExecStart={command}',
        f'ExecStopPost=+/usr/bin/python3 /usr/local/lib/ethics-ec2/lifecycle.py finalize --output {output}{suffix}',
        f'RuntimeMaxSec={timeout}', 'TimeoutStopSec=45', 'KillMode=control-group',
        'Restart=no', f'StandardOutput=append:{output}/console.log',
        f'StandardError=append:{output}/console.log', ''])


if __name__ == '__main__':
    build()
