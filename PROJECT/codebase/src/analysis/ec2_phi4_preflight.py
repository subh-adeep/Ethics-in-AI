"""Four isolated Phi-4 conversations on EC2, using the existing exact-ID protocol."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    home = Path('/home/ubuntu/ethics-ec2')
    project = home/'PROJECT'
    sys.path.insert(0, str(project/'codebase/src'))
    # Models are downloaded before importing the offline-only experiment helper.
    from analysis import exact_lookup_smoke as engine
    from analysis import local_gpu_followup as local
    from analysis import web_search_pilot as pilot
    from analysis.ec2_job_lifecycle import durable_json
    import torch

    output = home/'runs/phi4-smoke'
    output.mkdir(parents=True, exist_ok=True)
    full = json.loads((home/'manifest.json').read_text(encoding='utf-8'))
    items = []
    for domain in ('medqa', 'math'):
        items.extend([r for r in full['items'] if r['dataset'] == domain][:2])
    assert len(items) == 4 and Counter(r['dataset'] for r in items) == {'medqa': 2, 'math': 2}
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    manifest = dict(items=items, total=4, selection='First two fixed IDs per dataset; no selection by new outputs')
    protocol = json.loads((home/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(scope='EC2 preflight only; never part of full-run results', questions=4,
                    conversations=4, final_answer_turns=12, models=['phi4'])
    durable_json(output/'protocol.json', protocol)
    durable_json(output/'manifest.json', manifest)
    environment = dict(at=pilot.now(), host='AWS EC2 i-03f96c0b0edb6326a',
                       gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                       gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                       python=sys.version, torch_cuda=torch.version.cuda,
                       packages={p: importlib.metadata.version(p) for p in
                                 ('torch', 'transformers', 'accelerate', 'bitsandbytes')})
    durable_json(output/'environment.json', environment)
    local.MODEL_ROOT = home/'model-cache'
    pilot.OUT = output
    engine.OUT = output
    engine.run_model('phi4', manifest, full['items'], run_label='ec2_preflight',
                     execution_host='AWS EC2 Tesla T4 preflight; separate from Kaggle and full results')
    rows = engine.validate(output, manifest, models=['phi4'])
    saved = json.loads((output/'phi4_checkpoint.json').read_text(encoding='utf-8'))
    assert saved['completed'] == 4
    # Verify that rerunning a completed checkpoint generates no extra replies.
    calls = output/'phi4_model_calls.jsonl'
    before = pilot.sha(calls)
    engine.run_model('phi4', manifest, full['items'], run_label='ec2_preflight',
                     execution_host='AWS EC2 Tesla T4 preflight; separate from Kaggle and full results')
    assert pilot.sha(calls) == before, 'Resume duplicated model calls'
    durable_json(output/'completed.json', dict(at=pilot.now(), passed=True,
                 conversations=4, answer_turns=12, full_run_started=False,
                 checkpoint_resume_no_duplicate_calls=True, rows=rows))
    print('EC2_PHI4_PREFLIGHT_COMPLETE', json.dumps(rows), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        output = Path('/home/ubuntu/ethics-ec2/runs/phi4-smoke')
        output.mkdir(parents=True, exist_ok=True)
        (output/'runtime_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
