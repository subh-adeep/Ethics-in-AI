"""Full fixed-cohort exact-ID lookup: 146 MedQA + 215 MATH, optional tool use."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import exact_lookup_smoke as engine
from analysis import web_search_pilot as pilot
from analysis import local_gpu_followup as local

OUT=pilot.PROJECT/'codebase/results/exact_lookup_full_20261020'
MODELS=engine.MODELS
REVISIONS=engine.REVISIONS
validate=engine.validate


def prepare():
    smoke_dir=engine.OUT
    proof=json.loads((smoke_dir/'final_verification.json').read_text(encoding='utf-8'))
    assert proof['passed'] and proof['conversations']==60 and proof['lookup_failures']==0
    cohort={r['id']:r['domain'] for r in
        (json.loads(s) for s in pilot.COHORT_SOURCE.read_text(encoding='utf-8').splitlines())}
    assert Counter(cohort.values())=={'medqa':146,'math':215}
    canonical=json.loads(pilot.MANIFEST_SOURCE.read_text(encoding='utf-8'))
    lookup={r['id']:r for r in canonical['items']}
    items=[lookup[k] for domain in ('medqa','math') for k in sorted(cohort) if cohort[k]==domain]
    assert len(items)==361 and len({r['id'] for r in items})==361
    assert all(r.get('solution') and r['level'] in (1,2,'Level 1','Level 2') for r in items if r['dataset']=='math')
    manifest=dict(items=items,total=361,dataset_counts={'medqa':146,'math':215},items_sha256=pilot.sha(items),
        selection='All 361 fixed shared initially-correct questions; no selection using fresh answers',
        cohort_source_sha256=pilot.sha(pilot.COHORT_SOURCE),canonical_manifest_sha256=pilot.sha(pilot.MANIFEST_SOURCE))
    original_protocol=json.loads((smoke_dir/'protocol.json').read_text(encoding='utf-8'))
    protocol=dict(original_protocol,version='exact-id-one-entry-full-2-model-scheduling',questions=361,
        dataset_counts=manifest['dataset_counts'],conversations=1083,final_answer_turns=3249,
        pressure_tool_opportunities=2166,scope='Full fixed cohort; no BM25 ranking or web search',
        based_on_smoke_protocol_sha256=pilot.sha(smoke_dir/'protocol.json'),
        smoke_verification_sha256=pilot.sha(smoke_dir/'final_verification.json'),
        fresh_p0=True,fixed_cohort_not_reselected_on_new_p0=True,reuse_smoke_answers=False,
        scheduling='One different model per GPU; Qwen on GPU 0, LLaMA on GPU 1, Phi-4 on first available GPU',
        models_running_simultaneously=2,questions_per_model=361,
        initial_gpu_models={'0':'qwen','1':'llama'},queued_model='phi4',
        ordering='Each model starts with one MedQA and one MATH question, then remaining MedQA and MATH IDs',
        checkpoint_after_every_answer=True,isolation='New full-run checkpoints; previous experiment results and report unchanged')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,value in [('manifest.json',manifest),('protocol.json',protocol)]:
        path=OUT/name
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value,f'Incompatible {name}'
        else:pilot.atomic_json(path,value)
    return manifest,protocol


def task_manifest(manifest):
    assert len(manifest['items'])==361 and Counter(r['dataset'] for r in manifest['items'])=={'medqa':146,'math':215}
    med=[r for r in manifest['items'] if r['dataset']=='medqa']
    math=[r for r in manifest['items'] if r['dataset']=='math']
    rows=med[:1]+math[:1]+med[1:]+math[1:]
    assert len(rows)==len({r['id'] for r in rows})==361
    assert {r['id'] for r in rows}=={r['id'] for r in manifest['items']}
    return dict(manifest,items=rows,total=361)


def worker(gpu,name):
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    local.MODEL_ROOT=Path('/tmp/ethics_exact_lookup_model_cache')
    engine.OUT=OUT
    full=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    task=task_manifest(full)
    pilot.OUT=OUT/name
    pilot.OUT.mkdir(parents=True,exist_ok=True)
    environment=dict(at=pilot.now(),physical_gpu=gpu,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        logical_gpu_count=1,gpu=torch.cuda.get_device_name(0),model=name,questions=len(task['items']),
        datasets=dict(Counter(r['dataset'] for r in task['items'])))
    pilot.atomic_json(pilot.OUT/'gpu_environment.json',environment)
    pilot.atomic_json(pilot.OUT/'task_manifest.json',task)
    print('GPU_READY',json.dumps(environment),flush=True)
    engine.run_model(name,task,full['items'],run_label='full')
    summary=validate(pilot.OUT,task,models=[name])
    pilot.atomic_json(pilot.OUT/'completed.json',dict(at=pilot.now(),physical_gpu=gpu,model=name,rows=summary))
    print('MODEL_COMPLETE',gpu,name,json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1]);parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    try:
        if args.gpu is None:
            manifest,protocol=prepare()
            print(json.dumps(dict(questions=len(manifest['items']),protocol=protocol)))
        else:
            assert args.model is not None
            worker(args.gpu,args.model)
    except Exception:
        pilot.append_jsonl(pilot.OUT/'runtime_errors.jsonl',dict(at=pilot.now(),error=traceback.format_exc()))
        raise
