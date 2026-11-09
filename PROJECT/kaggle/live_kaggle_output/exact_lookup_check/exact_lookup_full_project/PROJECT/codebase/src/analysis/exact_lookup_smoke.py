"""One exact answer-sheet entry by ID; isolated 10 MedQA + 10 MATH smoke test."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import web_search_pilot as pilot
from analysis import local_gpu_followup as local
from analysis.medqa_web_dual_gpu import MODELS, REVISIONS

OUT = pilot.PROJECT/'codebase/results/exact_lookup_smoke_20261019'


def prepare():
    previous_path = pilot.PROJECT/'codebase/results/web_search_pilot_20261018/manifest.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    source = json.loads(pilot.MANIFEST_SOURCE.read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in source['items']}
    items = [lookup[r['id']] for r in previous['items']]
    assert len({r['id'] for r in items}) == 20
    assert Counter(r['dataset'] for r in items) == {'medqa': 10, 'math': 10}
    assert all(r['level'] in (1, 2, 'Level 1', 'Level 2') and r.get('solution') for r in items if r['dataset'] == 'math')
    manifest = dict(items=items, total=20, selection='Same fixed 10 MedQA + 10 MATH IDs as earlier web pilot; no selection by new outcomes',
        items_sha256=pilot.sha(items), source_manifest_sha256=pilot.sha(pilot.MANIFEST_SOURCE),
        previous_pilot_manifest_sha256=pilot.sha(previous_path))
    protocol = dict(version='exact-id-one-entry-smoke-1', models=list(MODELS), revisions=REVISIONS,
        setting='Exact ID lookup, one answer-sheet entry, optional tool choice',
        questions=20, dataset_counts={'medqa':10,'math':10}, conversations=60, final_answer_turns=180,
        answer_max_new_tokens=160, fresh_p0=True, original_prompts=True, original_system_prompt=local.SYSTEM,
        original_single_token_action_selection=True, tool_name='retrieve_answer_sheet',
        lookup_method='Python uses item ID in a validated answer-sheet dictionary; original controller question text is not ranked',
        entries_per_call=1, medqa_evidence='Question, correct option and answer text; no generated explanation',
        math_evidence='Question, numerical answer and original worked solution',
        full_answer_sheet_in_model_input=False, history_preserved=True,
        hidden_evaluation_fields=['supports_ground_truth','false_answer_key','false_answer_text','false_answer_source'],
        deliberate_reference_answer_evidence=True, scoring='Same unchanged parser-v4, provisional',
        scope='Smoke sample only. No automatic full-cohort run.',
        model_revisions=REVISIONS, parallel_gpus=2)
    OUT.mkdir(parents=True, exist_ok=True)
    for name,value in [('manifest.json',manifest),('protocol.json',protocol)]:
        path=OUT/name
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8')) == value, f'Incompatible {name}'
        else:
            pilot.atomic_json(path,value)
    return manifest,protocol


def shards(manifest):
    result = [dict(manifest, items=manifest['items'][i::2], shard=i) for i in range(2)]
    assert all(Counter(r['dataset'] for r in s['items']) == {'medqa':5,'math':5} for s in result)
    assert not ({r['id'] for r in result[0]['items']} & {r['id'] for r in result[1]['items']})
    return result


class ExactAnswerSheet:
    def __init__(self, items, folder=None):
        self.items = {r['id']: r for r in items}
        assert len(self.items) == len(items), 'Duplicate lookup ID'
        self.folder = Path(folder) if folder else pilot.OUT
        self.context = {}

    def __call__(self, original_query, item):
        ref = self.items[item['id']]  # Fail on missing ID; never search or guess.
        assert ref['dataset'] == item['dataset'] and ref['question'] == item['question'], 'ID/question mismatch'
        document = dict(id=ref['id'], question=ref['question'], answer=ref['ground_truth'],
                        answer_text=ref.get('ground_truth_text',''), solution=ref.get('solution',''))
        evidence = dict(status='ok', backend='answer_sheet_exact_id', documents=[document], supports_ground_truth=True)
        event=dict(self.context)
        event.update(at=pilot.now(),lookup_key=item['id'],dataset=item['dataset'],original_controller_query=original_query,
            original_query_used_for_ranking=False,entry_count=1,evidence=evidence)
        pilot.append_jsonl(self.folder/'lookup_events.jsonl',event)
        return evidence


def run_model(name, manifest, sheet_items, run_label='smoke'):
    runner = pilot.frozen_runner()
    contract = dict(model=name, items_sha256=pilot.sha(manifest['items']), sheet_sha256=pilot.sha(sheet_items),
        protocol_sha256=pilot.sha(json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))))
    path = pilot.OUT/f'{name}_checkpoint.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(model=name,contract=contract,dialogues=[])
    assert saved['contract'] == contract
    by_id = {r['id']:r for r in saved['dialogues']}
    assert len(by_id) == len(saved['dialogues'])
    if all('p2' in by_id.get(item['id'],{}) for item in manifest['items']):
        return saved
    backend = pilot.PilotGPU(name)
    backend.metadata.update(execution_host=f'Kaggle exact ID lookup {run_label}', physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        changed_hardware_from_original='Actual Kaggle GPU in gpu field', attention_note=f'SDPA for every {run_label} turn')
    sheet = ExactAnswerSheet(sheet_items)
    try:
        for item in manifest['items']:
            row = by_id.get(item['id'])
            if row is None:
                row = dict(id=item['id'], dataset=item['dataset'], setting='exact_id_one_entry', policy='optional',
                    history=[dict(role='system',content=local.SYSTEM),dict(role='user',content=item['p0_prompt'])])
                saved['dialogues'].append(row)
                by_id[item['id']] = row
            for turn in ('p0','p1','p2'):
                if turn in row:
                    continue
                context = dict(model=name,question_id=item['id'],dataset=item['dataset'],turn=turn)
                backend.context = context
                sheet.context = context
                before = list(row['history'])
                if turn == 'p0':
                    response = backend.generate(before)
                    tool,augmented = None,before
                else:
                    before += [dict(role='user',content=item[f'{turn}_prompt'])]
                    response,tool,augmented = runner.challenged_turn(backend,before,'T1',item,sheet,'optional')
                    if tool['tool_invoked']:
                        tool.update(lookup_method='exact_question_id', lookup_key=item['id'], entries_returned=1)
                result = dict(response=response, evaluation=pilot.score(response,item,runner), tool=tool,
                              execution_metadata=dict(backend.metadata))
                if tool and tool['tool_invoked']:
                    docs = tool['evidence']['documents']
                    assert len(docs)==1 and docs[0]['id']==item['id'] and docs[0]['answer']==item['ground_truth']
                    observation = json.dumps({k:v for k,v in tool['evidence'].items() if k not in ('supports_ground_truth','cache_hit')},ensure_ascii=False)
                    assert any(observation in m['content'] for m in augmented if m['role']=='user')
                    result['delivery_check'] = dict(exact_observation_in_generation_input=True,
                        correct_matching_entry=True,entry_count=1,observation_sha256=pilot.sha(observation))
                row[turn] = result
                row['history'] = augmented + [dict(role='assistant',content=response['text'])]
                saved.update(completed=sum('p2' in r for r in saved['dialogues']),total=len(manifest['items']),updated_at=pilot.now())
                pilot.atomic_json(path,saved)
                print('TURN',os.environ.get('CUDA_VISIBLE_DEVICES'),name,item['dataset'],item['id'],turn,
                    'tool',bool(tool and tool['tool_invoked']),'completed',saved['completed'],'/',saved['total'],flush=True)
    finally:
        backend.close()
    return saved


def validate(folder, manifest, models=MODELS):
    items = {r['id']:r for r in manifest['items']}
    summaries = []
    for name in models:
        saved = json.loads((folder/f'{name}_checkpoint.json').read_text(encoding='utf-8'))
        assert len(saved['dialogues'])==len(items) and {r['id'] for r in saved['dialogues']}==set(items)
        calls = [json.loads(s) for s in (folder/f'{name}_model_calls.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in saved['dialogues']:
            for turn in ('p0','p1','p2'):
                r = row[turn]
                assert r['response']['max_new_tokens']==160 and r['response']['generated_tokens']<=160
                tool = r['tool']
                if tool and tool['tool_invoked']:
                    docs=tool['evidence']['documents']
                    assert len(docs)==1 and docs[0]['id']==row['id']
                    assert docs[0]['answer']==items[row['id']]['ground_truth']
                    assert docs[0]['solution']==items[row['id']].get('solution','')
                    observation=json.dumps({k:v for k,v in tool['evidence'].items() if k not in ('supports_ground_truth','cache_hit')},ensure_ascii=False)
                    actual=[c for c in calls if c['kind']=='generation' and c['question_id']==row['id'] and c['turn']==turn
                        and c['response']['text']==r['response']['text']
                        and any(observation in m['content'] for m in c['messages'] if m['role']=='user')]
                    assert actual, 'One matching entry was not present in actual generation input'
                    assert all('supports_ground_truth' not in m['content'] for c in actual for m in c['messages'])
                    if turn=='p1':
                        assert any(observation in m['content'] for c in calls if c['kind']=='generation'
                            and c['question_id']==row['id'] and c['turn']=='p2' for m in c['messages']), 'P1 evidence missing at P2'
        for domain in ('medqa','math'):
            selected=[r for r in saved['dialogues'] if r['dataset']==domain]
            for turn in ('p1','p2'):
                invocations=[r[turn] for r in selected if r[turn]['tool']['tool_invoked']]
                summaries.append(dict(model=name,dataset=domain,turn=turn,questions=len(selected),tool_calls=len(invocations),
                    correct_entry_delivered=len(invocations),lookup_failures=0))
    return summaries


def worker(index):
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    local.MODEL_ROOT = Path('/tmp/ethics_exact_lookup_model_cache')
    full = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest = shards(full)[index]
    pilot.OUT = OUT/f'gpu{index}'
    pilot.OUT.mkdir(parents=True,exist_ok=True)
    pilot.atomic_json(pilot.OUT/'gpu_environment.json',dict(at=pilot.now(),physical_gpu=index,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0)))
    print('GPU_READY',index,torch.cuda.get_device_name(0),'smoke_questions',len(manifest['items']),flush=True)
    for name in MODELS:
        run_model(name,manifest,full['items'])
    summary=validate(pilot.OUT,manifest)
    pilot.atomic_json(pilot.OUT/'completed.json',dict(at=pilot.now(),physical_gpu=index,rows=summary))
    print('SMOKE_SHARD_COMPLETE',index,json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=[0,1])
    args=parser.parse_args()
    try:
        if args.worker is None:
            manifest,protocol=prepare()
            print(json.dumps(dict(questions=len(manifest['items']),shards=[len(s['items']) for s in shards(manifest)],protocol=protocol)))
        else:
            worker(args.worker)
    except Exception:
        pilot.append_jsonl(pilot.OUT/'runtime_errors.jsonl',dict(at=pilot.now(),error=traceback.format_exc()))
        raise
