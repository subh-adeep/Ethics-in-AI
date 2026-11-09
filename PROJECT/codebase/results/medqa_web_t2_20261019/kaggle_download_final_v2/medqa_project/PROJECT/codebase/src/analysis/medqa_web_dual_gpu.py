"""Isolated MedQA T2 run: fixed cohort, two GPU shards, original dialogue protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from analysis import web_search_pilot as pilot
from analysis import local_gpu_followup as local

OUT = pilot.PROJECT / 'codebase/results/medqa_web_t2_20261019'
MODELS = ('qwen', 'llama', 'phi4')
REVISIONS = {
    'qwen': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
    'llama': 'bb1d317a108579fb40e646af8924a5e7ec5604b1',
    'phi4': 'cece1fd36f04ff79f55ec861f206ca4e16acea6e',
}


def prepare():
    """Keep the established 146 IDs; do not select again using fresh answers."""
    cohort = {r['id']: r['domain'] for r in
              (json.loads(s) for s in pilot.COHORT_SOURCE.read_text(encoding='utf-8').splitlines())}
    ids = sorted(k for k, domain in cohort.items() if domain == 'medqa')
    assert len(ids) == 146 and len(set(ids)) == 146
    source = json.loads(pilot.MANIFEST_SOURCE.read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in source['items']}
    items = [lookup[k] for k in ids]
    assert all(r['dataset'] == 'medqa' for r in items)
    manifest = dict(created_at=pilot.now(), items=items, total=146,
                    selection='All 146 fixed shared initially-correct MedQA questions; no new filtering',
                    cohort_source_sha256=pilot.sha(pilot.COHORT_SOURCE),
                    canonical_manifest_sha256=pilot.sha(pilot.MANIFEST_SOURCE),
                    items_sha256=pilot.sha(items))
    OUT.mkdir(parents=True, exist_ok=True)
    existing = OUT / 'manifest.json'
    if existing.exists():
        saved = json.loads(existing.read_text(encoding='utf-8'))
        assert saved['items_sha256'] == manifest['items_sha256']
        manifest = saved
    pilot.atomic_json(existing, manifest)
    protocol = dict(created_at=pilot.now(), dataset='MedQA', questions=146,
        models=list(MODELS), setting='T2 optional web search', parallel_gpus=2,
        shard_sizes=[73, 73], conversations=438, final_answer_turns=1314,
        pressure_search_opportunities=876, model_revisions=REVISIONS,
        original_prompts=True, answer_max_new_tokens=160, query_max_new_tokens=64,
        fresh_p0=True, fixed_cohort_not_reselected_on_new_p0=True,
        history_preserved=True, original_single_token_action_selection=True,
        search_adapter='Unchanged validated pilot-search-2, DDGS 9.16.0 auto backend',
        evidence='Up to three search-result snippets with titles and URLs; not full pages',
        status='Exploratory T2; operational smoke test is not a retrieval-accuracy test',
        known_limits=['Generated queries can omit clues or add unsupported details',
                      'Nonempty snippets can be incomplete or irrelevant',
                      'Scoring of unclear/conflicting answers remains provisional'],
        smoke='Two questions per model per GPU, plus independent connectivity probes; reuse smoke answers in full run',
        hidden_annotations=['ground_truth', 'false_answer_key', 'false_answer_text', 'supports_ground_truth'],
        isolation='New checkpoints only; existing report, parser and T0/T1 results are not changed')
    pilot.atomic_json(OUT / 'protocol.json', protocol)
    return manifest, protocol


def shards(manifest):
    items = manifest['items']
    assert len(items) == 146 and len({r['id'] for r in items}) == 146
    result = [dict(manifest, items=items[i::2], total=73, shard=i) for i in range(2)]
    assert not ({r['id'] for r in result[0]['items']} & {r['id'] for r in result[1]['items']})
    assert all(r['dataset'] == 'medqa' for s in result for r in s['items'])
    return result


def run_model(name, manifest, limit=None):
    """Checkpoint every answer. Smoke and full passes share the same contract."""
    path = pilot.OUT / f'{name}_checkpoint.json'
    contract = dict(model=name, items_sha256=pilot.sha(manifest['items']),
                    protocol_sha256=pilot.sha(json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))))
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(contract=contract, model=name, dialogues=[])
    assert saved['contract'] == contract, 'Refuse incompatible checkpoint'
    target = manifest['items'] if limit is None else manifest['items'][:limit]
    by_id = {r['id']: r for r in saved['dialogues']}
    assert len(by_id) == len(saved['dialogues'])
    if all('p2' in by_id.get(item['id'], {}) for item in target):
        return saved
    runner = pilot.frozen_runner()
    backend = pilot.PilotGPU(name)
    backend.metadata.update(execution_host='Kaggle private MedQA T2',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), model_revision=REVISIONS[name],
        changed_hardware_from_original='Actual Kaggle GPU recorded in gpu field',
        attention_note='SDPA for all turns in this new run')
    search = pilot.StrictSearch()
    try:
        for item in target:
            row = by_id.get(item['id'])
            if row is None:
                row = dict(id=item['id'], dataset='medqa', tier='T2', policy='optional',
                    history=[dict(role='system', content=local.SYSTEM), dict(role='user', content=item['p0_prompt'])])
                saved['dialogues'].append(row)
                by_id[item['id']] = row
            for turn in ('p0', 'p1', 'p2'):
                if turn in row:
                    continue
                backend.context = dict(model=name, question_id=item['id'], dataset='medqa', turn=turn)
                search.context = dict(phase='optional_model_tool_call', model=name, turn=turn)
                before = list(row['history'])
                if turn == 'p0':
                    response = backend.generate(before)
                    log, augmented = None, before
                else:
                    before += [dict(role='user', content=item[f'{turn}_prompt'])]
                    response, log, augmented = runner.challenged_turn(backend, before, 'T2', item, search, 'optional')
                row[turn] = dict(response=response, evaluation=pilot.score(response, item, runner),
                    tool=log, execution_metadata=dict(backend.metadata))
                if log and log['tool_invoked']:
                    observation = json.dumps({k: v for k, v in log['evidence'].items()
                        if k not in ('supports_ground_truth', 'cache_hit')}, ensure_ascii=False)
                    assert any(observation in m['content'] for m in augmented if m['role'] == 'user')
                    row[turn]['delivery_check'] = dict(exact_observation_in_generation_input=True,
                        observation_sha256=pilot.sha(observation), augmented_messages_sha256=pilot.sha(augmented),
                        document_count=len(log['evidence']['documents']), nonempty_evidence_delivered=log['tool_success'])
                row['history'] = augmented + [dict(role='assistant', content=response['text'])]
                saved.update(metadata=dict(backend.metadata), updated_at=pilot.now(),
                    completed=sum('p2' in r for r in saved['dialogues']), total=len(manifest['items']))
                pilot.atomic_json(path, saved)
                print('TURN', os.environ.get('CUDA_VISIBLE_DEVICES'), name, item['id'], turn,
                    'tokens', response['generated_tokens'], 'tool', bool(log and log['tool_invoked']),
                    'search', log['evidence']['status'] if log and log['evidence'] else '-',
                    'completed', saved['completed'], '/', saved['total'], flush=True)
    finally:
        backend.close()
    return saved


def validate_outputs(folder, manifest, expected):
    """Check exact model inputs, budgets, uniqueness and completed cohort IDs."""
    expected_ids = {r['id'] for r in manifest['items'][:expected]}
    summary = []
    for name in MODELS:
        saved = json.loads((folder/f'{name}_checkpoint.json').read_text(encoding='utf-8'))
        rows = saved['dialogues']
        assert {r['id'] for r in rows} == expected_ids
        calls = [json.loads(s) for s in (folder/f'{name}_model_calls.jsonl').read_text(encoding='utf-8').splitlines()]
        invoked = delivered = nonempty = 0
        for row in rows:
            assert row['dataset'] == 'medqa'
            for turn in ('p0', 'p1', 'p2'):
                result = row[turn]
                assert result['response']['max_new_tokens'] == 160
                assert result['response']['generated_tokens'] <= 160
                tool = result['tool']
                if tool and tool['tool_invoked']:
                    invoked += 1
                    obs = json.dumps({k: v for k, v in tool['evidence'].items()
                        if k not in ('supports_ground_truth', 'cache_hit')}, ensure_ascii=False)
                    found = [c for c in calls if c['kind'] == 'generation' and c['question_id'] == row['id']
                        and c['turn'] == turn and c['response']['text'] == result['response']['text']
                        and any(obs in m['content'] for m in c['messages'] if m['role'] == 'user')]
                    assert found, 'Evidence missing from actual generation call log'
                    assert all('supports_ground_truth' not in m['content'] for c in found for m in c['messages'])
                    delivered += 1
                    nonempty += bool(tool['tool_success'])
        summary.append(dict(model=name, questions=len(rows), tool_calls=invoked,
                            delivered=delivered, nonempty=nonempty))
    return summary


def worker(shard):
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    local.MODEL_ROOT = Path('/tmp/ethics_medqa_t2_model_cache')
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest = shards(manifest)[shard]
    pilot.OUT = OUT / f'gpu{shard}'
    pilot.OUT.mkdir(parents=True, exist_ok=True)
    pilot.atomic_json(pilot.OUT/'shard_manifest.json', manifest)
    pilot.atomic_json(pilot.OUT/'gpu_environment.json', dict(at=pilot.now(), physical_gpu=shard,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], logical_device_count=1,
        name=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda))
    print('GPU_READY', shard, torch.cuda.get_device_name(0), 'questions', len(manifest['items']), flush=True)
    search = pilot.StrictSearch()
    search.context = dict(phase='connectivity_probe', model=None, turn=None)
    probe_rows = []
    for query in ['site:nih.gov HIV diagnosis infants', 'site:cdc.gov Helicobacter pylori breath test',
                  'alpha 1 antitrypsin deficiency elastase']:
        result = search(query, dict(id='connectivity_probe', dataset='medqa'))
        probe_rows.append(dict(query=query, evidence=result))
        if result['status'] == 'ok':
            break
    pilot.atomic_json(pilot.OUT/'connectivity_probe.json', probe_rows)
    assert any(r['evidence']['status'] == 'ok' for r in probe_rows), 'Search connectivity gate failed'
    for name in MODELS:
        run_model(name, manifest, limit=2)
    smoke_rows = validate_outputs(pilot.OUT, manifest, 2)
    pilot.atomic_json(pilot.OUT/'smoke_ready.json', dict(at=pilot.now(), physical_gpu=shard, rows=smoke_rows))
    print('SMOKE_READY', shard, json.dumps(smoke_rows), flush=True)
    deadline = time.monotonic() + 900
    while not (OUT/'smoke_passed.json').exists():
        if (OUT/'abort.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Full run not released by controller')
        time.sleep(2)
    print('FULL_RUN_STARTED', shard, flush=True)
    for name in MODELS:
        run_model(name, manifest)
    result = validate_outputs(pilot.OUT, manifest, 73)
    pilot.atomic_json(pilot.OUT/'completed.json', dict(at=pilot.now(), rows=result))
    print('SHARD_COMPLETE', shard, json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        if args.worker is None:
            manifest, protocol = prepare()
            print(json.dumps(dict(questions=len(manifest['items']), shard_sizes=[len(s['items']) for s in shards(manifest)], protocol=protocol)))
        else:
            worker(args.worker)
    except Exception:
        pilot.append_jsonl(pilot.OUT/'runtime_errors.jsonl', dict(at=pilot.now(), error=traceback.format_exc()))
        raise
