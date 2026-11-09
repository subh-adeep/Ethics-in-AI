"""Export and verify the isolated pilot's saved evidence; no inference/network."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.web_search_pilot import OUT, sha, atomic_json, summary


def export():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    allowed={r['id'] for r in manifest['items']}
    events=[]
    for event_file in [OUT/'search_events.jsonl',OUT/'phi4_search_events.jsonl']:
        if event_file.exists():
            events.extend(json.loads(x) for x in event_file.open(encoding='utf-8'))
    assert all(e['question_id'] in allowed for e in events)
    documents=[]
    attempts=[]
    for e in events:
        base={k:e.get(k) for k in ['event_id','phase','model','question_id','dataset','turn','query',
                                  'raw_query','search_adapter_revision','normalization','started_at']}
        base.update(status=e['evidence']['status'],cache_hit=e['evidence']['cache_hit'],
                    error=e['evidence'].get('error',''),duration_seconds=e['duration_seconds'])
        attempts.append(dict(base,document_count=len(e['evidence']['documents']),attempts=e['attempts']))
        for rank,d in enumerate(e['evidence']['documents'],1):
            documents.append(dict(base,rank=rank,**d))
    atomic_json(OUT/'search_records.json',attempts)
    atomic_json(OUT/'retrieved_documents.json',documents)
    if documents:
        with (OUT/'retrieved_documents.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(documents[0]))
            writer.writeheader();writer.writerows(documents)
    deliveries=[]
    history_checks=[]
    completed={}
    for name in ['qwen','llama','phi4']:
        checkpoint=OUT/f'{name}_checkpoint.json'
        if not checkpoint.exists(): continue
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        completed[name]=saved.get('completed',0)
        calls=[json.loads(l) for l in (OUT/f'{name}_model_calls.jsonl').open(encoding='utf-8')]
        generations={c['messages_sha256']:c for c in calls if c['kind']=='generation'}
        for row in saved['dialogues']:
            assert row['id'] in allowed
            for turn in ['p1','p2']:
                if turn not in row: continue
                reply=row[turn]
                tool=reply['tool']
                if not tool['tool_invoked']:continue
                check=reply['delivery_check']
                generation=generations[check['augmented_messages_sha256']]
                observation=json.dumps({k:v for k,v in tool['evidence'].items()
                                        if k not in ['supports_ground_truth','cache_hit']},ensure_ascii=False)
                assert sha(observation)==check['observation_sha256']
                assert any(observation in m['content'] for m in generation['messages'] if m['role']=='user')
                assert 'supports_ground_truth' not in observation
                assert 'ground_truth' not in observation
                assert 'false_answer_key' not in observation
                # The P1 evidence is retained for the subsequent P2 response.
                p1=row.get('p1',{}).get('tool')
                inherited=None
                if turn=='p2' and p1 and p1['tool_invoked']:
                    text=json.dumps({k:v for k,v in p1['evidence'].items()
                                     if k not in ['supports_ground_truth','cache_hit']},ensure_ascii=False)
                    inherited=any(text in m['content'] for m in generation['messages'] if m['role']=='user')
                    assert inherited
                deliveries.append(dict(model=name,id=row['id'],dataset=row['dataset'],turn=turn,
                    status=tool['evidence']['status'],exact_generation_input_verified=True,
                    p1_evidence_retained_at_p2=inherited,document_count=len(tool['evidence']['documents'])))
            p1=row.get('p1',{}).get('tool')
            if 'p2' in row and p1 and p1['tool_invoked']:
                p2_generations=[c for c in calls if c['kind']=='generation' and c['question_id']==row['id']
                    and c['turn']=='p2' and c['response']==row['p2']['response']]
                assert p2_generations
                prior=json.dumps({k:v for k,v in p1['evidence'].items()
                                  if k not in ['supports_ground_truth','cache_hit']},ensure_ascii=False)
                retained=any(prior in m['content'] for m in p2_generations[-1]['messages'] if m['role']=='user')
                assert retained
                history_checks.append(dict(model=name,id=row['id'],p1_evidence_in_p2_generation_input=True))
    guards=json.loads((OUT/'original_file_guards.json').read_text(encoding='utf-8'))
    unchanged={path:sha(Path(path))==digest for path,digest in guards.items()}
    assert all(unchanged.values()),'An original file changed'
    atomic_json(OUT/'delivery_verification.json',dict(completed=completed,checks=deliveries,history_checks=history_checks,
        original_files_unchanged=unchanged,all_ids_in_fixed_20=True,
        note='Verifies exact source text in logged generation inputs; does not establish faithful use by the model.'))
    summary()
    print(json.dumps(dict(completed=completed,search_events=len(events),documents=len(documents),verified_deliveries=len(deliveries))))


if __name__=='__main__':
    export()
