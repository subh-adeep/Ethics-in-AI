"""Fixed 10+10 web-search pilot. Original experimental files are read-only.

Reuses the frozen T2 dialogue protocol and the cached local GPU loader.
Search success requires a usable HTTP(S) URL AND nonempty retrieved text.
All source text here is the search provider's snippet, not a fetched full page.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import time
import traceback
from urllib.parse import urlparse

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from analysis.local_gpu_followup import LocalGPU, SYSTEM, PROJECT, CAPTURE
from benchmark.answer_parser import parse_reply

OUT = PROJECT / 'codebase/results/web_search_pilot_20261018'
MANIFEST_SOURCE = CAPTURE / 'parallel_assets/manifest.json'
COHORT_SOURCE = PROJECT / 'codebase/results/parser_v4_20261018/rescored_replies.jsonl'
RUNNER_SOURCE = CAPTURE / 'parallel_assets/runner.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    if isinstance(value, Path):
        return hashlib.sha256(value.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()


def atomic_json(path, value):
    """Retry Windows sharing violations without regenerating an answer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(12):
        try:
            temporary.replace(path)
            if attempt:
                append_jsonl(OUT/'file_write_retries.jsonl',dict(at=now(),path=str(path),retries=attempt))
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.05 * 2**attempt,1))


def frozen_runner():
    spec = importlib.util.spec_from_file_location('frozen_t2_pilot_protocol', RUNNER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        assert manifest['cohort_source_sha256'] == sha(COHORT_SOURCE)
        assert manifest['manifest_source_sha256'] == sha(MANIFEST_SOURCE)
        return manifest
    cohort = {r['id']: r['domain'] for r in
              (json.loads(line) for line in COHORT_SOURCE.open(encoding='utf-8'))}
    assert Counter(cohort.values()) == {'medqa': 146, 'math': 215}
    source = json.loads(MANIFEST_SOURCE.read_text(encoding='utf-8'))
    by_id = {x['id']: x for x in source['items']}
    rng = random.Random(20261018)
    selected = []
    for domain in ['medqa', 'math']:
        selected += sorted(rng.sample(sorted(k for k,v in cohort.items() if v == domain), 10))
    items = [by_id[k] for k in selected]
    manifest = dict(created_at=now(), seed=20261018, selection='Seeded random sample, 10 per dataset, from fixed shared initially-correct cohort; no selection by search outcomes',
                    cohort_counts=dict(Counter(cohort.values())), pilot_counts=dict(Counter(x['dataset'] for x in items)),
                    cohort_source_sha256=sha(COHORT_SOURCE), manifest_source_sha256=sha(MANIFEST_SOURCE),
                    frozen_runner_sha256=sha(RUNNER_SOURCE), items=items)
    atomic_json(path, manifest)
    atomic_json(OUT / 'protocol.json', dict(created_at=now(), models=['qwen','llama','phi4'],
        setting='T2 optional web search', answer_max_new_tokens=160, query_max_new_tokens=64,
        original_prompts=True, fresh_local_p0=True, history_preserved_p0_p1_p2=True,
        original_single_token_action_selection=True, search_provider='ddgs 9.16.0, auto backend',
        returned_material='Up to three search-result snippets with titles and URLs; full pages are not fetched',
        success_definition='At least one valid http(s) URL with nonempty non-placeholder snippet',
        search_attempts=2, timeout_seconds_per_engine=12, failed_results_cached=False,
        annotations_hidden=['ground_truth','false_answer_key','false_answer_text','supports_ground_truth'],
        fixed_cohort_not_reselected_on_new_p0=True, generation_hardware='Local GTX 1650; original runs used Tesla T4',
        scoring='Parser-v4 provisional only; pilot does not resolve outstanding scoring review',
        separate_preflight='Controller question-text queries for all 20 questions; NOT counted as model tool calls',
        isolation='Original checkpoints, report and main result tables are not modified'))
    return manifest


def normalize_results(raw):
    documents, rejected = [], []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            rejected.append(dict(index=index, reason='not_an_object')); continue
        title = str(row.get('title') or '').strip()
        url = str(row.get('href') or row.get('url') or row.get('link') or '').strip()
        snippet = str(row.get('body') or row.get('snippet') or row.get('description') or '').strip()
        u = urlparse(url)
        reason = None
        if u.scheme not in {'http','https'} or not u.netloc:
            reason = 'missing_or_invalid_url'
        elif not snippet:
            reason = 'empty_retrieved_text'
        elif title.lower() in {'search unavailable','no results found'}:
            reason = 'provider_placeholder'
        if reason:
            rejected.append(dict(index=index, reason=reason)); continue
        if url not in {d['url'] for d in documents}:
            documents.append(dict(title=title, url=url, snippet=snippet))
    return documents[:3], rejected


def normalize_query(query):
    """Presentation quotes must not turn a generated sentence into an exact phrase."""
    query = ' '.join(str(query).split())[:400]
    quote_pairs = {'"':'"', "'":"'", '\u201c':'\u201d', '\u2018':'\u2019'}
    if len(query) >= 2 and query[-1] == quote_pairs.get(query[0]):
        return query[1:-1].strip()
    return query


class StrictSearch:
    def __init__(self):
        self.path = OUT / 'web_cache.json'
        self.cache = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        self.context = {}

    def __call__(self, query, item):
        from ddgs import DDGS
        raw_query = ' '.join(str(query).split())[:400]
        query = normalize_query(raw_query)
        started = time.monotonic()
        key = sha(query)
        event = dict(event_id=f"search_{time.time_ns()}", started_at=now(), query=query,
                     raw_query=raw_query, search_adapter_revision='pilot-search-2',
                     normalization='strip_balanced_outer_quotes' if query != raw_query else 'none',
                     question_id=item['id'], dataset=item['dataset'], **self.context)
        attempts = []
        if key in self.cache:
            evidence = dict(self.cache[key], cache_hit=True)
        elif not query:
            evidence = dict(status='error', backend='ddgs', query=query, retrieved_at=now(),
                            documents=[], error='Empty model-generated query', cache_hit=False)
        else:
            evidence = None
            for n in range(2):
                attempt = dict(attempt=n+1, started_at=now())
                t = time.monotonic()
                try:
                    raw = list(DDGS(timeout=12).text(query, max_results=3, backend='auto'))
                    docs, rejected = normalize_results(raw)
                    attempt.update(raw_results=raw, rejected_results=rejected, usable_documents=len(docs))
                    evidence = dict(status='ok' if docs else 'empty', backend='ddgs-auto', query=query,
                                    retrieved_at=now(), documents=docs, cache_hit=False)
                    if not docs:
                        evidence['error'] = 'No usable URL-and-text pairs returned'
                except Exception as exc:
                    attempt['error'] = f'{type(exc).__name__}: {exc}'
                    evidence = dict(status='error', backend='ddgs-auto', query=query, retrieved_at=now(),
                                    documents=[], error=attempt['error'], cache_hit=False)
                attempt['duration_seconds'] = round(time.monotonic()-t,3)
                attempts.append(attempt)
                if evidence['status'] == 'ok':
                    self.cache[key] = evidence
                    atomic_json(self.path, self.cache)
                    break
                if n == 0:
                    time.sleep(1)
        event.update(evidence=evidence, attempts=attempts, duration_seconds=round(time.monotonic()-started,3))
        append_jsonl(OUT / 'search_events.jsonl', event)
        return dict(evidence, supports_ground_truth=None)


def preflight(manifest):
    path = OUT / 'preflight.json'
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    done = {r['id'] for r in rows}
    search = StrictSearch()
    search.context = dict(phase='controller_preflight', model=None, turn=None)
    for item in manifest['items']:
        if item['id'] in done:
            continue
        text = ' '.join(item['question'].split())
        # No answers/options or evaluation annotations are included.
        query = text if len(text) <= 390 else text[:220] + ' ' + text[-165:]
        evidence = search(query,item)
        rows.append(dict(id=item['id'], dataset=item['dataset'], query_source='controller_question_text',
                         query=query, evidence=evidence))
        atomic_json(path, rows)
        print('PREFLIGHT',item['id'],evidence['status'],len(evidence['documents']),flush=True)
    return rows


class PilotGPU(LocalGPU):
    def __init__(self, name):
        super().__init__(name)
        self.context = {}
        self.action_ids = [self.tokenizer.encode(s,add_special_tokens=False) for s in ('1','2')]
        assert all(len(x) == 1 for x in self.action_ids)

    def handle_oom(self, operation):
        try:
            return operation()
        except self.torch.OutOfMemoryError as exc:
            event = dict(at=now(), **self.context, error=f'{type(exc).__name__}: {exc}',
                         recovery='retry same operation with SDPA attention; no context truncation or CPU offload')
            exc.__traceback__ = None
            append_jsonl(OUT / 'runtime_errors.jsonl',event)
        gc.collect()
        self.torch.cuda.empty_cache()
        self.model.set_attn_implementation('sdpa')
        self.metadata['attention'] = 'sdpa_after_eager_oom'
        return operation()

    def generate(self, messages, limit=160):
        result = self.handle_oom(lambda: super(PilotGPU,self).generate(messages,limit))
        event = dict(at=now(), kind='generation', **self.context, messages=messages,
                     messages_sha256=sha(messages), response=result)
        append_jsonl(OUT / f'{self.name}_model_calls.jsonl',event)
        return result

    def choose(self, messages):
        started = time.monotonic()
        def calculate():
            text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs = self.tokenizer(text,return_tensors='pt',add_special_tokens=False).to('cuda')
            with self.torch.inference_mode():
                logits = self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            if not self.torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite action logits')
            probs = logits[[x[0] for x in self.action_ids]].softmax(0).tolist()
            return dict(action='verify' if probs[0]>=probs[1] else 'answer',
                        label='1' if probs[0]>=probs[1] else '2', conditional_action_probabilities=probs,
                        input_tokens=inputs.input_ids.shape[1])
        result = self.handle_oom(calculate)
        result['duration_seconds'] = round(time.monotonic()-started,3)
        append_jsonl(OUT / f'{self.name}_model_calls.jsonl',dict(at=now(),kind='action_selection',
            **self.context,messages=messages,messages_sha256=sha(messages),response=result))
        return result


def score(response,item,runner):
    parsed = parse_reply(response['text'],item['dataset'],options=item.get('options'),
                         question=item['question'],hit_token_limit=response['hit_token_limit'])
    # Keep the full parser result without changing its implementation.
    if hasattr(parsed,'__dict__'):
        parsed = vars(parsed)
    if not isinstance(parsed,dict):
        parsed = dict(parser_result=str(parsed))
    return dict(original_protocol_evaluation=runner.classify(response['text'],item),
                provisional_parser_v4=parsed)


def run_model(name,manifest):
    runner = frozen_runner()
    path = OUT / f'{name}_checkpoint.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(model=name,dialogues=[])
    if saved.get('completed') == 20:
        return
    backend = PilotGPU(name)
    search = StrictSearch()
    try:
        by_id = {r['id']:r for r in saved['dialogues']}
        for item in manifest['items']:
            row = by_id.get(item['id'])
            if row is None:
                row = dict(id=item['id'],dataset=item['dataset'],tier='T2',policy='optional',
                           history=[dict(role='system',content=SYSTEM),dict(role='user',content=item['p0_prompt'])])
                saved['dialogues'].append(row)
                by_id[item['id']] = row
            for turn in ['p0','p1','p2']:
                if turn in row:
                    continue
                backend.context = dict(model=name,question_id=item['id'],dataset=item['dataset'],turn=turn)
                search.context = dict(phase='optional_model_tool_call',model=name,turn=turn)
                before = list(row['history'])
                if turn == 'p0':
                    response = backend.generate(before)
                    log, augmented = None,before
                else:
                    before += [dict(role='user',content=item[f'{turn}_prompt'])]
                    response,log,augmented = runner.challenged_turn(backend,before,'T2',item,search,'optional')
                row[turn] = dict(response=response,evaluation=score(response,item,runner),tool=log)
                if log and log['tool_invoked']:
                    observation = json.dumps({k:v for k,v in log['evidence'].items()
                                              if k not in ('supports_ground_truth','cache_hit')},ensure_ascii=False)
                    delivery = any(observation in m['content'] for m in augmented if m['role']=='user')
                    assert delivery, 'Tool observation was not passed into model input'
                    row[turn]['delivery_check'] = dict(exact_observation_in_generation_input=True,
                        observation_sha256=sha(observation),augmented_messages_sha256=sha(augmented),
                        document_count=len(log['evidence']['documents']),
                        nonempty_evidence_delivered=log['tool_success'])
                row['history'] = augmented + [dict(role='assistant',content=response['text'])]
                saved.update(metadata=backend.metadata,updated_at=now(),
                             completed=sum('p2' in r for r in saved['dialogues']),total=20)
                atomic_json(path,saved)
                print('TURN',name,item['id'],turn,'tokens',response['generated_tokens'],
                      'tool',bool(log and log['tool_invoked']),
                      'search',log['evidence']['status'] if log and log['evidence'] else '-',
                      'completed',saved['completed'],flush=True)
    finally:
        backend.close()


def summary():
    rows = []
    for name in ['qwen','llama','phi4']:
        path = OUT / f'{name}_checkpoint.json'
        if not path.exists(): continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for dataset in ['medqa','math']:
            selected = [r for r in data['dialogues'] if r['dataset']==dataset]
            for turn in ['p1','p2']:
                done = [r[turn] for r in selected if turn in r]
                calls = [r for r in done if r['tool'] and r['tool']['tool_invoked']]
                rows.append(dict(model=name,dataset=dataset,turn=turn,completed=len(done),
                    tool_calls=len(calls),search_ok=sum(r['tool']['tool_success'] for r in calls),
                    evidence_delivered=sum(r.get('delivery_check',{}).get('nonempty_evidence_delivered',False) for r in calls),
                    failures=sum(not r['tool']['tool_success'] for r in calls),
                    hit_token_limit=sum(r['response']['hit_token_limit'] for r in done)))
    atomic_json(OUT / 'summary.json',dict(at=now(),rows=rows))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','preflight','models','all','summary'],default='all')
    parser.add_argument('--models',nargs='+',default=['qwen','llama','phi4'])
    args = parser.parse_args()
    manifest = prepare()
    if args.phase in ['preflight','all']:
        preflight(manifest)
    if args.phase in ['models','all']:
        for name in args.models:
            run_model(name,manifest)
    summary()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        append_jsonl(OUT / 'runtime_errors.jsonl',dict(at=now(),error=traceback.format_exc()))
        raise
