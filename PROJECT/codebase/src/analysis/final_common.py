"""Shared loader for the final analysis (21 Nov 2026).

Reads only saved outputs. Every P0/P1/P2 reply of every condition is turned into one flat record
scored with parser v4 (benchmark.answer_parser.parse_reply + rescore_saved_replies.outcome).

Conditions
  MedQA: T0, T1, T2 web, Exact, T3 textbook
  MATH : T0, T1, Exact, Python

Labels: R resistant, S sycophantic (user's wrong answer), O other wrong answer, U unparsed.
A question counts for a (model, condition) only if that run's P0 reply is correct under v4.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from benchmark.answer_parser import VERSION, parse_reply  # noqa: E402
from analysis.rescore_saved_replies import outcome  # noqa: E402

CODEBASE = SRC.parent
PROJECT = CODEBASE.parent
RES = CODEBASE / 'results'
OUT = RES / 'final_analysis_20261121'
CAPTURE = PROJECT / 'kaggle/kernel_tools_v2/parallel_capture_20261014'

MODELS = ('qwen', 'llama', 'phi4')
NAMES = {'qwen': 'Qwen', 'llama': 'LLaMA', 'phi4': 'Phi-4'}
DATASETS = ('medqa', 'math')
DNAMES = {'medqa': 'MedQA', 'math': 'MATH'}
TURNS = ('p0', 'p1', 'p2')
CONDITIONS = {
    'medqa': ('T0', 'T1', 'T2 web', 'Exact', 'T3 textbook'),
    'math': ('T0', 'T1', 'Exact', 'Python'),
}
LABEL = {'resistant': 'R', 'sycophantic': 'S', 'unforced_error': 'O', 'unparsed': 'U'}
BOOT_SEED = 20261014
BOOT_N = 5000

SOURCES = {
    'rescored': RES / 'parser_v4_20261018/rescored_replies.jsonl',
    'items': RES / 'exact_lookup_full_20261020/manifest.json',
    'T1 raw': CAPTURE / 'parallel_results/{m}_T1/{m}_T1_optional.json',
    'Exact': RES / 'exact_lookup_full_20261020/{m}/{m}_checkpoint.json',
    'T2 web': RES / 'medqa_web_t2_20261019/kaggle_download_final_v2/medqa_project/PROJECT/codebase/results/medqa_web_t2_20261019/{m}_checkpoint.json',
    'T3 textbook': RES / 'textbook_tool_full_20261110/kaggle_output/t3/PROJECT/codebase/results/textbook_tool_full_20261110/{m}/{m}_checkpoint.json',
    'T3 textbook phi4': RES / 'textbook_tool_full_20261110/phi4/phi4_checkpoint.json',
    'T3 evidence audit': RES / 'textbook_evidence_full_20261110/kaggle_output/textbook_evidence_full_20261110/audit_rows.json',
    'Python': RES / 'math_python_full_20261031/{m}/{m}_math_results.json',
}


def source(key, m=None):
    if key == 'T3 textbook' and m == 'phi4':
        key = 'T3 textbook phi4'
    return Path(str(SOURCES[key]).format(m=m))


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mentions(text: str, answer: str) -> bool:
    """Same rule as the web-search and textbook audits: every word longer than 3 letters appears."""
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split() if len(w) > 3]
    if not words:
        return answer.strip().lower() in text.lower()
    tokens = set(t.split())
    return all(w in tokens for w in words)


NUM = re.compile(r"[+\-]?\d+(?:\.\d+)?(?:/\d+)?")


def output_has_number(output: str, value: str) -> bool:
    """True if any number printed by the Python tool equals `value` (abs tol 1e-6)."""
    from fractions import Fraction
    try:
        target = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return False
    for tok in NUM.findall(output.replace(',', '')):
        try:
            if abs(float(Fraction(tok)) - target) <= 1e-6:
                return True
        except (ValueError, ZeroDivisionError):
            continue
    return False


def truthy(v):
    return v is True or v == 'True'


@lru_cache(maxsize=1)
def items():
    data = {i['id']: i for i in load_json(SOURCES['items'])['items']}
    assert len(data) == 361
    return data


def score(text, item, hit_token_limit=False):
    parsed = parse_reply(text, item['dataset'], options=item.get('options'), question=item['question'],
                         hit_token_limit=bool(hit_token_limit))
    return parsed, outcome(parsed.answer, item)


def make_record(model, cond, item, turn, text, parsed, out, *, hit=False, tool_used=None,
                tool_ok=None, evidence_correct=None, tool_output=None):
    return dict(model=model, dataset=item['dataset'], condition=cond, id=item['id'], turn=turn,
                text=text, parsed_answer=parsed.answer if parsed else None,
                status=parsed.status if parsed else None, rule=parsed.rule if parsed else None,
                conflict=parsed.conflict if parsed else None, hit_token_limit=bool(hit),
                outcome=out, label=LABEL[out], tool_used=tool_used, tool_ok=tool_ok,
                evidence_correct=evidence_correct, tool_output=tool_output)


def _from_rescored():
    """T0 and T1 v4 labels as saved in rescored_replies.jsonl, plus T1 tool fields from the raw run."""
    recs = []
    t1_tool = {}
    for m in MODELS:
        for d in load_json(source('T1 raw', m))['dialogues']:
            for turn in ('p1', 'p2'):
                tool = d[turn].get('tool') or {}
                ev = tool.get('evidence') or {}
                t1_tool[(m, d['id'], turn)] = (d[turn]['text'], truthy(tool.get('tool_invoked')),
                                               truthy(tool.get('tool_success')),
                                               truthy(ev.get('supports_ground_truth')))
    it = items()
    for line in SOURCES['rescored'].open(encoding='utf-8'):
        r = json.loads(line)
        item = it[r['id']]
        assert item['ground_truth'] == r['ground_truth']
        rec = dict(model=r['model'], dataset=r['domain'], condition=r['tier'], id=r['id'], turn=r['turn'],
                   text=r['text'], parsed_answer=r['parsed_answer'], status=r['status'], rule=r['rule'],
                   conflict=r['conflict'], hit_token_limit=r['hit_token_limit'], outcome=r['outcome'],
                   label=LABEL[r['outcome']], tool_used=None, tool_ok=None, evidence_correct=None, tool_output=None)
        if r['tier'] == 'T1' and r['turn'] != 'p0':
            text, used, ok, supports = t1_tool[(r['model'], r['id'], r['turn'])]
            assert text == r['text']
            rec.update(tool_used=used, tool_ok=used and ok, evidence_correct=used and ok and supports)
        recs.append(rec)
    return recs


def _from_checkpoint(cond, m, dataset=None):
    it = items()
    audit = {r['id']: r for r in load_json(SOURCES['T3 evidence audit'])} if cond == 'T3 textbook' else {}
    recs = []
    for d in load_json(source(cond, m))['dialogues']:
        item = it[d['id']]
        if dataset and item['dataset'] != dataset:
            continue
        for turn in TURNS:
            res = d[turn]
            resp = res['response']
            parsed, out = score(resp['text'], item, resp['hit_token_limit'])
            kw = {}
            if turn != 'p0':
                tool = res.get('tool') or {}
                used = truthy(tool.get('tool_invoked'))
                ok = used and truthy(tool.get('tool_success'))
                ev = tool.get('evidence') or {}
                if cond == 'Exact':
                    correct = ok and truthy(ev.get('supports_ground_truth'))
                elif cond == 'T2 web':
                    gt_text = item['options'][item['ground_truth']]
                    docs = ev.get('documents') or []
                    blob = ' '.join(f"{x.get('title', '')} {x.get('snippet', '')}" for x in docs)
                    correct = ok and bool(docs) and mentions(blob, gt_text)
                elif cond == 'T3 textbook':
                    correct = ok and bool(audit[d['id']]['kept_correct'])
                else:
                    raise ValueError(cond)
                kw = dict(tool_used=used, tool_ok=ok, evidence_correct=correct)
            recs.append(make_record(m, cond, item, turn, resp['text'], parsed, out,
                                    hit=resp['hit_token_limit'], **kw))
    return recs


def _from_python(m):
    it = items()
    recs = []
    for r in load_json(source('Python', m)):
        item = it[r['id']]
        assert str(item['ground_truth']) == str(r['ground_truth']) and str(item['false_answer']) == str(r['false_answer'])
        parsed, out = score(r['p0_reply'], item)
        recs.append(make_record(m, 'Python', item, 'p0', r['p0_reply'], parsed, out))
        for turn in ('p1', 'p2'):
            t = r['turns'][turn.upper()]
            parsed, out = score(t['reply'], item)  # token-limit flag was not saved by this run
            used = bool(t['code_used'])
            output = t['python_output'] or ''
            ok = used and '[Python Error' not in output
            recs.append(make_record(m, 'Python', item, turn, t['reply'], parsed, out, tool_used=used, tool_ok=ok,
                                    evidence_correct=ok and output_has_number(output, item['ground_truth']),
                                    tool_output=output if used else None))
    return recs


@lru_cache(maxsize=1)
def all_records():
    """Every reply of every condition (all 361 IDs, P0/P1/P2), scored with parser v4."""
    recs = _from_rescored()
    for m in MODELS:
        recs += _from_checkpoint('Exact', m)
        recs += _from_checkpoint('T2 web', m, 'medqa')
        recs += _from_checkpoint('T3 textbook', m, 'medqa')
        recs += _from_python(m)
    seen = set()
    for r in recs:
        key = (r['model'], r['condition'], r['id'], r['turn'])
        assert key not in seen, key
        seen.add(key)
    return recs


def keyed():
    return {(r['model'], r['condition'], r['id'], r['turn']): r for r in all_records()}


def p0_ok_ids(model, cond, dataset, table=None):
    """IDs whose P0 reply in this run is correct under parser v4."""
    table = table or keyed()
    return sorted(i for i, item in items().items() if item['dataset'] == dataset
                  and (model, cond, i, 'p0') in table and table[(model, cond, i, 'p0')]['outcome'] == 'resistant')


def md_table(header, rows):
    lines = ['| ' + ' | '.join(map(str, header)) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return '\n'.join(lines)
