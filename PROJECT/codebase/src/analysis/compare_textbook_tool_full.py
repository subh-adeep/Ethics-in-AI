"""Read-only comparison of the full T3 textbook-tool run with T0/T1/T2/Exact on the same 146 MedQA IDs.

Every reply is scored twice: the frozen protocol scorer (as saved) and parser v4 (re-parsed from text).
Writes codebase/results/textbook_tool_full_20261110/comparison/{summary.csv,summary.json,paired_t0_t3.csv}.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import web_search_pilot as pilot
from analysis.rescore_saved_replies import outcome
from benchmark.answer_parser import parse_reply

RES = pilot.PROJECT / 'codebase/results'
T3 = RES / 'textbook_tool_full_20261110'
MODELS = ('qwen', 'llama', 'phi4')
NAMES = {'qwen': 'Qwen', 'llama': 'LLaMA', 'phi4': 'Phi-4'}
FROZEN = {'resistant': 'R', 'sycophantic': 'S'}
SOURCES = {
    'T2 web': RES / 'medqa_web_t2_20261019/kaggle_download_final_v2/medqa_project/PROJECT/codebase/results/medqa_web_t2_20261019/{m}_checkpoint.json',
    'Exact': RES / 'exact_lookup_full_20261020/{m}_checkpoint.json',
    'T3 textbook': T3 / 'kaggle_output/t3/PROJECT/codebase/results/textbook_tool_full_20261110/{m}/{m}_checkpoint.json',
}
CONDITIONS = ('T0', 'T1', 'T2 web', 'Exact', 'T3 textbook')


def main():
    items = {i['id']: i for i in json.loads((T3 / 'manifest.json').read_text(encoding='utf-8'))['items']}
    assert len(items) == 146
    rows = {}  # (model, cond, id, turn) -> dict(frozen, v4, tool)
    for line in (RES / 'parser_v4_20261018/rescored_replies.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        if r['id'] in items and r['domain'] == 'medqa':
            rows[(r['model'], r['tier'], r['id'], r['turn'])] = dict(frozen=r['original_outcome'], v4=r['outcome'], tool=None)
    for cond, pattern in SOURCES.items():
        for m in MODELS:
            path = Path(str(pattern).format(m=m))
            if cond == 'T3 textbook' and m == 'phi4':
                path = T3 / 'phi4/phi4_checkpoint.json'
            dialogues = [d for d in json.loads(path.read_text(encoding='utf-8'))['dialogues'] if d['id'] in items]
            assert {d['id'] for d in dialogues} == set(items), (cond, m)
            for d in dialogues:
                item = items[d['id']]
                for turn in ('p0', 'p1', 'p2'):
                    res = d[turn]
                    resp = res['response']
                    parsed = parse_reply(resp['text'], 'medqa', options=item['options'], question=item['question'],
                                         hit_token_limit=resp['hit_token_limit'])
                    tool = res.get('tool')
                    rows[(m, cond, d['id'], turn)] = dict(
                        frozen=res['evaluation']['original_protocol_evaluation']['outcome'],
                        v4=outcome(parsed.answer, item),
                        tool=bool(tool and tool['tool_invoked']) if turn != 'p0' else None)
    summary = []
    for m in MODELS:
        for cond in CONDITIONS:
            rec = dict(model=NAMES[m], condition=cond)
            for turn in ('p0', 'p1', 'p2'):
                sel = [rows[(m, cond, i, turn)] for i in items]
                for scorer in ('frozen', 'v4'):
                    c = Counter(FROZEN.get(r[scorer], 'other') for r in sel)
                    rec[f'{turn}_{scorer}'] = f"{c['R']}/{c['S']}/{c['other']}"
                if turn != 'p0' and cond not in ('T0', 'T1'):
                    rec[f'{turn}_tool'] = sum(r['tool'] for r in sel)
            summary.append(rec)
    out = T3 / 'comparison'
    out.mkdir(exist_ok=True)
    fields = ['model', 'condition'] + [f'{t}_{s}' for t in ('p0', 'p1', 'p2') for s in ('frozen', 'v4')] + ['p1_tool', 'p2_tool']
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # T3 split by whether the tool was actually used (v4 outcomes).
    split = []
    for m in MODELS:
        for turn in ('p1', 'p2'):
            for used in (True, False):
                ids = [i for i in items if rows[(m, 'T3 textbook', i, turn)]['tool'] is used]
                c3 = Counter(FROZEN.get(rows[(m, 'T3 textbook', i, turn)]['v4'], 'other') for i in ids)
                c0 = Counter(FROZEN.get(rows[(m, 'T0', i, turn)]['v4'], 'other') for i in ids)
                split.append(dict(model=NAMES[m], turn=turn, tool_used=used, n=len(ids),
                                  t3_v4=f"{c3['R']}/{c3['S']}/{c3['other']}", t0_v4_same_ids=f"{c0['R']}/{c0['S']}/{c0['other']}"))
    (out / 'tool_split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    # Paired T0 -> T3 transitions at P2 (v4): resistant in one, sycophantic in the other.
    paired = []
    for m in MODELS:
        for turn in ('p1', 'p2'):
            t = Counter((rows[(m, 'T0', i, turn)]['v4'], rows[(m, 'T3 textbook', i, turn)]['v4']) for i in items)
            paired.append(dict(model=NAMES[m], turn=turn,
                               t0R_t3S=t[('resistant', 'sycophantic')], t0S_t3R=t[('sycophantic', 'resistant')],
                               both_R=t[('resistant', 'resistant')], both_S=t[('sycophantic', 'sycophantic')]))
    (out / 'paired_t0_t3.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
    for rec in summary:
        print(rec)
    for r in split:
        print(r)
    for r in paired:
        print(r)


if __name__ == '__main__':
    main()
