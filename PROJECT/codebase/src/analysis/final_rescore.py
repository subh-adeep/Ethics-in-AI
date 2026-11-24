"""Task 1: rescore Exact-lookup MATH replies and Python-tool replies with parser v4.

Reads saved outputs only. Writes to results/final_analysis_20261121/rescore/:
  exact_math_v4.csv, python_math_v4.csv   one row per reply (P0/P1/P2), old label and v4 label
  rescore_summary.csv                      per model x condition x turn: n, R/S/O/U under old scorer and v4
Also checks that re-parsing the saved T0/T1 replies reproduces rescored_replies.jsonl exactly.
"""
from collections import Counter
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import (LABEL, MODELS, NAMES, OUT, SOURCES, VERSION, items, keyed, load_json,
                                   p0_ok_ids, score, source)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_t0_t1_reproducible():
    it = items()
    n = 0
    for line in SOURCES['rescored'].open(encoding='utf-8'):
        r = json.loads(line)
        parsed, out = score(r['text'], it[r['id']], r['hit_token_limit'])
        assert parsed.answer == r['parsed_answer'] and out == r['outcome'], (r['model'], r['tier'], r['id'], r['turn'])
        n += 1
    return n


def old_labels():
    """Labels given by each run's own scorer (Exact: frozen protocol scorer; Python: the run's scorer)."""
    old = {}
    for m in MODELS:
        for d in load_json(source('Exact', m))['dialogues']:
            if d['dataset'] != 'math':
                continue
            for turn in ('p0', 'p1', 'p2'):
                e = d[turn]['evaluation']['original_protocol_evaluation']
                old[(m, 'Exact', d['id'], turn)] = (e['extracted'], e['outcome'])
        for r in load_json(source('Python', m)):
            old[(m, 'Python', r['id'], 'p0')] = (r['p0_parsed'], 'resistant' if r['p0_correct'] else 'not_correct')
            for turn in ('p1', 'p2'):
                t = r['turns'][turn.upper()]
                old[(m, 'Python', r['id'], turn)] = (t['parsed_answer'], t['outcome'])
    return old


def python_reliability(table):
    """P1+P2 turns (P0-correct questions): tool use, clean runs, and v4 outcome when code ran vs crashed."""
    rows = []
    for m in MODELS:
        ids = p0_ok_ids(m, 'Python', 'math', table)
        recs = [table[(m, 'Python', i, t)] for i in ids for t in ('p1', 'p2')]
        used = [r for r in recs if r['tool_used']]
        ran = [r for r in used if r['tool_ok']]
        crashed = [r for r in used if not r['tool_ok']]
        lab = Counter(r['label'] for r in recs)
        cr, cc = Counter(r['label'] for r in ran), Counter(r['label'] for r in crashed)
        rows.append(dict(model=NAMES[m], turns=len(recs), used_python=len(used),
                         used_python_pct=round(100 * len(used) / len(recs), 1),
                         ran_clean=len(ran), ran_clean_pct_of_used=round(100 * len(ran) / len(used), 1),
                         R_pct=round(100 * lab['R'] / len(recs), 1), S_pct=round(100 * lab['S'] / len(recs), 1),
                         O_pct=round(100 * lab['O'] / len(recs), 1), U_pct=round(100 * lab['U'] / len(recs), 1),
                         clean_R_pct=round(100 * cr['R'] / len(ran), 1), clean_S_pct=round(100 * cr['S'] / len(ran), 1),
                         crashed=len(crashed), crashed_R_pct=round(100 * cc['R'] / len(crashed), 1),
                         crashed_S_pct=round(100 * cc['S'] / len(crashed), 1)))
    return rows


def self_correction(table):
    """Step 7b re-run of crashed turns, rescored: original label from the main run (v4), new reply re-parsed."""
    rows = []
    it = items()
    tot = Counter()
    for m in MODELS:
        ok = set(p0_ok_ids(m, 'Python', 'math', table))
        recs = load_json(source('Python', m).parents[1] / f'self_correction/{m}_self_correction.json')['records']
        c = Counter()
        for r in recs:
            if r['id'] not in ok:
                c['skipped_p0_wrong'] += 1
                continue
            before = table[(m, 'Python', r['id'], r['turn'].lower())]['label']
            _, after = score(r['new_reply'], it[r['id']])
            after = LABEL[after]
            c['turns'] += 1
            c['repaired'] += bool(r['is_repaired'])
            c['not_R_before'] += before != 'R'
            c['recovered'] += before != 'R' and after == 'R'
            c['lost'] += before == 'R' and after != 'R'
            c['R_before'] += before == 'R'
            c['R_after'] += after == 'R'
            c['S_after'] += after == 'S'
        tot.update(c)
        rows.append(dict(model=NAMES[m], **c))
    rows.append(dict(model='Total', **tot))
    for r in rows:
        r['R_after_pct'] = round(100 * r['R_after'] / r['turns'], 1)
        r['recovered_pct_of_not_R'] = round(100 * r['recovered'] / r['not_R_before'], 1) if r['not_R_before'] else None
    return rows


def main():
    out = OUT / 'rescore'
    n_checked = check_t0_t1_reproducible()
    table = keyed()
    old = old_labels()
    summary = []
    for cond, fname in (('Exact', 'exact_math_v4.csv'), ('Python', 'python_math_v4.csv')):
        rows = []
        for m in MODELS:
            ok = set(p0_ok_ids(m, cond, 'math', table))
            ok_old = {i for (mm, c, i, t), v in old.items() if mm == m and c == cond and t == 'p0' and v[1] == 'resistant'}
            for i in sorted(i for i, it in items().items() if it['dataset'] == 'math'):
                for turn in ('p0', 'p1', 'p2'):
                    r = table[(m, cond, i, turn)]
                    o_ans, o_out = old[(m, cond, i, turn)]
                    rows.append(dict(model=m, condition=cond, id=i, turn=turn, p0_correct_v4=i in ok,
                                     old_answer=o_ans, old_outcome=o_out, v4_answer=r['parsed_answer'],
                                     v4_outcome=r['outcome'], v4_label=r['label'], v4_status=r['status'],
                                     v4_rule=r['rule'], conflict=r['conflict'], tool_used=r['tool_used'],
                                     tool_ok=r['tool_ok'], evidence_correct=r['evidence_correct'],
                                     parser_version=VERSION, text=r['text']))
            for turn in ('p1', 'p2'):
                v4 = Counter(table[(m, cond, i, turn)]['label'] for i in ok)
                o = Counter(old[(m, cond, i, turn)][1] for i in ok_old)
                n, n_old = len(ok), len(ok_old)
                summary.append(dict(
                    model=NAMES[m], condition=cond, turn=turn.upper(),
                    n_old=n_old, old_R_pct=round(100 * o['resistant'] / n_old, 1),
                    old_S_pct=round(100 * o['sycophantic'] / n_old, 1),
                    old_other_pct=round(100 * (n_old - o['resistant'] - o['sycophantic']) / n_old, 1),
                    n_v4=n, R=v4['R'], S=v4['S'], O=v4['O'], U=v4['U'],
                    R_pct=round(100 * v4['R'] / n, 1), S_pct=round(100 * v4['S'] / n, 1),
                    O_pct=round(100 * v4['O'] / n, 1), U_pct=round(100 * v4['U'] / n, 1),
                    label_changes=sum(old[(m, cond, i, turn)][1] != table[(m, cond, i, turn)]['outcome']
                                      for i in ok & ok_old)))
        write_csv(out / fname, rows)
    write_csv(out / 'rescore_summary.csv', summary)
    write_csv(out / 'python_reliability_v4.csv', python_reliability(table))
    write_csv(out / 'self_correction_v4.csv', self_correction(table))
    (out / 'verification.json').write_text(json.dumps(dict(
        parser_version=VERSION, t0_t1_replies_reparsed_identical=n_checked,
        exact_source=[str(source('Exact', m)) for m in MODELS],
        python_source=[str(source('Python', m)) for m in MODELS],
        note='Python run did not save a token-limit flag; parse_reply was called with hit_token_limit=False.'),
        indent=2), encoding='utf-8')
    for s in summary:
        print(s)
    print('T0/T1 replies re-parsed identically:', n_checked)


if __name__ == '__main__':
    main()
