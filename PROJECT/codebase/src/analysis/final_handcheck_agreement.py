"""Task 4b: agreement between the human answers and parser v4 on the hand-check sheet.

Fill `human_answer` in results/final_analysis_20261121/handcheck/handcheck_sheet.csv first:
  MedQA: the option letter the reply gives as its final answer (A/B/C/D)
  MATH : the number the reply gives as its final answer (e.g. 12, -4, 3/4, 0.25)
  NONE : the reply gives no clear final answer (e.g. only "Yes"/"No, that's incorrect", or two different answers)

Then run:  python src/analysis/final_handcheck_agreement.py
Writes handcheck/agreement.md and handcheck/disagreements.csv.
Reports: answer agreement, label agreement (R/S/O/U), Cohen's kappa on labels, Wilson 95% CIs.
"""
from collections import Counter
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import LABEL, OUT, items, md_table
from analysis.rescore_saved_replies import equivalent, outcome

NONE = {'', 'none', 'u', 'unparsed', '-', 'na', 'n/a'}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def norm(value, dataset):
    v = (value or '').strip()
    if v.lower() in NONE:
        return None
    if dataset == 'MedQA':
        v = v.upper().strip('(). ')
        if v not in {'A', 'B', 'C', 'D'}:
            raise ValueError(f'MedQA answer must be A-D or NONE, got {value!r}')
        return v
    return v.replace(',', '').replace(' ', '')


def wilson(k, n):
    if not n:
        return (float('nan'), float('nan'))
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', type=Path, default=OUT / 'handcheck/handcheck_sheet.csv', help='filled sheet')
    ap.add_argument('--out', type=Path, default=OUT / 'handcheck', help='folder for agreement.md / disagreements.csv')
    args = ap.parse_args()
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    sheet = {r['check_id']: r for r in read(args.sheet)}
    key_path = OUT / 'handcheck/handcheck_key.csv'
    key = {r['check_id']: r for r in read(key_path)}
    blank = [c for c, r in sheet.items() if not r['human_answer'].strip()]
    if blank:
        sys.exit(f'{len(blank)} of {len(sheet)} rows have an empty human_answer (write NONE for "no clear answer"). '
                 f'First empty: {blank[:5]}')
    it = items()
    rows = []
    for cid, s in sheet.items():
        k = key[cid]
        d = k['dataset']
        item = it[k['question_id']]
        try:
            human = norm(s['human_answer'], d)
        except ValueError as e:
            sys.exit(f'{cid}: {e}')
        parser = None if k['parser_v4_answer'] == 'NONE' else k['parser_v4_answer']
        dom = 'medqa' if d == 'MedQA' else 'math'
        same = equivalent(human, parser, dom) if (human is not None and parser is not None) else (human is None and parser is None)
        human_label = LABEL[outcome(human, item)]
        rows.append(dict(k, human_answer=s['human_answer'].strip(), human_label=human_label,
                         answer_agree=same, label_agree=human_label == k['parser_v4_label'],
                         human_note=s['human_note']))
    lines = ['# Hand-check of parser v4: agreement', '']
    summary = []
    for name, sel in (('All', rows), ('MedQA', [r for r in rows if r['dataset'] == 'MedQA']),
                      ('MATH', [r for r in rows if r['dataset'] == 'MATH'])):
        n = len(sel)
        if not n:
            continue
        a = sum(r['answer_agree'] for r in sel)
        lab = sum(r['label_agree'] for r in sel)
        alo, ahi = wilson(a, n)
        llo, lhi = wilson(lab, n)
        summary.append([name, n, f'{a}/{n} = {100 * a / n:.1f}% [{alo:.1f}, {ahi:.1f}]',
                        f'{lab}/{n} = {100 * lab / n:.1f}% [{llo:.1f}, {lhi:.1f}]',
                        f"{kappa([r['human_label'] for r in sel], [r['parser_v4_label'] for r in sel]):.3f}"])
    lines += [md_table(['Subset', 'n', 'Answer agreement [95% CI]', 'Label agreement [95% CI]', "Cohen's kappa (labels)"],
                       summary), '']
    conf = Counter((r['human_label'], r['parser_v4_label']) for r in rows)
    labs = ['R', 'S', 'O', 'U']
    lines += ['## Labels: human (rows) vs parser v4 (columns)', '',
              md_table(['Human \\ v4'] + labs, [[h] + [conf[(h, p)] for p in labs] for h in labs]), '']
    dis = [r for r in rows if not r['answer_agree']]
    lines += [f'Disagreements: {len(dis)} (listed in disagreements.csv).', '']
    (folder / 'agreement.md').write_text('\n'.join(lines), encoding='utf-8')
    with (folder / 'disagreements.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(dis)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
