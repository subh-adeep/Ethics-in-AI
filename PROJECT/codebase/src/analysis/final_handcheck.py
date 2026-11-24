"""Task 4a: draw a stratified sample of 100 P1/P2 replies for a human check of parser v4.

Strata = dataset x model x condition x turn (MedQA 5 conditions, MATH 4 -> 54 strata).
Every stratum gets 1 reply; 46 strata (chosen at random) get a second one. Only questions correct at
P0 in that run are eligible. Seed 20261121.

Writes results/final_analysis_20261121/handcheck/:
  handcheck_sheet.csv  what the reviewer sees and fills (column human_answer; human_note optional)
  handcheck_key.csv    correct answer, user's wrong answer and the v4 label (used by the agreement script;
                       the reviewer should not open it before finishing)
The sheet is never overwritten if it already exists (so filled answers are safe); pass --force to redo.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import CONDITIONS, DATASETS, DNAMES, MODELS, NAMES, OUT, items, keyed, p0_ok_ids

SEED = 20261121
TOTAL = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out = OUT / 'handcheck'
    sheet, key = out / 'handcheck_sheet.csv', out / 'handcheck_key.csv'
    if sheet.exists() and not args.force:
        sys.exit(f'{sheet} already exists; not overwriting (use --force).')
    table = keyed()
    it = items()
    rng = random.Random(SEED)
    strata = [(d, m, c, t) for d in DATASETS for m in MODELS for c in CONDITIONS[d] for t in ('p1', 'p2')]
    extra = set(rng.sample(range(len(strata)), TOTAL - len(strata)))
    picks = []
    for k, (d, m, c, t) in enumerate(strata):
        ids = p0_ok_ids(m, c, d, table)
        for i in rng.sample(ids, 2 if k in extra else 1):
            picks.append(table[(m, c, i, t)])
    assert len(picks) == TOTAL
    rng.shuffle(picks)  # reviewer should not see the strata in order
    sheet_rows, key_rows = [], []
    for n, r in enumerate(picks, 1):
        item = it[r['id']]
        cid = f'HC{n:03d}'
        options = ' | '.join(f'{k}: {v}' for k, v in item['options'].items()) if item['dataset'] == 'medqa' else ''
        sheet_rows.append(dict(check_id=cid, dataset=DNAMES[item['dataset']], question_id=item['id'],
                               question=item['question'], options=options,
                               user_message=item[f"{r['turn']}_prompt"], reply_text=r['text'],
                               parser_v4_answer='NONE' if r['parsed_answer'] is None else r['parsed_answer'],
                               human_answer='', human_note=''))
        key_rows.append(dict(check_id=cid, model=NAMES[r['model']], dataset=DNAMES[item['dataset']],
                             condition=r['condition'], turn=r['turn'].upper(), question_id=item['id'],
                             ground_truth=item['ground_truth'],
                             user_wrong_answer=item['false_answer_key'] if item['dataset'] == 'medqa' else item['false_answer'],
                             parser_v4_answer='NONE' if r['parsed_answer'] is None else r['parsed_answer'],
                             parser_v4_label=r['label'], parser_v4_rule=r['rule'] or '', hit_token_limit=r['hit_token_limit']))
    out.mkdir(parents=True, exist_ok=True)
    for path, rows in ((sheet, sheet_rows), (key, key_rows)):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    print('sampled', len(picks), Counter((DNAMES[r['dataset']], r['label']) for r in picks))


if __name__ == '__main__':
    main()
