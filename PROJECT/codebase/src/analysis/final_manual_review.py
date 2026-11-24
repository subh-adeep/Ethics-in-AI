"""Manual review of unparsed (U) replies: apply the hand decisions and rebuild the tables.

Input (hand-made, not generated): results/final_analysis_20261121/manual_unparsed_review/manual_decisions.csv
  one row per unique (question, reply text) among the P1/P2 replies labelled U by parser v4 (questions correct at P0).
  The answer was read by hand from the reply, without looking at the correct or the user's wrong answer.
  Recovered answers carry an exact quote from the reply (checked here).

Outputs (all new, same folder):
  unparsed_review.csv          one row per reviewed reply (365), with the new label
  metrics_table_manual.csv/.md  same layout as metrics/metrics_table.csv, U replaced where an answer was recovered,
                                plus the v4 counts next to the manual counts
  significance_manual.csv/.md   the 42 paired tests rerun on the manual labels, with the v4 result alongside
Existing files are never written.
"""
from collections import Counter
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import (CONDITIONS, DATASETS, DNAMES, LABEL, MODELS, NAMES, OUT, items, keyed, md_table,
                                   p0_ok_ids)
from analysis.rescore_saved_replies import outcome
from analysis import final_metrics_table, final_significance

FOLDER = OUT / 'manual_unparsed_review'


def sha1(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def manual_table():
    """Copy of the v4 table with reviewed U replies relabelled. Returns (table, review_rows)."""
    v4 = keyed()
    it = items()
    decisions = {(d['question_id'], d['text_sha1']): d for d in read(FOLDER / 'manual_decisions.csv')}
    table = {k: dict(v) for k, v in v4.items()}
    review, used = [], set()
    for d in DATASETS:
        for m in MODELS:
            for c in CONDITIONS[d]:
                for i in p0_ok_ids(m, c, d, v4):
                    for turn in ('p1', 'p2'):
                        r = v4[(m, c, i, turn)]
                        if r['label'] != 'U':
                            continue
                        key = (i, sha1(r['text']))
                        dec = decisions[key]
                        used.add(key)
                        ans = dec['extracted_answer'] or None
                        if ans:
                            assert dec['evidence_quote'] in r['text'], (m, c, i, turn)
                        new_out = outcome(ans, it[i])
                        table[(m, c, i, turn)].update(outcome=new_out, label=LABEL[new_out], parsed_answer=ans)
                        review.append(dict(dataset=DNAMES[d], model=NAMES[m], condition=c, turn=turn.upper(),
                                           question_id=i, review_uid=dec['review_uid'], reply_text=r['text'],
                                           v4_status=r['status'], extracted_answer=ans or '',
                                           evidence_quote=dec['evidence_quote'], decision=dec['decision'],
                                           reason=dec['reason'], stance=dec['stance'],
                                           old_label='U', new_label=LABEL[new_out]))
    assert used == set(decisions), 'decisions not matched to any reply'
    return table, review


def main():
    table, review = manual_table()
    write_csv(FOLDER / 'unparsed_review.csv', review)

    old = final_metrics_table.build_rows(keyed())
    new = final_metrics_table.build_rows(table)
    rows = []
    for o, n in zip(old, new):
        assert (o['dataset'], o['model'], o['condition'], o['turn']) == (n['dataset'], n['model'], n['condition'], n['turn'])
        row = dict(n)
        for k in ('resistant', 'flip', 'other_wrong', 'unparsed'):
            row[f'{k}_n_v4'] = o[f'{k}_n']
        row['recovered_n'] = o['unparsed_n'] - n['unparsed_n']
        row['evidence_override_n_v4'] = o['evidence_override_n']
        rows.append(row)
    write_csv(FOLDER / 'metrics_table_manual.csv', rows)

    def pair(newv, oldv):
        return f'{newv}' if newv == oldv else f'**{newv}** ({oldv})'

    lines = ['# Metrics table after manual review of unparsed replies', '',
             'Same layout as `metrics/metrics_table.md`. Where a reviewer found a clear answer in an unparsed reply, '
             'the U label was replaced by R / S / O. A changed cell is **bold** with the parser v4 value in brackets. '
             'n is unchanged (no P0 reply changed).', '']
    for d in DATASETS:
        lines += [f'## {DNAMES[d]}', '']
        body = []
        for r, o in zip(rows, old):
            if r['dataset'] != DNAMES[d]:
                continue
            body.append([r['model'], r['condition'], r['turn'], r['n'], pair(r['resistant_pct'], o['resistant_pct']),
                         pair(r['flip_pct'], o['flip_pct']), pair(r['other_wrong_pct'], o['other_wrong_pct']),
                         pair(r['unparsed_pct'], o['unparsed_pct']), r['recovered_n'],
                         '–' if r['tool_use_pct'] is None else r['tool_use_pct'],
                         '–' if r['evidence_override_n'] is None else
                         pair(f"{r['evidence_override_n']}/{r['evidence_correct_n']}",
                              f"{o['evidence_override_n']}/{o['evidence_correct_n']}")])
        lines += [md_table(['Model', 'Condition', 'Turn', 'n', 'R %', 'Flip %', 'Other %', 'U %', 'Recovered',
                            'Tool use %', 'Override'], body), '']
    (FOLDER / 'metrics_table_manual.md').write_text('\n'.join(lines), encoding='utf-8')

    s_old = final_significance.build_rows(keyed())
    s_new = final_significance.build_rows(table)
    srows = []
    for o, n in zip(s_old, s_new):
        row = dict(n)
        for k in ('T0_R_pct', 'tool_R_pct', 'diff_R_pp', 'improved', 'worsened', 'mcnemar_exact_p', 'holm_p', 'verdict'):
            row[f'{k}_v4'] = o[k]
        row['verdict_changed'] = o['verdict'] != n['verdict']
        srows.append(row)
    write_csv(FOLDER / 'significance_manual.csv', srows)
    fp = final_significance.fmt_p
    lines = ['# Significance tests after manual review of unparsed replies', '',
             'Same method as `significance/significance.md` (paired, exact McNemar, Holm over 42 tests, bootstrap 95% CI, '
             '5,000 resamples, seed 20261014), run on the manual labels. The v4 result is shown for comparison. '
             '"Changed" marks a test whose verdict is different.', '']
    for d in DATASETS:
        lines += [f'## {DNAMES[d]}', '']
        body = [[r['model'], r['condition'], r['turn'], r['n_paired'], f"{r['T0_R_pct']} ({r['T0_R_pct_v4']})",
                 f"{r['tool_R_pct']} ({r['tool_R_pct_v4']})",
                 f"{r['diff_R_pp']:+.1f} [{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]",
                 f"{r['improved']} / {r['worsened']}", fp(r['mcnemar_exact_p']), fp(r['holm_p']), r['verdict'],
                 r['verdict_v4'], 'yes' if r['verdict_changed'] else '']
                for r in srows if r['dataset'] == DNAMES[d]]
        lines += [md_table(['Model', 'Tool', 'Turn', 'n', 'T0 R % (v4)', 'Tool R % (v4)', 'Diff pp [95% CI]',
                            'Improved / worsened', 'p', 'Holm p', 'Verdict', 'Verdict v4', 'Changed'], body), '']
    (FOLDER / 'significance_manual.md').write_text('\n'.join(lines), encoding='utf-8')

    # console summary used for the README
    print('reviewed replies:', len(review), Counter(r['decision'] for r in review))
    print('new labels of recovered:', Counter(r['new_label'] for r in review if r['decision'] == 'recovered'))
    print('by reason:', Counter(r['reason'] for r in review))
    print('recovered per table:')
    for (ds, cond), c in sorted(Counter((r['dataset'], r['condition']) for r in review).items()):
        rec = [r for r in review if (r['dataset'], r['condition']) == (ds, cond) and r['decision'] == 'recovered']
        print(f'  {ds} {cond}: {len(rec)}/{c}', Counter(r['new_label'] for r in rec))
    print('changed verdicts:', [(r['dataset'], r['model'], r['condition'], r['turn'], r['verdict_v4'], '->', r['verdict'])
                                for r in srows if r['verdict_changed']])


if __name__ == '__main__':
    main()
