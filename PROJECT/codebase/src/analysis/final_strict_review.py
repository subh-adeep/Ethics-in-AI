"""Strict re-check of the answers recovered in the manual review of unparsed replies.

Inputs (hand-made):
  manual_unparsed_review/manual_decisions.csv          first manual review (not modified here)
  manual_unparsed_review_strict/strict_decisions.csv   keep / back_to_U for every recovered reply text
Outputs (all new, in manual_unparsed_review_strict/):
  strict_recheck.csv              one row per recovered reply (109): quote, first decision, strict decision, reason
  metrics_table_strict.csv/.md    metrics table with strict labels, plus v4 and manual values
  significance_strict.csv/.md     the 42 paired tests on strict labels, plus v4 and manual verdicts
"""
from collections import Counter
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import CONDITIONS, DATASETS, DNAMES, LABEL, MODELS, NAMES, OUT, items, keyed, md_table, p0_ok_ids
from analysis.rescore_saved_replies import outcome
from analysis import final_metrics_table, final_significance

MANUAL = OUT / 'manual_unparsed_review'
FOLDER = OUT / 'manual_unparsed_review_strict'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def relabel(decisions):
    """v4 table with U replies relabelled from {(question_id, sha1): answer or None}."""
    v4 = keyed()
    it = items()
    table = {k: dict(v) for k, v in v4.items()}
    hits = []
    for d in DATASETS:
        for m in MODELS:
            for c in CONDITIONS[d]:
                for i in p0_ok_ids(m, c, d, v4):
                    for turn in ('p1', 'p2'):
                        r = v4[(m, c, i, turn)]
                        if r['label'] != 'U':
                            continue
                        ans = decisions[(i, hashlib.sha1(r['text'].encode('utf-8')).hexdigest())]
                        new = outcome(ans, it[i])
                        table[(m, c, i, turn)].update(outcome=new, label=LABEL[new], parsed_answer=ans)
                        hits.append((m, c, i, turn, r['text']))
    return table, hits


def main():
    manual = {(d['question_id'], d['text_sha1']): d for d in read(MANUAL / 'manual_decisions.csv')}
    strict = {(d['question_id'], d['text_sha1']): d for d in read(FOLDER / 'strict_decisions.csv')}
    assert set(strict) == {k for k, d in manual.items() if d['decision'] == 'recovered'}
    man_ans = {k: d['extracted_answer'] or None for k, d in manual.items()}
    str_ans = {k: (None if k in strict and strict[k]['strict_decision'] == 'back_to_U' else a) for k, a in man_ans.items()}
    v4 = keyed()
    t_man, _ = relabel(man_ans)
    t_str, hits = relabel(str_ans)
    it = items()

    rows = []
    for m, c, i, turn, text in hits:
        key = (i, hashlib.sha1(text.encode('utf-8')).hexdigest())
        if key not in strict:
            continue
        s, d = strict[key], manual[key]
        rows.append(dict(dataset=DNAMES[it[i]['dataset']], model=NAMES[m], condition=c, turn=turn.upper(),
                         question_id=i, review_uid=d['review_uid'], reply_text=text,
                         evidence_quote=d['evidence_quote'], first_answer=d['extracted_answer'],
                         first_label=t_man[(m, c, i, turn)]['label'], first_reason=d['reason'],
                         strict_decision=s['strict_decision'], strict_reason=s['strict_reason'],
                         strict_label=t_str[(m, c, i, turn)]['label']))
    assert len(rows) == 109, len(rows)
    write_csv(FOLDER / 'strict_recheck.csv', rows)

    m_v4, m_man, m_str = (final_metrics_table.build_rows(t) for t in (v4, t_man, t_str))
    out = []
    for a, b, c in zip(m_v4, m_man, m_str):
        row = dict(c)
        for k in ('resistant', 'flip', 'other_wrong', 'unparsed'):
            row[f'{k}_n_v4'] = a[f'{k}_n']
            row[f'{k}_n_manual'] = b[f'{k}_n']
        row['evidence_override_n_v4'] = a['evidence_override_n']
        row['evidence_override_n_manual'] = b['evidence_override_n']
        out.append(row)
    write_csv(FOLDER / 'metrics_table_strict.csv', out)

    def cell(s, mn, v):
        if s == mn == v:
            return f'{s}'
        return f'**{s}** ({mn} / {v})'

    lines = ['# Metrics table — strict re-check of recovered answers', '',
             'Same layout as `metrics/metrics_table.md`. Labels: parser v4, plus answers from the manual review that '
             'pass the strict rule. A cell that differs from either older version is **bold**, with '
             '(manual / v4) in brackets. n is unchanged.', '']
    for ds in DATASETS:
        lines += [f'## {DNAMES[ds]}', '']
        body = [[c['model'], c['condition'], c['turn'], c['n'],
                 cell(c['resistant_pct'], b['resistant_pct'], a['resistant_pct']),
                 cell(c['flip_pct'], b['flip_pct'], a['flip_pct']),
                 cell(c['other_wrong_pct'], b['other_wrong_pct'], a['other_wrong_pct']),
                 cell(c['unparsed_pct'], b['unparsed_pct'], a['unparsed_pct']),
                 '–' if c['tool_use_pct'] is None else c['tool_use_pct'],
                 '–' if c['evidence_override_n'] is None else
                 cell(f"{c['evidence_override_n']}/{c['evidence_correct_n']}",
                      f"{b['evidence_override_n']}/{b['evidence_correct_n']}",
                      f"{a['evidence_override_n']}/{a['evidence_correct_n']}")]
                for a, b, c in zip(m_v4, m_man, m_str) if c['dataset'] == DNAMES[ds]]
        lines += [md_table(['Model', 'Condition', 'Turn', 'n', 'R %', 'Flip %', 'Other %', 'U %', 'Tool use %', 'Override'],
                           body), '']
    (FOLDER / 'metrics_table_strict.md').write_text('\n'.join(lines), encoding='utf-8')

    s_v4, s_man, s_str = (final_significance.build_rows(t) for t in (v4, t_man, t_str))
    srows = []
    for a, b, c in zip(s_v4, s_man, s_str):
        row = dict(c)
        for tag, src in (('v4', a), ('manual', b)):
            for k in ('diff_R_pp', 'mcnemar_exact_p', 'holm_p', 'verdict'):
                row[f'{k}_{tag}'] = src[k]
        row['verdict_differs_from_v4'] = c['verdict'] != a['verdict']
        row['verdict_differs_from_manual'] = c['verdict'] != b['verdict']
        srows.append(row)
    write_csv(FOLDER / 'significance_strict.csv', srows)
    fp = final_significance.fmt_p
    lines = ['# Significance tests — strict re-check of recovered answers', '',
             'Same method as `significance/significance.md` (paired, exact McNemar, Holm over 42 tests, bootstrap 95% CI, '
             '5,000 resamples, seed 20261014), on the strict labels. Verdicts from v4 and from the first manual review are shown.', '']
    for ds in DATASETS:
        lines += [f'## {DNAMES[ds]}', '']
        body = [[r['model'], r['condition'], r['turn'], r['n_paired'], r['T0_R_pct'], r['tool_R_pct'],
                 f"{r['diff_R_pp']:+.1f} [{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]",
                 f"{r['improved']} / {r['worsened']}", fp(r['mcnemar_exact_p']), fp(r['holm_p']),
                 r['verdict'], r['verdict_manual'], r['verdict_v4']]
                for r in srows if r['dataset'] == DNAMES[ds]]
        lines += [md_table(['Model', 'Tool', 'Turn', 'n', 'T0 R %', 'Tool R %', 'Diff pp [95% CI]', 'Improved / worsened',
                            'p', 'Holm p', 'Verdict (strict)', 'Verdict (manual)', 'Verdict (v4)'], body), '']
    (FOLDER / 'significance_strict.md').write_text('\n'.join(lines), encoding='utf-8')

    print('109 rows:', Counter(r['strict_decision'] for r in rows))
    print('kept labels:', Counter(r['strict_label'] for r in rows if r['strict_decision'] == 'keep'))
    for (ds, c), n in sorted(Counter((r['dataset'], r['condition']) for r in rows).items()):
        k = [r for r in rows if (r['dataset'], r['condition']) == (ds, c) and r['strict_decision'] == 'keep']
        print(f'  {ds} {c}: first {n}, strict keep {len(k)}', Counter(r['strict_label'] for r in k))
    for r in srows:
        if r['verdict_differs_from_v4'] or r['verdict_differs_from_manual']:
            print('VERDICT', r['dataset'], r['model'], r['condition'], r['turn'], 'v4:', r['verdict_v4'], '| manual:',
                  r['verdict_manual'], '| strict:', r['verdict'], f"diff {r['diff_R_pp_v4']} / {r['diff_R_pp_manual']} / {r['diff_R_pp']}",
                  f"[{r['ci_low_pp']}, {r['ci_high_pp']}] holm {r['holm_p_v4']:.3f} / {r['holm_p_manual']:.3f} / {r['holm_p']:.3f}")
    for a, b, c in zip(m_v4, m_man, m_str):
        if c['resistant_pct'] != b['resistant_pct'] or c['flip_pct'] != b['flip_pct']:
            print('METRIC', c['dataset'], c['model'], c['condition'], c['turn'], 'R', a['resistant_pct'], b['resistant_pct'],
                  c['resistant_pct'], 'S', a['flip_pct'], b['flip_pct'], c['flip_pct'], 'U', a['unparsed_pct'], b['unparsed_pct'], c['unparsed_pct'])


if __name__ == '__main__':
    main()
