"""Task 2: paired, question-level significance tests of each tool condition vs T0 (parser v4).

For each model x dataset x turn (P1, P2) and each tool condition:
  - paired set = question IDs correct at P0 in BOTH the T0 run and the tool run
  - outcome per question = resistant (R) or not resistant (S, O or U)
  - improved = not resistant at T0 but resistant with the tool; worsened = the reverse
  - exact McNemar test = two-sided binomial test on the discordant pairs (improved vs worsened, p = 0.5)
  - Holm-adjusted p over all 42 comparisons (also given)
  - paired bootstrap 95% CI of (tool resistant % - T0 resistant %): resample question IDs with
    replacement, 5,000 resamples, numpy default_rng(20261014) restarted for every comparison,
    percentile interval (same method as the midsem figure 09)
  - the same difference and CI for the sycophantic (flip) rate
Writes results/final_analysis_20261121/significance/{significance.csv, significance.md}.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import (BOOT_N, BOOT_SEED, CONDITIONS, DATASETS, DNAMES, MODELS, NAMES, OUT, keyed,
                                   md_table, p0_ok_ids)


def boot_ci(delta):
    rng = np.random.default_rng(BOOT_SEED)
    boot = delta[rng.integers(0, len(delta), size=(BOOT_N, len(delta)))].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def compare(table, model, dataset, cond, turn):
    ids = sorted(set(p0_ok_ids(model, 'T0', dataset, table)) & set(p0_ok_ids(model, cond, dataset, table)))
    t0 = np.array([table[(model, 'T0', i, turn)]['label'] for i in ids])
    tc = np.array([table[(model, cond, i, turn)]['label'] for i in ids])
    r0, rc = (t0 == 'R').astype(int), (tc == 'R').astype(int)
    s0, sc = (t0 == 'S').astype(int), (tc == 'S').astype(int)
    improved = int(((r0 == 0) & (rc == 1)).sum())
    worsened = int(((r0 == 1) & (rc == 0)).sum())
    p = binomtest(improved, improved + worsened, 0.5).pvalue if improved + worsened else 1.0
    lo, hi = boot_ci(rc - r0)
    slo, shi = boot_ci(sc - s0)
    return dict(dataset=DNAMES[dataset], model=NAMES[model], condition=cond, turn=turn.upper(), n_paired=len(ids),
                T0_R_pct=round(100 * r0.mean(), 1), tool_R_pct=round(100 * rc.mean(), 1),
                diff_R_pp=round(100 * (rc.mean() - r0.mean()), 1), ci_low_pp=round(lo, 1), ci_high_pp=round(hi, 1),
                improved=improved, worsened=worsened, mcnemar_exact_p=float(p),
                T0_S_pct=round(100 * s0.mean(), 1), tool_S_pct=round(100 * sc.mean(), 1),
                diff_S_pp=round(100 * (sc.mean() - s0.mean()), 1), S_ci_low_pp=round(slo, 1), S_ci_high_pp=round(shi, 1),
                flips_added=int(((s0 == 0) & (sc == 1)).sum()), flips_removed=int(((s0 == 1) & (sc == 0)).sum()))


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def verdict(row):
    if row['holm_p'] < 0.05:
        return 'better' if row['diff_R_pp'] > 0 else 'worse'
    if row['mcnemar_exact_p'] < 0.05:
        return ('better' if row['diff_R_pp'] > 0 else 'worse') + ' (not after Holm)'
    return 'no clear change'


def build_rows(table):
    rows = [compare(table, m, d, c, t) for d in DATASETS for m in MODELS for c in CONDITIONS[d] if c != 'T0'
            for t in ('p1', 'p2')]
    adj = holm(np.array([r['mcnemar_exact_p'] for r in rows]))
    for r, a in zip(rows, adj):
        r['holm_p'] = float(a)
        r['verdict'] = verdict(r)
    return rows


def main():
    rows = build_rows(keyed())
    out = OUT / 'significance'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'significance.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ['# Paired significance tests: each tool condition vs no tool (T0), parser v4', '',
             'Same question IDs in both runs (correct at P0 in both). "Resistant" vs "not resistant" per question.',
             'Improved = not resistant at T0, resistant with the tool. Worsened = the reverse.',
             'p = exact McNemar (two-sided binomial test on improved vs worsened). Holm p corrects for all '
             f'{len(rows)} tests. CI = paired bootstrap 95% interval of the difference in resistant %, '
             f'{BOOT_N:,} resamples, seed {BOOT_SEED}.', '']
    for d in DATASETS:
        lines += [f'## {DNAMES[d]}', '']
        body = [[r['model'], r['condition'], r['turn'], r['n_paired'], r['T0_R_pct'], r['tool_R_pct'],
                 f"{r['diff_R_pp']:+.1f} [{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]",
                 f"{r['improved']} / {r['worsened']}", fmt_p(r['mcnemar_exact_p']), fmt_p(r['holm_p']),
                 f"{r['diff_S_pp']:+.1f}", r['verdict']]
                for r in rows if r['dataset'] == DNAMES[d]]
        lines += [md_table(['Model', 'Tool', 'Turn', 'n', 'T0 R %', 'Tool R %', 'Diff pp [95% CI]',
                            'Improved / worsened', 'McNemar p', 'Holm p', 'Flip diff pp', 'Verdict'], body), '']
    (out / 'significance.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
