"""Direct paired comparison of Exact (one answer-sheet entry) against T1 (BM25 top-2 entries).

Uses the final labels (parser v4 + strict AI-agent review) on the questions that are correct at P0
in both runs, and an exact McNemar test (two-sided binomial test on the discordant pairs) with Holm
correction over the 12 tests (2 datasets x 3 models x 2 turns).

Output: results/final_analysis_20261121/exact_vs_t1/exact_vs_t1.csv
Run from PROJECT/codebase:  python src/analysis/final_exact_vs_t1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_common import OUT, all_records  # noqa: E402

STRICT = OUT / 'manual_unparsed_review_strict' / 'strict_recheck.csv'
NAME = {'qwen': 'Qwen', 'llama': 'LLaMA', 'phi4': 'Phi-4'}
DS = {'medqa': 'MedQA', 'math': 'MATH'}


def final_labels():
    with open(STRICT, encoding='utf-8-sig') as f:
        strict = {(r['dataset'], r['model'], r['condition'], r['turn'], r['question_id']): r['strict_label']
                  for r in csv.DictReader(f)}
    labels = {}
    for r in all_records():
        key = (DS[r['dataset']], NAME[r['model']], r['condition'], r['turn'].upper(), r['id'])
        labels[key] = strict.get(key, r['label'])
    return labels


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    lab = final_labels()
    p0_ok = {}
    for (d, m, c, t, q), label in lab.items():
        if t == 'P0' and label == 'R':
            p0_ok.setdefault((d, m, c), set()).add(q)
    rows = []
    for d in ('MedQA', 'MATH'):
        for m in ('Qwen', 'LLaMA', 'Phi-4'):
            common = sorted(p0_ok[(d, m, 'T1')] & p0_ok[(d, m, 'Exact')])
            for t in ('P1', 'P2'):
                r_t1 = [lab[(d, m, 'T1', t, q)] == 'R' for q in common]
                r_ex = [lab[(d, m, 'Exact', t, q)] == 'R' for q in common]
                better = sum(e and not o for e, o in zip(r_ex, r_t1))
                worse = sum(o and not e for e, o in zip(r_ex, r_t1))
                p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
                rows.append(dict(dataset=d, model=m, turn=t, n_paired=len(common),
                                 T1_R_pct=round(100 * sum(r_t1) / len(common), 1),
                                 Exact_R_pct=round(100 * sum(r_ex) / len(common), 1),
                                 exact_better=better, exact_worse=worse, mcnemar_exact_p=p))
    for row, adj in zip(rows, holm([r['mcnemar_exact_p'] for r in rows])):
        row['holm_p'] = adj
        row['verdict'] = ('Exact better' if row['exact_better'] > row['exact_worse'] else 'Exact worse') if adj < 0.05 else 'no clear difference'
    out = OUT / 'exact_vs_t1'
    out.mkdir(exist_ok=True)
    with open(out / 'exact_vs_t1.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:5} {r['model']:5} {r['turn']}  T1 {r['T1_R_pct']:5.1f}  Exact {r['Exact_R_pct']:5.1f}  "
              f"+{r['exact_better']}/-{r['exact_worse']}  Holm p {r['holm_p']:.2g}  {r['verdict']}")


if __name__ == '__main__':
    main()
