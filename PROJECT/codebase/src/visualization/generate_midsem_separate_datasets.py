"""Export standalone MedQA and MATH mid-semester packages; T2 excluded."""
import hashlib
import html
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from generate_midsem_t0_t1 import OUT, MODELS, NAMES, OUTCOMES, LABELS, COLORS, TIER_COLORS


def export(domain, frame):
    title = 'MedQA' if domain == 'medqa' else 'MATH'
    root = OUT / domain
    for sub in ('plots', 'tables'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    data = frame[frame.domain == domain].copy()
    n = data.id.nunique()
    shared_ids = set(data.loc[data.common_t0_correct, 'id'])
    pressure = data[data.turn != 'p0'].copy()
    shared = pressure[pressure.common_t0_correct].copy()
    assert n == (171 if domain == 'medqa' else 250)
    assert all(len(g) == n for _, g in data.groupby(['model', 'tier', 'turn']))
    assert set(data.tier) == {'T0', 'T1'}
    data.to_csv(root / 'tables/turn_level_metrics.csv', index=False)
    combined = pd.read_csv(OUT / 'tables/outcome_summary.csv')
    combined[combined.domain == domain].to_csv(root / 'tables/outcome_summary.csv', index=False)
    quality = data.groupby(['model', 'tier', 'turn']).agg(n=('id', 'size'),
            unparsed=('outcome', lambda x: (x == 'unparsed').sum()),
            token_limit=('hit_token_limit', 'sum'), parser_changes=('parser_changed', 'sum')).reset_index()
    quality.to_csv(root / 'tables/quality_audit.csv', index=False)
    tool_rows = []
    for m in MODELS:
        g = pressure[(pressure.model == m) & (pressure.tier == 'T1')]
        called = g[g.tool_invoked]
        success = called[called.tool_success]
        eligible = g[g.evidence_override.notna()]
        tool_rows.append({'model': m, 'opportunities': len(g), 'calls': len(called), 'successful': len(success),
                         'call_rate_pct': 100 * len(called) / len(g),
                         'evidence_denominator': len(eligible), 'evidence_override_n': int(eligible.evidence_override.sum()),
                         **{o: int((success.outcome == o).sum()) for o in OUTCOMES}})
    tools = pd.DataFrame(tool_rows).set_index('model').loc[MODELS]
    tools.to_csv(root / 'tables/tool_metrics.csv')
    rng = np.random.default_rng(20261014)
    paired_rows = []
    transition_rows = []
    for m in MODELS:
        for p in ('p1', 'p2'):
            wide = shared[(shared.model == m) & (shared.turn == p)].pivot(index='id', columns='tier', values='outcome')
            assert len(wide) == len(shared_ids)
            for o in OUTCOMES:
                delta = (wide.T1 == o).astype(int).to_numpy() - (wide.T0 == o).astype(int).to_numpy()
                samples = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(axis=1) * 100
                lo, hi = np.percentile(samples, [2.5, 97.5])
                paired_rows.append({'model': m, 'turn': p, 'outcome': o, 'n': len(delta),
                                    'delta_pp': delta.mean()*100, 'ci_low': lo, 'ci_high': hi})
            for a in OUTCOMES:
                for b in OUTCOMES:
                    transition_rows.append({'model': m, 'turn': p, 'T0': a, 'T1': b,
                                            'n': int(((wide.T0 == a) & (wide.T1 == b)).sum())})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / 'tables/paired_differences.csv', index=False)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(root / 'tables/paired_transitions.csv', index=False)
    figures = []
    book = PdfPages(root / f'{domain}_T0_T1_plots.pdf')

    primary_plots = {'03_shared_correct_outcomes', '04_resistance', '05_false_agreement',
                     '06_pressure_sequence', '07_paired_differences'}
    def save(fig, name, caption):
        count = len(shared_ids) if name in primary_plots else n
        subset = 'shared_T0_correct' if name in primary_plots else 'all_questions'
        scope = f'{title}: main comparison, {count} shared questions' if name in primary_plots else f'{title}: full-run diagnostics, {count} questions'
        fig.text(.015, .038, scope, fontsize=9, color='#223344')
        fig.text(.015, .012, f'{title} only · T0/T1 · Stored scorer labels; unresolved replies retained. T2 excluded.', fontsize=8, color='#525964')
        fig.tight_layout(rect=(0, .085, 1, .94))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(root / 'plots' / f'{name}.{ext}', dpi=220, bbox_inches='tight')
        book.savefig(fig, bbox_inches='tight')
        figures.append({'name': name, 'caption': caption, 'question_subset': subset,
                        'questions_per_model_setting': count, 'scope': scope})
        plt.close(fig)

    def stack(dataset, name, heading):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, p in zip(axes, ('p1', 'p2')):
            groups = [dataset[(dataset.model == m) & (dataset.tier == t) & (dataset.turn == p)] for m in MODELS for t in ('T0', 'T1')]
            bottom = np.zeros(6)
            for o, label, color in zip(OUTCOMES, LABELS, COLORS):
                vals = np.array([100*(g.outcome == o).mean() for g in groups])
                ax.bar(range(6), vals, bottom=bottom, color=color, label=label)
                for x, v, b in zip(range(6), vals, bottom):
                    if v >= 9: ax.text(x, b+v/2, f'{v:.0f}%', ha='center', va='center', color='white')
                bottom += vals
            ax.set_xticks(range(6), [NAMES[m]+'\n'+t for m in MODELS for t in ('T0','T1')], rotation=25, ha='right')
            ax.set_title(p.upper()); ax.set_ylim(0,100)
        axes[0].set_ylabel('Percentage of pressure replies')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(.5,1.01))
        fig.suptitle(heading, y=1.09)
        save(fig, name, heading)

    fig, ax = plt.subplots(figsize=(9,4.5))
    for offset, t, color in zip((-.18,.18),('T0','T1'),TIER_COLORS):
        vals = [100*data[(data.model==m)&(data.tier==t)&(data.turn=='p0')].p0_correct.mean() for m in MODELS]
        ax.bar(np.arange(3)+offset, vals, .34, color=color, label=t)
        for x,v in zip(np.arange(3)+offset,vals): ax.text(x,v+1,f'{v:.1f}%',ha='center')
    ax.set_xticks(range(3),[NAMES[m] for m in MODELS]); ax.set_ylim(0,110); ax.set_ylabel('P0 correctness (%)'); ax.legend()
    ax.set_title(f'{title}: initial accuracy (n={n} per run)')
    save(fig,'01_initial_accuracy','Initial correctness on the historical selection, assessed again in the current runs.')
    stack(pressure,'02_all_question_outcomes',f'{title}: all {n} questions')
    stack(shared,'03_shared_correct_outcomes',f'{title}: shared T0-correct cohort (n={len(shared_ids)})')
    for o, slug in [('resistant','04_resistance'),('sycophantic','05_false_agreement')]:
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,p in zip(axes,('p1','p2')):
            for offset,t,color in zip((-.18,.18),('T0','T1'),TIER_COLORS):
                vals = [100*(shared[(shared.model==m)&(shared.tier==t)&(shared.turn==p)].outcome==o).mean() for m in MODELS]
                ax.bar(np.arange(3)+offset,vals,.34,label=t,color=color)
                for x,v in zip(np.arange(3)+offset,vals): ax.text(x,v+1,f'{v:.1f}',ha='center',fontsize=9)
            ax.set_xticks(range(3),[NAMES[m] for m in MODELS],rotation=15); ax.set_ylim(0,110); ax.set_title(p.upper()); ax.legend()
        axes[0].set_ylabel('Percentage of replies'); fig.suptitle(f'{title}: {o.replace("_"," ")} (shared n={len(shared_ids)})')
        save(fig,slug,'Comparison on questions all three models answered correctly in current T0.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,t in zip(axes,('T0','T1')):
        for m,color in zip(MODELS,('#436baf','#b66b25','#16836b')):
            vals=[100*(shared[(shared.model==m)&(shared.tier==t)&(shared.turn==p)].outcome=='resistant').mean() for p in ('p1','p2')]
            ax.plot([1,2],vals,'o-',color=color,label=NAMES[m])
        ax.set_xticks([1,2],['P1: ordinary user','P2: claimed expertise']); ax.set_ylim(0,105); ax.set_title(t); ax.legend(fontsize=9)
    axes[0].set_ylabel('Resistance (%)'); fig.suptitle(title+': sequential pressure turns')
    save(fig,'06_pressure_sequence','P2 follows P1; this does not isolate the authority effect from conversation history.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,o in zip(axes,('resistant','sycophantic')):
        g=paired[paired.outcome==o].set_index(['model','turn']).loc[[(m,p) for m in MODELS for p in ('p1','p2')]].reset_index()
        val=g.delta_pp.to_numpy(); ax.errorbar(val,range(6),xerr=np.array([val-g.ci_low.to_numpy(),g.ci_high.to_numpy()-val]),fmt='o',capsize=4)
        ax.axvline(0,color='#777',ls='--');ax.set_yticks(range(6),[NAMES[m]+' '+p.upper() for m in MODELS for p in ('p1','p2')]);ax.invert_yaxis();ax.set_xlabel('T1 minus T0 (percentage points)');ax.set_title(o.capitalize())
    save(fig,'07_paired_differences','Shared T0-correct cohort; 95% paired question-bootstrap intervals, 5,000 resamples. Descriptive, not causal.')
    fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(range(3),tools.call_rate_pct,color=['#436baf','#b66b25','#16836b'])
    for x,m in enumerate(MODELS):ax.text(x,tools.loc[m,'call_rate_pct']+1,f'{tools.loc[m,"call_rate_pct"]:.1f}%\n{int(tools.loc[m,"calls"])}/{2*n}',ha='center')
    ax.set_xticks(range(3),[NAMES[m] for m in MODELS]);ax.set_ylim(0,110);ax.set_ylabel('RAG call rate (%)');ax.set_title(f'{title}: optional RAG use')
    save(fig,'08_rag_utilization','Two pressure-turn opportunities per question. Model chooses verification; Python executes retrieval.')
    fig,ax=plt.subplots(figsize=(10,5));bottom=np.zeros(3)
    for o,label,color in zip(OUTCOMES,LABELS,COLORS):
        vals=np.array([100*tools.loc[m,o]/tools.loc[m,'successful'] if tools.loc[m,'successful'] else 0 for m in MODELS])
        ax.bar(range(3),vals,bottom=bottom,label=label,color=color)
        for x,v,b in zip(range(3),vals,bottom):
            if v>7:ax.text(x,b+v/2,f'{v:.1f}%',ha='center',va='center',color='white')
        bottom+=vals
    for x,m in enumerate(MODELS):
        if not tools.loc[m,'successful']:ax.text(x,45,'No RAG calls\nRate undefined',ha='center')
    ax.set_xticks(range(3),[NAMES[m]+f'\nn={int(tools.loc[m,"successful"])} calls' for m in MODELS]);ax.set_ylim(0,100);ax.set_ylabel('Outcome after successful RAG (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,fontsize=9)
    save(fig,'09_after_rag_outcomes',title+': tool-calling subsets differ between models; zero calls means no defined conditional rate.')
    fig,ax=plt.subplots(figsize=(9,4.5))
    vals=[100*tools.loc[m,'evidence_override_n']/tools.loc[m,'evidence_denominator'] if tools.loc[m,'evidence_denominator'] else np.nan for m in MODELS]
    ax.bar(range(3),vals,color='#c74752')
    for x,m in enumerate(MODELS):
        if np.isfinite(vals[x]):ax.text(x,vals[x]+1,f'{vals[x]:.1f}%\n{int(tools.loc[m,"evidence_override_n"])}/{int(tools.loc[m,"evidence_denominator"])}',ha='center')
        else:ax.text(x,30,'No eligible calls\nUndefined',ha='center')
    ax.set_xticks(range(3),[NAMES[m] for m in MODELS]);ax.set_ylim(0,110);ax.set_ylabel('False agreement after supported retrieval (%)');ax.set_title(title+': evidence override')
    save(fig,'10_evidence_override','Denominator: successful target-answer retrievals. Unparsed answers are not presumed resistant.')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,kind in zip(axes,('unparsed','cap')):
        for offset,p,color in zip((-.24,0,.24),('p0','p1','p2'),('#436baf','#b66b25','#16836b')):
            vals=[]
            for m in MODELS:
                for t in ('T0','T1'):
                    g=data[(data.model==m)&(data.tier==t)&(data.turn==p)]
                    vals.append(100*((g.outcome=='unparsed').mean() if kind=='unparsed' else g.hit_token_limit.mean()))
            ax.bar(np.arange(6)+offset,vals,.23,label=p.upper(),color=color)
        ax.set_xticks(range(6),[NAMES[m]+'\n'+t for m in MODELS for t in ('T0','T1')],rotation=25,ha='right');ax.set_ylim(0,100);ax.set_title('Unparsed' if kind=='unparsed' else 'Reached token limit');ax.legend()
    axes[0].set_ylabel('Percentage of replies');save(fig,'11_response_quality','Token-limit flags indicate possible truncation. Missing text cannot be reconstructed offline.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,t in zip(axes,('T0','T1')):
        groups=[pressure[(pressure.model==m)&(pressure.tier==t)&(pressure.turn==p)].generated_tokens.to_numpy() for m in MODELS for p in ('p1','p2')]
        ax.boxplot(groups,tick_labels=[NAMES[m]+'\n'+p.upper() for m in MODELS for p in ('p1','p2')],showfliers=False)
        ax.tick_params(axis='x',rotation=25);ax.axhline(160,color='#c74752',ls='--');ax.set_ylabel('Generated tokens');ax.set_title(t)
    save(fig,'12_response_lengths',title+': median, quartiles and 1.5-IQR whiskers; outliers hidden, token cap shown.')
    book.close()
    pd.DataFrame(figures).to_csv(root / 'tables/plot_question_sets.csv', index=False)
    assert len(shared_ids)==(146 if domain=='medqa' else 215)
    assert all(len(g)==len(shared_ids) for _,g in shared.groupby(['model','tier','turn']))
    lines=[f'# {title}: separate T0 / T1 results','',f'All six runs complete: **{n} questions per run, {6*n} dialogues, {18*n} saved answer turns**.',
           f'Shared current T0-correct subset: **{len(shared_ids)} questions**. T2 web search and the extra experiment are excluded.','',
           '| Model | Turn | T0 resistance | T1 resistance | T0 false agreement | T1 false agreement |','|---|---|---:|---:|---:|---:|']
    for m in MODELS:
        for p in ('p1','p2'):
            vals=[100*(shared[(shared.model==m)&(shared.tier==t)&(shared.turn==p)].outcome==o).mean() for o in ('resistant','sycophantic') for t in ('T0','T1')]
            lines.append(f'| {NAMES[m]} | {p.upper()} | '+' | '.join(f'{v:.1f}%' for v in vals)+' |')
    lines += ['', '## RAG outcomes on all domain questions','', '| Model | Calls | Successful | Resistant | False agreement | Other wrong | Unparsed |','|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:lines.append(f'| {NAMES[m]} | '+' | '.join(str(int(tools.loc[m,c])) for c in ('calls','successful',*OUTCOMES))+' |')
    lines += ['', '## Reporting notes','',
              'Results retain the frozen stored scorer and are provisional pending raw-response review. Unparsed replies remain in denominators. The common subset is chosen using T0 initial correctness only; full-domain and strict-correct sensitivity tables are also included.',
              'T1 uses a privileged answer sheet and optional scaffolded tool use. Tool use is not randomized; conditional outcomes do not establish that RAG caused the difference. P2 follows P1, so pressure differences include conversation history.',
              'Bootstrap intervals describe question-sampling variability, not repeated-run variability. The combined package contains the raw review queue and parser sensitivity figure.', '',
              'Files: 12 plots in PNG/PDF/SVG, one combined PDF, CSV tables, and this summary. Original checkpoints are unchanged.','']
    (root/'RESULTS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    metadata={'domain':domain,'tiers':['T0','T1'],'questions_per_run':n,'shared_T0_correct_n':len(shared_ids),'figures':figures,
              'source_turn_table_sha256':hashlib.sha256((OUT/'tables/turn_level_metrics.csv').read_bytes()).hexdigest()}
    (root/'analysis_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(f["name"])}</h2><img src="plots/{f["name"]}.png"><p>{html.escape(f["caption"])}</p></section>' for f in figures)
    (root/'index.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{title} results</title><style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f5f7fa}}section{{background:white;padding:20px;margin:24px 0}}img{{width:100%}}</style></head><body><h1>{title}: T0 versus T1</h1><p>{n} questions per run; shared T0-correct subset {len(shared_ids)}. T2 excluded. Provisional stored scores.</p><a href="{domain}_T0_T1_plots.pdf">All plots PDF</a>'+cards+'</body></html>',encoding='utf-8')
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.zip']
    with zipfile.ZipFile(root/f'{domain}_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(root))
    print(json.dumps({'domain':domain,'questions':n,'shared_correct':len(shared_ids),'figures':len(figures),'output':str(root)}))


if __name__ == '__main__':
    frame=pd.read_csv(OUT/'tables/turn_level_metrics.csv')
    assert set(frame.domain)=={'medqa','math'} and set(frame.tier)=={'T0','T1'}
    for domain in ('medqa','math'):
        export(domain,frame)
    gallery=OUT/'index.html'
    links='<section id="separate-datasets"><h2>Separate dataset results</h2><p><a href="medqa/index.html">MedQA gallery</a> · <a href="medqa/medqa_T0_T1_plots.pdf">MedQA PDF</a> · <a href="math/index.html">MATH gallery</a> · <a href="math/math_T0_T1_plots.pdf">MATH PDF</a></p></section>'
    content=gallery.read_text(encoding='utf-8')
    if 'id="separate-datasets"' not in content:
        gallery.write_text(content.replace('</body>',links+'</body>'),encoding='utf-8')
