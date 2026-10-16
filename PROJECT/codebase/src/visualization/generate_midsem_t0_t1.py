"""Reproducible mid-semester figures from verified T0/T1 checkpoints only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
TOOLS = PROJECT / 'kaggle/kernel_tools_v2'
CAPTURE = TOOLS / 'parallel_capture_20261014'
OUT = PROJECT / 'codebase/results/midsem_t0_t1'
MODELS = ['qwen', 'llama', 'phi4']
NAMES = {'qwen': 'Qwen 2.5 3B', 'llama': 'LLaMA 3.2 3B', 'phi4': 'Phi-4 Mini'}
OUTCOMES = ['resistant', 'sycophantic', 'unforced_error', 'unparsed']
LABELS = ['Resistant / correct', 'Agrees with false answer', 'Other wrong answer', 'Unparsed']
COLORS = ['#16836b', '#c74752', '#da9d32', '#89929e']
TIER_COLORS = ['#436baf', '#b66b25']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('plots', 'tables', 'audit'):
        (OUT / folder).mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('midsem_evaluator', TOOLS / 'runner.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    manifest_path = CAPTURE / 'parallel_assets/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    items = {i['id']: i for i in manifest['items']}
    verification = json.loads((CAPTURE / 'verification_report.json').read_text(encoding='utf-8'))
    rows, sources, checkpoints = [], {}, {}
    for model in MODELS:
        for tier in ('T0', 'T1'):
            path = CAPTURE / 'parallel_results' / f'{model}_{tier}' / f'{model}_{tier}_optional.json'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = next(e for e in verification['entries'] if e['model'] == model and e['tier'] == tier)
            assert digest == entry['sha256'], f'Checkpoint changed: {path}'
            c = json.loads(path.read_text(encoding='utf-8'))
            assert c['completed'] == c['total'] == 421
            checkpoints[(model, tier)] = c
            sources[str(path)] = digest
            for dialogue in c['dialogues']:
                for turn in ('p0', 'p1', 'p2'):
                    t = dialogue[turn]
                    revised = evaluator.classify(t['text'], items[dialogue['id']])
                    tool = t.get('tool', {})
                    rows.append({'model': model, 'tier': tier, 'id': dialogue['id'], 'domain': dialogue['dataset'],
                                 'turn': turn, 'p0_correct': dialogue['p0_correct'], 'outcome': t['outcome'],
                                 'extracted': t.get('extracted'), 'text': t['text'],
                                 'generated_tokens': t['generated_tokens'], 'hit_token_limit': t['hit_token_limit'],
                                 'tool_invoked': bool(tool.get('tool_invoked')), 'tool_success': bool(tool.get('tool_success')),
                                 'evidence_override': t.get('evidence_override'), 'revised_outcome': revised['outcome'],
                                 'revised_extracted': revised['extracted'],
                                 'parser_changed': revised['outcome'] != t['outcome'] or revised['extracted'] != t.get('extracted')})
    df = pd.DataFrame(rows)
    common_t0 = set.intersection(*[set(df.loc[(df.model == m) & (df.tier == 'T0') & (df.turn == 'p0') & df.p0_correct, 'id']) for m in MODELS])
    common_both = set.intersection(*[set(df.loc[(df.model == m) & (df.tier == t) & (df.turn == 'p0') & df.p0_correct, 'id']) for m in MODELS for t in ('T0', 'T1')])
    df['common_t0_correct'] = df.id.isin(common_t0)
    df['common_both_correct'] = df.id.isin(common_both)
    pressure = df[df.turn != 'p0'].copy()
    shared = pressure[pressure.common_t0_correct].copy()
    summary_rows = []
    for subset, data in [('all_421', pressure), ('common_T0_correct', shared),
                         ('common_T0_T1_correct_sensitivity', pressure[pressure.common_both_correct]),
                         ('run_specific_P0_correct', pressure[pressure.p0_correct])]:
        for domain in ('all', 'medqa', 'math'):
            selected = data if domain == 'all' else data[data.domain == domain]
            for (model, tier, turn), group in selected.groupby(['model', 'tier', 'turn']):
                counts = group.outcome.value_counts()
                entry = {'subset': subset, 'domain': domain, 'model': model, 'tier': tier, 'turn': turn,
                         'denominator': len(group)}
                for outcome in OUTCOMES:
                    entry[outcome + '_n'] = int(counts.get(outcome, 0))
                    entry[outcome + '_pct'] = 100 * counts.get(outcome, 0) / len(group)
                summary_rows.append(entry)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'tables/outcome_summary.csv', index=False)
    df.drop(columns=['text']).to_csv(OUT / 'tables/turn_level_metrics.csv', index=False)
    pd.DataFrame([{'id': i, 'domain': items[i]['dataset']} for i in items if i in common_t0]).to_csv(OUT / 'tables/common_T0_correct_questions.csv', index=False)
    review = df[(df.outcome == 'unparsed') | df.hit_token_limit | df.parser_changed | ((df.turn == 'p0') & ~df.p0_correct)].copy()
    review['ground_truth'] = review.id.map(lambda i: items[i]['ground_truth'])
    review['false_answer'] = review.id.map(lambda i: items[i].get('false_answer_key') if items[i]['dataset'] == 'medqa' else items[i].get('false_answer'))
    review['review_status'] = 'manual_review_required'
    review.to_json(OUT / 'audit/raw_response_review.jsonl', orient='records', lines=True, force_ascii=False)
    quality = df.groupby(['model', 'tier', 'turn']).agg(n=('id', 'size'), unparsed=('outcome', lambda x: (x == 'unparsed').sum()),
               token_limit=('hit_token_limit', 'sum'), parser_changes=('parser_changed', 'sum'),
               revised_unparsed=('revised_outcome', lambda x: (x == 'unparsed').sum())).reset_index()
    quality.to_csv(OUT / 'tables/quality_and_parser_audit.csv', index=False)
    tools = []
    for (model, tier), g in pressure.groupby(['model', 'tier']):
        called = g[g.tool_invoked]
        success = called[called.tool_success]
        eligible = g[g.evidence_override.notna()]
        record = {'model': model, 'tier': tier, 'opportunities': len(g), 'requested': len(called), 'successful': len(success),
                  'call_rate_pct': 100*len(called)/len(g), 'success_rate_pct': 100*len(success)/len(called) if len(called) else None,
                  'supported_evidence_n': len(eligible), 'false_agreement_after_supported_evidence_n': int(eligible.evidence_override.sum())}
        record.update({o+'_after_success_n': int((success.outcome == o).sum()) for o in OUTCOMES})
        tools.append(record)
    tool_table = pd.DataFrame(tools)
    tool_table.to_csv(OUT / 'tables/tool_metrics.csv', index=False)
    paired_rows, transition_rows = [], []
    rng = np.random.default_rng(20261014)
    for model in MODELS:
        for turn in ('p1', 'p2'):
            data = shared[(shared.model == model) & (shared.turn == turn)]
            wide = data.pivot(index='id', columns='tier', values='outcome')
            assert len(wide) == len(common_t0) and wide.notna().all().all()
            for outcome in OUTCOMES:
                delta = (wide.T1 == outcome).astype(int).to_numpy() - (wide.T0 == outcome).astype(int).to_numpy()
                boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)*100
                lo, hi = np.percentile(boot, [2.5, 97.5])
                paired_rows.append({'model': model, 'turn': turn, 'outcome': outcome, 'n': len(delta),
                                    'T1_minus_T0_pp': delta.mean()*100, 'ci_low_pp': lo, 'ci_high_pp': hi})
            for before in OUTCOMES:
                for after in OUTCOMES:
                    transition_rows.append({'model': model, 'turn': turn, 'T0_outcome': before, 'T1_outcome': after,
                                            'n': int(((wide.T0 == before) & (wide.T1 == after)).sum())})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / 'tables/paired_differences.csv', index=False)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(OUT / 'tables/T0_to_T1_transitions.csv', index=False)
    figures = []
    book = PdfPages(OUT / 'midsem_T0_T1_plots.pdf')

    primary_plots = {'04_common_correct_outcomes', '05_resistance', '06_false_agreement',
                     '07_pressure_sequence', '08_domain_comparison', '09_paired_differences',
                     '13_paired_transitions'}
    def plot_scope(slug):
        if slug in primary_plots:
            return ('Main comparison: shared 361 questions (146 MedQA + 215 MATH)', 361, 'shared_T0_correct')
        return ('Full-run diagnostics: 421 questions (171 MedQA + 250 MATH)', 421, 'all_questions')

    def save(fig, slug, caption, footer='Stored scorer labels; unparsed replies stay in the denominator. T2 excluded.'):
        scope, count, subset = plot_scope(slug)
        fig.text(.015, .038, scope, fontsize=9, color='#223344')
        fig.text(.015, .012, footer, fontsize=8, color='#525964')
        fig.tight_layout(rect=(0, .085, 1, .96))
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(OUT / 'plots' / f'{slug}.{extension}', dpi=220, bbox_inches='tight')
        book.savefig(fig, bbox_inches='tight')
        figures.append({'file': slug, 'caption': caption, 'question_subset': subset,
                        'questions_per_model_setting': count, 'scope': scope})
        plt.close(fig)

    def stacked(data, slug, title):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, turn in zip(axes, ('p1', 'p2')):
            groups = [data[(data.model == m) & (data.tier == t) & (data.turn == turn)] for m in MODELS for t in ('T0', 'T1')]
            bottom = np.zeros(6)
            for outcome, label, color in zip(OUTCOMES, LABELS, COLORS):
                vals = np.array([100*(g.outcome == outcome).mean() for g in groups])
                ax.bar(np.arange(6), vals, bottom=bottom, color=color, label=label)
                for x, val, base in zip(range(6), vals, bottom):
                    if val >= 9: ax.text(x, base+val/2, f'{val:.0f}%', ha='center', va='center', fontsize=9, color='white')
                bottom += vals
            ax.set_xticks(range(6), [NAMES[m]+'\n'+t for m in MODELS for t in ('T0', 'T1')], rotation=25, ha='right')
            ax.set_title('P1: ordinary user' if turn == 'p1' else 'P2: claimed expertise')
            ax.set_ylim(0, 100)
        axes[0].set_ylabel('Percentage of pressure replies')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.01), fontsize=9)
        fig.suptitle(title, y=1.09)
        save(fig, slug, title)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([NAMES[m]+' '+t for m in MODELS for t in ('T0', 'T1')], [421]*6, color=[TIER_COLORS[i%2] for i in range(6)])
    for y in range(6): ax.text(425, y, '421 / 421', va='center')
    ax.set_xlim(0, 495); ax.invert_yaxis(); ax.set_xlabel('Completed question dialogues')
    ax.set_title('Mid-semester scope: six complete T0/T1 runs (2,526 dialogues)')
    save(fig, '01_completed_scope', 'All six runs are complete; each dialogue has P0, P1 and P2.', 'T0: no tools. T1: optional privileged answer-sheet RAG. T2 excluded.')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, tier, color in zip((-.18,.18), ('T0','T1'), TIER_COLORS):
        vals = [100*df[(df.model == m)&(df.tier == tier)&(df.turn == 'p0')].p0_correct.mean() for m in MODELS]
        ax.bar(np.arange(3)+offset, vals, .34, label=tier, color=color)
        for x,v in zip(np.arange(3)+offset, vals): ax.text(x,v+1,f'{v:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(range(3),[NAMES[m] for m in MODELS]); ax.set_ylim(0,110); ax.set_ylabel('P0 correctness (%)'); ax.legend()
    ax.set_title('Initial correctness on the 421-question historical selection')
    save(fig,'02_initial_accuracy','Current P0 accuracy is below 100% despite historical all-model-correct selection.')
    stacked(pressure,'03_all_question_outcomes','T0 versus T1: outcomes on all 421 questions')
    stacked(shared,'04_common_correct_outcomes',f'T0 versus T1: shared T0-correct cohort (n={len(common_t0)})')

    for outcome, slug, title in [('resistant','05_resistance','Resistance'),('sycophantic','06_false_agreement','Agreement with the supplied false answer')]:
        fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharey=True)
        for ax, turn in zip(axes, ('p1','p2')):
            for offset,tier,color in zip((-.18,.18),('T0','T1'),TIER_COLORS):
                vals=[100*(shared[(shared.model==m)&(shared.tier==tier)&(shared.turn==turn)].outcome==outcome).mean() for m in MODELS]
                ax.bar(np.arange(3)+offset,vals,.34,label=tier,color=color)
                for x,v in zip(np.arange(3)+offset,vals): ax.text(x,v+1,f'{v:.1f}',ha='center',fontsize=9)
            ax.set_xticks(range(3),[NAMES[m] for m in MODELS],rotation=15); ax.set_ylim(0,110); ax.set_title(turn.upper()); ax.legend()
        axes[0].set_ylabel('Percentage of replies'); fig.suptitle(f'{title}: common T0-correct cohort (n={len(common_t0)})')
        save(fig,slug,title+' on the common T0-correct subset. T1 is optional RAG availability, not universal RAG exposure.')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,tier in zip(axes,('T0','T1')):
        for m,color in zip(MODELS,('#436baf','#b66b25','#16836b')):
            vals=[100*(shared[(shared.model==m)&(shared.tier==tier)&(shared.turn==p)].outcome=='resistant').mean() for p in ('p1','p2')]
            ax.plot([1,2],vals,'o-',label=NAMES[m],color=color)
        ax.set_xticks([1,2],['P1: ordinary user','P2: claimed expertise']);ax.set_ylim(0,105);ax.set_title(tier);ax.legend(fontsize=9)
    axes[0].set_ylabel('Resistance (%)');fig.suptitle('Resistance across sequential pressure turns')
    save(fig,'07_pressure_sequence','P2 follows P1 in the same conversation; this is not an isolated authority effect.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,outcome in zip(axes,('resistant','sycophantic')):
        matrix=[]
        for m in MODELS:
            row=[]
            for domain in ('medqa','math'):
                for tier in ('T0','T1'):
                    for p in ('p1','p2'):
                        g=shared[(shared.model==m)&(shared.domain==domain)&(shared.tier==tier)&(shared.turn==p)]
                        row.append(100*(g.outcome==outcome).mean())
            matrix.append(row)
        matrix=np.array(matrix);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        for y in range(3):
            for x in range(8): ax.text(x,y,f'{matrix[y,x]:.0f}',ha='center',va='center',color='white' if matrix[y,x]<50 else 'black',fontsize=9)
        ax.set_yticks(range(3),[NAMES[m] for m in MODELS]);ax.set_xticks(range(8),[d+'\n'+t+' '+p for d in ('MedQA','MATH') for t in ('T0','T1') for p in ('P1','P2')],rotation=45,ha='right')
        ax.set_title('Resistance (%)' if outcome=='resistant' else 'False agreement (%)');fig.colorbar(im,ax=ax,shrink=.65)
    save(fig,'08_domain_comparison','Domain-specific rates on shared T0-correct questions: 146 MedQA and 215 MATH.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,outcome in zip(axes,('resistant','sycophantic')):
        selected=paired[paired.outcome==outcome].set_index(['model','turn']).loc[[(m,p) for m in MODELS for p in ('p1','p2')]].reset_index()
        vals=selected.T1_minus_T0_pp.to_numpy();lo=selected.ci_low_pp.to_numpy();hi=selected.ci_high_pp.to_numpy()
        ax.errorbar(vals,np.arange(6),xerr=np.array([vals-lo,hi-vals]),fmt='o',color='#436baf',capsize=4)
        ax.axvline(0,color='#777',ls='--');ax.set_yticks(range(6),[NAMES[m]+' '+p.upper() for m in MODELS for p in ('p1','p2')]);ax.invert_yaxis()
        ax.set_xlabel('T1 minus T0 (percentage points)');ax.set_title(outcome.capitalize())
    save(fig,'09_paired_differences','Paired rate differences; 95% question-bootstrap intervals, 5,000 resamples, seed 20261014.', 'Common T0-correct cohort. Descriptive uncertainty; no causal attribution or multiplicity-adjusted hypothesis test.')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    t1=tool_table[tool_table.tier=='T1'].set_index('model').loc[MODELS]
    for ax,column,title in zip(axes,('call_rate_pct','success_rate_pct'),('RAG call rate / 842 opportunities','Retrieval success / requested calls')):
        vals=t1[column].to_numpy();ax.bar(range(3),vals,color=['#436baf','#b66b25','#16836b'])
        for x,v in enumerate(vals):ax.text(x,v+1,f'{v:.1f}%',ha='center')
        ax.set_xticks(range(3),[NAMES[m] for m in MODELS],rotation=15);ax.set_ylim(0,110);ax.set_title(title);ax.set_ylabel('Percentage')
    save(fig,'10_rag_utilization','Model chooses verify/answer; Python executes retrieval. Successful lookup does not imply correct final answer.')

    fig,ax=plt.subplots(figsize=(10,5));bottom=np.zeros(3)
    for outcome,label,color in zip(OUTCOMES,LABELS,COLORS):
        vals=100*t1[outcome+'_after_success_n'].to_numpy()/t1.successful.to_numpy()
        ax.bar(range(3),vals,bottom=bottom,color=color,label=label)
        for x,v,b in zip(range(3),vals,bottom):
            if v>7:ax.text(x,b+v/2,f'{v:.1f}%',ha='center',va='center',color='white')
        bottom+=vals
    ax.set_xticks(range(3),[NAMES[m]+f'\nSuccessful calls n={int(t1.loc[m,"successful"])}' for m in MODELS]);ax.set_ylim(0,100);ax.set_ylabel('Percentage of successful RAG turns');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,fontsize=9)
    save(fig,'11_after_rag_outcomes','Conditional outcomes after successful RAG calls. Qwen n=13 is small; tool-calling subsets differ between models.')

    fig,ax=plt.subplots(figsize=(9,4.5));vals=100*t1.false_agreement_after_supported_evidence_n/t1.supported_evidence_n
    ax.bar(range(3),vals,color='#c74752')
    for x,m in enumerate(MODELS):ax.text(x,vals.iloc[x]+1,f'{vals.iloc[x]:.1f}%\n{int(t1.loc[m,"false_agreement_after_supported_evidence_n"])}/{int(t1.loc[m,"supported_evidence_n"])}',ha='center')
    ax.set_xticks(range(3),[NAMES[m] for m in MODELS]);ax.set_ylim(0,108);ax.set_ylabel('False agreement after supported retrieval (%)');ax.set_title('Evidence override: retrieved correct answer, then chose supplied false answer')
    save(fig,'12_evidence_override','Denominator includes successful target-answer retrievals only; all 421-question runs, both pressure turns combined.')

    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for col,m in enumerate(MODELS):
        for row,p in enumerate(('p1','p2')):
            ax=axes[row,col];g=transitions[(transitions.model==m)&(transitions.turn==p)]
            mat=g.pivot(index='T0_outcome',columns='T1_outcome',values='n').reindex(index=OUTCOMES,columns=OUTCOMES).to_numpy()
            ax.imshow(mat,cmap='Blues',vmin=0,vmax=len(common_t0))
            for y in range(4):
                for x in range(4):ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white' if mat[y,x]>len(common_t0)/2 else 'black')
            short=['Resistant','False agree','Other wrong','Unparsed'];ax.set_xticks(range(4),short,rotation=35,ha='right');ax.set_yticks(range(4),short);ax.set_xlabel('T1 outcome');ax.set_ylabel('T0 outcome');ax.set_title(NAMES[m]+' '+p.upper())
    save(fig,'13_paired_transitions','Shared 361 questions per matrix; pairs the same question and turn across T0/T1. Cells are question counts.')

    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,kind in zip(axes,('unparsed','token_limit')):
        for offset,p,color in zip((-.24,0,.24),('p0','p1','p2'),('#436baf','#b66b25','#16836b')):
            vals=[]
            for m in MODELS:
                for t in ('T0','T1'):
                    g=df[(df.model==m)&(df.tier==t)&(df.turn==p)]
                    vals.append(100*((g.outcome=='unparsed').mean() if kind=='unparsed' else g.hit_token_limit.mean()))
            ax.bar(np.arange(6)+offset,vals,.23,label=p.upper(),color=color)
        ax.set_xticks(range(6),[NAMES[m]+'\n'+t for m in MODELS for t in ('T0','T1')],rotation=25,ha='right');ax.set_ylim(0,100);ax.legend();ax.set_title('Unparsed replies' if kind=='unparsed' else 'Reached generation token limit')
    axes[0].set_ylabel('Percentage of replies');save(fig,'14_response_quality','Hitting the token cap flags possible truncation; it does not prove the final answer is unusable.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,tier in zip(axes,('T0','T1')):
        groups=[pressure[(pressure.model==m)&(pressure.tier==tier)&(pressure.turn==p)].generated_tokens.to_numpy() for m in MODELS for p in ('p1','p2')]
        ax.boxplot(groups,tick_labels=[NAMES[m]+'\n'+p.upper() for m in MODELS for p in ('p1','p2')],showfliers=False)
        ax.set_xticks(range(1,7),[NAMES[m]+'\n'+p.upper() for m in MODELS for p in ('p1','p2')],rotation=25,ha='right');ax.axhline(160,ls='--',color='#c74752',label='160-token cap');ax.set_title(tier);ax.set_ylabel('Generated tokens');ax.legend(fontsize=9)
    save(fig,'15_response_lengths','Generated token lengths for pressure replies; boxplots show median, quartiles and 1.5-IQR whiskers, with outliers hidden.')

    fig,ax=plt.subplots(figsize=(11,5))
    for offset,column,color in zip((-.18,.18),('outcome','revised_outcome'),TIER_COLORS):
        vals=[100*(pressure[(pressure.model==m)&(pressure.tier==t)][column]=='unparsed').mean() for m in MODELS for t in ('T0','T1')]
        ax.bar(np.arange(6)+offset,vals,.34,color=color,label='Stored scorer' if column=='outcome' else 'Parser 3.1 audit')
    ax.set_xticks(range(6),[NAMES[m]+'\n'+t for m in MODELS for t in ('T0','T1')]);ax.set_ylabel('Unparsed pressure replies (%)');ax.set_title('Parser sensitivity: offline audit only');ax.legend()
    save(fig,'16_parser_sensitivity','Parser 3.1 is applied consistently to all raw replies for sensitivity only; manual review remains necessary.', 'Original checkpoints and primary stored-label figures are unchanged. T2 excluded.')
    book.close()
    pd.DataFrame(figures).to_csv(OUT / 'tables/plot_question_sets.csv', index=False)
    protocol={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'tiers_included':['T0','T1'],
              'questions_per_run':421,'completed_dialogues':2526,'saved_answer_turns':len(df),
              'common_T0_correct_n':len(common_t0),'common_T0_T1_correct_n':len(common_both),
              'bootstrap_seed':20261014,'bootstrap_resamples':5000,'primary_labels':'stored frozen-run scorer',
              'sensitivity_evaluator':evaluator.EVALUATOR_VERSION,'sources':sources,'figures':figures}
    (OUT/'analysis_manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    lines=['# Mid-semester results: T0 versus T1 RAG','',
           'Scope: the main initially-correct-to-false-assertion experiment only. T2 web-search results and the extra wrong-to-right experiment are excluded.', '',
           '**Completed: 6/6 T0/T1 runs; 421 questions per run; 2,526 dialogues; 7,578 saved answer turns.**', '',
           'The cohort contains 171 MedQA and 250 MATH questions. It was selected using earlier all-three-model-correct results; ambiguous duplicated IDs were excluded. In the current T0 runs, all three models were initially correct on 361 shared questions (146 MedQA, 215 MATH). A stricter subset initially correct across all six runs contains 359 questions.', '',
           '## Comparisons on the shared T0-correct cohort', '',
           'Rates use all 361 questions per cell, including unparsed responses. Positive resistance change means more correct answers; positive false-agreement change means more agreement with the supplied false answer.', '',
           '| Model | Turn | T0 resistant | T1 resistant | Change (pp) | T0 false agreement | T1 false agreement |',
           '|---|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        for p in ('p1','p2'):
            a=shared[(shared.model==m)&(shared.tier=='T0')&(shared.turn==p)];b=shared[(shared.model==m)&(shared.tier=='T1')&(shared.turn==p)]
            ar=100*(a.outcome=='resistant').mean();br=100*(b.outcome=='resistant').mean()
            lines.append(f'| {NAMES[m]} | {p.upper()} | {ar:.1f}% | {br:.1f}% | {br-ar:+.1f} | {100*(a.outcome=="sycophantic").mean():.1f}% | {100*(b.outcome=="sycophantic").mean():.1f}% |')
    lines += ['', '## Outcomes after successful RAG calls (all 421 questions)', '',
              '| Model | Calls | Resistant | False agreement | Other wrong | Unparsed |', '|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        r=t1.loc[m];lines.append(f'| {NAMES[m]} | {int(r.successful)} | '+ ' | '.join(str(int(r[o+'_after_success_n'])) for o in OUTCOMES)+' |')
    lines += ['', '## Interpretation and limitations', '',
              '- T1 supplies a privileged ground-truth answer sheet through RAG. It is not independent external evidence.',
              '- Tool use is scaffolded: the model chooses a single-token verify/answer action and Python executes retrieval. No claim of native JSON function calling is made.',
              '- Qwen called RAG only 13 times; its T1 results mainly measure tool availability without use. Tool-calling subsets differ across models and must not be compared as randomized groups.',
              '- P2 follows P1 in the same dialogue. Differences between P1 and P2 include conversation history and cannot isolate authority pressure alone.',
              '- Primary figures retain the frozen run scorer. Parser 3.1 is shown only as a consistent offline sensitivity audit. Unparsed, changed and token-limit cases are supplied for manual review; no missing response text is reconstructed.',
              '- A decrease in unparsed replies can alter apparent resistance or false-agreement rates. Strong behavioral claims require raw-response review.',
              '- Conditioning on correctness in both T0 and T1 is supplied only as a secondary sensitivity table. The shared comparison cohort is selected from T0 correctness alone.',
              '- Paired bootstrap intervals describe question-sampling variability in this fixed selected cohort. They do not establish generalization, isolate tool effects, or include variability over repeated model runs.',
              '- Prior historical no-tool runs used different execution settings and are not merged with these matched controls.', '',
              '## Files', '',
              '- `midsem_T0_T1_plots.pdf`: all 16 figures in one PDF.',
              '- `plots/`: individual PNG (220 dpi), PDF and SVG figures.',
              '- `tables/`: full-cohort, shared-correct, run-specific and strict-sensitivity summaries; paired differences; tool counts; turn-level metrics.',
              '- `audit/raw_response_review.jsonl`: original raw text and proposed parser labels requiring review.',
              '- `analysis_manifest.json`: source hashes, scope, evaluator version and reproducibility settings.', '']
    (OUT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    cards='\n'.join(f'<section><h2>{html.escape(f["file"].replace("_"," "))}</h2><img src="plots/{f["file"]}.png" alt="{html.escape(f["caption"])}"><p>{html.escape(f["caption"])}</p><a href="plots/{f["file"]}.pdf">PDF</a> · <a href="plots/{f["file"]}.svg">SVG</a></section>' for f in figures)
    (OUT/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Mid-semester T0 / T1 results</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#223}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}h1{font-size:30px}p{line-height:1.5}</style></head><body><h1>Mid-semester results: T0 versus T1 RAG</h1><p>Main pressure comparison: 361 shared questions (146 MedQA + 215 MATH). Full-run diagnostics: 421 questions per run, 2,526 dialogues. Each plot states its question set. T2 excluded. Stored-label results are provisional pending raw-response review.</p><p><a href="midsem_T0_T1_plots.pdf">Download all plots as PDF</a> · <a href="RESULTS_SUMMARY.md">Results summary</a></p>'+cards+'</body></html>',encoding='utf-8')
    assert set(df.tier)=={'T0','T1'} and len(df)==7578
    assert all(len(g)==421 for _,g in df.groupby(['model','tier','turn']))
    assert len(common_t0)==361 and len(common_both)==359
    assert len(figures)==16
    assert all(len(g)==361 for _,g in shared.groupby(['model','tier','turn']))
    assert shared.groupby('domain').id.nunique().to_dict()=={'math':215,'medqa':146}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in sources.items())
    print(json.dumps({'output':str(OUT),'figures':len(figures),'dialogues':2526,'common_T0_correct':len(common_t0),
                      'review_turns':len(review),'parser_changed_turns':int(df.parser_changed.sum())},indent=2))


if __name__=='__main__':
    main()
