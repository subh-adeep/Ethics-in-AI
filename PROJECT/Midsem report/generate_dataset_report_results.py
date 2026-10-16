"""Build dataset-specific report tables and figures from checked shared-question metrics."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DRAFT=ROOT/'draft'
RESULTS=ROOT.parent/'codebase/results/midsem_t0_t1'
MODELS=['qwen','llama','phi4']
NAMES={'qwen':'Qwen','llama':'LLaMA','phi4':'Phi-4'}
OUTCOMES=['resistant','sycophantic','unforced_error','unparsed']
summary=pd.read_csv(RESULTS/'tables/outcome_summary.csv')
turns=pd.read_csv(RESULTS/'tables/turn_level_metrics.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.spines.top':False,'axes.spines.right':False})
for domain,title,n in [('medqa','MedQA',146),('math','MATH',215)]:
    selected=turns[(turns.domain==domain)&turns.common_t0_correct&(turns.turn!='p0')]
    s=summary[(summary.domain==domain)&(summary.subset=='common_T0_correct')].set_index(['model','turn','tier'])
    assert len(s)==12 and (s.denominator==n).all()
    for key,g in selected.groupby(['model','turn','tier']):
        assert len(g)==n and g.id.nunique()==n
        for o in OUTCOMES:assert int((g.outcome==o).sum())==int(s.loc[key,o+'_n'])
    lines=[r'\begingroup',r'\setlength{\tabcolsep}{2.2pt}',r'\begin{tabular}{@{}llrrrrrrrr@{}}',r'\toprule',
           r'Model & Turn & $R_0$ (\%) & $R_1$ (\%) & $F_0$ (\%) & $F_1$ (\%) & $O_0$ (\%) & $O_1$ (\%) & $U_0$ (\%) & $U_1$ (\%) \\',r'\midrule']
    matrix=[];counts=[]
    for m in MODELS:
        for p in ('p1','p2'):
            vals=[s.loc[(m,p,t),o+'_pct'] for o in OUTCOMES for t in ('T0','T1')]
            lines.append((NAMES[m] if p=='p1' else '')+' & '+p.upper()+' & '+' & '.join(f'{v:.1f}' for v in vals)+r' \\')
            matrix.append([s.loc[(m,p,t),o+'_pct'] for t in ('T0','T1') for o in OUTCOMES])
            counts.append([int(s.loc[(m,p,t),o+'_n']) for t in ('T0','T1') for o in OUTCOMES])
        if m!=MODELS[-1]:lines.append(r'\addlinespace')
    lines += [r'\bottomrule',r'\end{tabular}',r'\endgroup']
    (DRAFT/f'{domain}_outcomes_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    out=s.reset_index();out.to_csv(DRAFT/f'{domain}_report_outcomes.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(6.8,4.7),sharey=True)
    for row,(o,label) in enumerate([('resistant','Correct answers (%)'),('sycophantic',"User's wrong answer (%)")]):
        for col,p in enumerate(('p1','p2')):
            ax=axes[row,col]
            for offset,t,color in [(-.18,'T0','#436baf'),(.18,'T1','#b66b25')]:
                vals=[s.loc[(m,p,t),o+'_pct'] for m in MODELS]
                bars=ax.bar(np.arange(3)+offset,vals,.34,label=t,color=color)
                ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],fontsize=8,padding=2)
            ax.set_xticks(range(3),[NAMES[m] for m in MODELS]);ax.set_ylim(0,112)
            if col==0:ax.set_ylabel(label)
            panel = chr(ord('a') + 2*row + col)
            pressure_label = 'P1: ordinary user' if p=='p1' else 'P2: claimed expertise'
            metric_label = 'Correct answers (resistance)' if row==0 else "Agreement with user's wrong answer"
            ax.set_title(f'({panel}) {pressure_label}\n{metric_label}', fontsize=9, pad=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,['T0: no tool','T1: optional BM25 retrieval'],loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False,fontsize=9)
    fig.suptitle(f'{title}: same {n} initially correct questions in every comparison',fontsize=11,x=.01,ha='left')
    fig.text(.01,.01,'T0: no tool. T1: optional BM25 answer retrieval. Unclear replies stay in the denominator.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.89))
    for ext in ('pdf','png','svg'):fig.savefig(DRAFT/'figures'/f'{domain}_comparison.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    a=np.array(matrix);fig,ax=plt.subplots(figsize=(6.8,3.7))
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for y in range(6):
        for x in range(8):ax.text(x,y,f'{a[y,x]:.1f}%\n({counts[y][x]}/{n})',ha='center',va='center',fontsize=8,color='white' if a[y,x]>52 else '#122438')
    ax.set_yticks(range(6),[NAMES[m]+' '+p.upper() for m in MODELS for p in ('p1','p2')])
    ax.set_xticks(range(8),[t+'\n'+o for t in ('T0','T1') for o in ('R','F','O','U')])
    ax.axvline(3.5,color='white',linewidth=3)
    ax.set_title(f'{title}: complete outcome matrix (n={n} per row and setting)',pad=12)
    fig.colorbar(im,ax=ax,label='Percent',shrink=.8)
    fig.text(.01,.015,"R: correct; F: user's wrong answer; O: other wrong; U: unclear. Cells show % and count.",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('pdf','png','svg'):fig.savefig(DRAFT/'figures'/f'{domain}_outcome_matrix.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    tools=pd.read_csv(RESULTS/domain/'tables/tool_metrics.csv').set_index('model')
    lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Model & Calls & Correct & False agree & Other wrong & Unclear & Call rate \\',r'\midrule']
    for m in MODELS:
        r=tools.loc[m];assert int(r.successful)==int(r.calls)
        lines.append(NAMES[m]+' & '+' & '.join(str(int(r[k])) for k in ('calls','resistant','sycophantic','unforced_error','unparsed'))+' & '+f'{r.call_rate_pct:.1f}'+r'\% \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (DRAFT/f'{domain}_tool_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    print(domain,'verified',n,'questions; outcome table, comparison graph, matrix and tool table saved.')

# Keep pressure-level call tables in step with the figure package.
import runpy
runpy.run_path(str(ROOT / "update_tool_call_tables.py"), run_name="__main__")
