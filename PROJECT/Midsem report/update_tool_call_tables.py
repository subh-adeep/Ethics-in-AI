"""Export per-pressure tool calls, with shared and full-run scopes kept separate."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
DRAFT=ROOT/'draft'
f=pd.read_csv(ROOT.parent/'codebase/results/midsem_t0_t1/tables/turn_level_metrics.csv')
rows=[]
for domain,n in [('medqa',146),('math',215)]:
    lines=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',
           r'Model & P1 calls & P1 rate (\%) & P2 calls & P2 rate (\%) & Total calls \\',r'\midrule']
    for model,label in [('qwen','Qwen'),('llama','LLaMA'),('phi4','Phi-4')]:
        counts=[]
        for turn in ('p1','p2'):
            g=f[(f.domain==domain)&(f.model==model)&(f.tier=='T1')&(f.turn==turn)&f.common_t0_correct]
            assert len(g)==n
            c=int(g.tool_invoked.sum());assert int(g.tool_success.sum())==c
            counts.append(c)
            rows.append({'domain':domain,'model':model,'turn':turn,'question_subset':'shared_T0_correct','opportunities':n,'calls':c,'successful':c,'call_rate_pct':100*c/n})
        a,b=counts
        lines.append(f'{label} & {a} & {100*a/n:.1f} & {b} & {100*b/n:.1f} & {a+b}'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (DRAFT/f'{domain}_pressure_tool_calls.tex').write_text('\n'.join(lines),encoding='utf-8')
pd.DataFrame(rows).to_csv(DRAFT/'pressure_tool_calls_shared.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
