# MedQA: separate T0 / T1 results

All six runs complete: **171 questions per run, 1026 dialogues, 3078 saved answer turns**.
Shared current T0-correct subset: **146 questions**. T2 web search and the extra experiment are excluded.

| Model | Turn | T0 resistance | T1 resistance | T0 false agreement | T1 false agreement |
|---|---|---:|---:|---:|---:|
| Qwen 2.5 3B | P1 | 66.4% | 66.4% | 4.1% | 4.1% |
| Qwen 2.5 3B | P2 | 18.5% | 18.5% | 56.8% | 56.8% |
| LLaMA 3.2 3B | P1 | 45.2% | 0.7% | 47.3% | 99.3% |
| LLaMA 3.2 3B | P2 | 36.3% | 2.7% | 42.5% | 95.9% |
| Phi-4 Mini | P1 | 87.0% | 96.6% | 7.5% | 3.4% |
| Phi-4 Mini | P2 | 89.7% | 71.9% | 6.8% | 26.0% |

## RAG outcomes on all domain questions

| Model | Calls | Successful | Resistant | False agreement | Other wrong | Unparsed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 3B | 4 | 4 | 1 | 3 | 0 | 0 |
| LLaMA 3.2 3B | 342 | 342 | 8 | 331 | 0 | 3 |
| Phi-4 Mini | 334 | 334 | 285 | 47 | 2 | 0 |

## Reporting notes

Results retain the frozen stored scorer and are provisional pending raw-response review. Unparsed replies remain in denominators. The common subset is chosen using T0 initial correctness only; full-domain and strict-correct sensitivity tables are also included.
T1 uses a privileged answer sheet and optional scaffolded tool use. Tool use is not randomized; conditional outcomes do not establish that RAG caused the difference. P2 follows P1, so pressure differences include conversation history.
Bootstrap intervals describe question-sampling variability, not repeated-run variability. The combined package contains the raw review queue and parser sensitivity figure.

Files: 12 plots in PNG/PDF/SVG, one combined PDF, CSV tables, and this summary. Original checkpoints are unchanged.
