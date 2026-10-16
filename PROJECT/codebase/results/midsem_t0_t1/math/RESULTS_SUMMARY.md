# MATH: separate T0 / T1 results

All six runs complete: **250 questions per run, 1500 dialogues, 4500 saved answer turns**.
Shared current T0-correct subset: **215 questions**. T2 web search and the extra experiment are excluded.

| Model | Turn | T0 resistance | T1 resistance | T0 false agreement | T1 false agreement |
|---|---|---:|---:|---:|---:|
| Qwen 2.5 3B | P1 | 58.1% | 58.1% | 0.0% | 0.0% |
| Qwen 2.5 3B | P2 | 1.4% | 1.9% | 98.6% | 97.7% |
| LLaMA 3.2 3B | P1 | 54.0% | 34.0% | 4.7% | 64.7% |
| LLaMA 3.2 3B | P2 | 1.4% | 1.4% | 71.6% | 98.1% |
| Phi-4 Mini | P1 | 40.0% | 97.7% | 12.6% | 1.9% |
| Phi-4 Mini | P2 | 32.1% | 74.0% | 43.3% | 26.0% |

## RAG outcomes on all domain questions

| Model | Calls | Successful | Resistant | False agreement | Other wrong | Unparsed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 3B | 9 | 9 | 1 | 7 | 1 | 0 |
| LLaMA 3.2 3B | 500 | 500 | 92 | 403 | 4 | 1 |
| Phi-4 Mini | 498 | 498 | 427 | 70 | 1 | 0 |

## Reporting notes

Results retain the frozen stored scorer and are provisional pending raw-response review. Unparsed replies remain in denominators. The common subset is chosen using T0 initial correctness only; full-domain and strict-correct sensitivity tables are also included.
T1 uses a privileged answer sheet and optional scaffolded tool use. Tool use is not randomized; conditional outcomes do not establish that RAG caused the difference. P2 follows P1, so pressure differences include conversation history.
Bootstrap intervals describe question-sampling variability, not repeated-run variability. The combined package contains the raw review queue and parser sensitivity figure.

Files: 12 plots in PNG/PDF/SVG, one combined PDF, CSV tables, and this summary. Original checkpoints are unchanged.
