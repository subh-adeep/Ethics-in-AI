# Mid-semester results: T0 versus T1 RAG

Scope: the main initially-correct-to-false-assertion experiment only. T2 web-search results and the extra wrong-to-right experiment are excluded.

**Completed: 6/6 T0/T1 runs; 421 questions per run; 2,526 dialogues; 7,578 saved answer turns.**

The cohort contains 171 MedQA and 250 MATH questions. It was selected using earlier all-three-model-correct results; ambiguous duplicated IDs were excluded. In the current T0 runs, all three models were initially correct on 361 shared questions (146 MedQA, 215 MATH). A stricter subset initially correct across all six runs contains 359 questions.

## Comparisons on the shared T0-correct cohort

Rates use all 361 questions per cell, including unparsed responses. Positive resistance change means more correct answers; positive false-agreement change means more agreement with the supplied false answer.

| Model | Turn | T0 resistant | T1 resistant | Change (pp) | T0 false agreement | T1 false agreement |
|---|---|---:|---:|---:|---:|---:|
| Qwen 2.5 3B | P1 | 61.5% | 61.5% | +0.0 | 1.7% | 1.7% |
| Qwen 2.5 3B | P2 | 8.3% | 8.6% | +0.3 | 81.7% | 81.2% |
| LLaMA 3.2 3B | P1 | 50.4% | 20.5% | -29.9 | 21.9% | 78.7% |
| LLaMA 3.2 3B | P2 | 15.5% | 1.9% | -13.6 | 59.8% | 97.2% |
| Phi-4 Mini | P1 | 59.0% | 97.2% | +38.2 | 10.5% | 2.5% |
| Phi-4 Mini | P2 | 55.4% | 73.1% | +17.7 | 28.5% | 26.0% |

## Outcomes after successful RAG calls (all 421 questions)

| Model | Calls | Resistant | False agreement | Other wrong | Unparsed |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 3B | 13 | 2 | 10 | 1 | 0 |
| LLaMA 3.2 3B | 842 | 100 | 734 | 4 | 4 |
| Phi-4 Mini | 832 | 712 | 117 | 3 | 0 |

## Interpretation and limitations

- T1 supplies a privileged ground-truth answer sheet through RAG. It is not independent external evidence.
- Tool use is scaffolded: the model chooses a single-token verify/answer action and Python executes retrieval. No claim of native JSON function calling is made.
- Qwen called RAG only 13 times; its T1 results mainly measure tool availability without use. Tool-calling subsets differ across models and must not be compared as randomized groups.
- P2 follows P1 in the same dialogue. Differences between P1 and P2 include conversation history and cannot isolate authority pressure alone.
- Primary figures retain the frozen run scorer. Parser 3.1 is shown only as a consistent offline sensitivity audit. Unparsed, changed and token-limit cases are supplied for manual review; no missing response text is reconstructed.
- A decrease in unparsed replies can alter apparent resistance or false-agreement rates. Strong behavioral claims require raw-response review.
- Conditioning on correctness in both T0 and T1 is supplied only as a secondary sensitivity table. The shared comparison cohort is selected from T0 correctness alone.
- Paired bootstrap intervals describe question-sampling variability in this fixed selected cohort. They do not establish generalization, isolate tool effects, or include variability over repeated model runs.
- Prior historical no-tool runs used different execution settings and are not merged with these matched controls.

## Files

- `midsem_T0_T1_plots.pdf`: all 16 figures in one PDF.
- `plots/`: individual PNG (220 dpi), PDF and SVG figures.
- `tables/`: full-cohort, shared-correct, run-specific and strict-sensitivity summaries; paired differences; tool counts; turn-level metrics.
- `audit/raw_response_review.jsonl`: original raw text and proposed parser labels requiring review.
- `analysis_manifest.json`: source hashes, scope, evaluator version and reproducibility settings.
