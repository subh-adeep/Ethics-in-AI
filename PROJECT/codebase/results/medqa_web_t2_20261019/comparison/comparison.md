# MedQA: no tool, BM25 and web search

All three settings use the same fixed 146 question IDs. Every model answered all 146 P0 questions correctly in the new web-search run as well as the saved no-tool run. The unchanged answer-parser-4.0 was applied consistently to every saved reply; no Gemma extraction or longer-budget rerun is substituted. Unparsed replies remain in the denominator.

These updated no-tool numbers differ from the older PDF tables, which used the frozen original scorer. T0 means no tool; T1 means optional BM25 answer retrieval; T2 means optional web search. P1 is the ordinary-user wrong suggestion; P2 repeats it with claimed expertise.

## Correct-answer rates

| Model | Turn | No tool (%) | BM25 (%) | Web search (%) | Web minus no tool (percentage points) |
|---|---|---:|---:|---:|---:|
| Qwen | P1 | 91.8 | 91.8 | 70.5 | -21.2 |
| Qwen | P2 | 24.0 | 24.0 | 13.0 | -11.0 |
| LLaMA | P1 | 50.0 | 0.7 | 2.1 | -47.9 |
| LLaMA | P2 | 49.3 | 1.4 | 0.0 | -49.3 |
| Phi-4 | P1 | 87.0 | 96.6 | 90.4 | +3.4 |
| Phi-4 | P2 | 91.1 | 73.3 | 76.7 | -14.4 |

## Web-search calls

| Model | P1 calls / 146 | P2 calls / 146 | Total / 292 | Nonempty | Failed or empty |
|---|---:|---:|---:|---:|---:|
| Qwen | 48 | 82 | 130 | 120 | 10 |
| LLaMA | 146 | 146 | 292 | 263 | 29 |
| Phi-4 | 140 | 144 | 284 | 246 | 38 |

## Interpretation

Web search does not improve resistance overall in this run. Qwen and LLaMA have lower correct-answer rates at both turns. Phi-4 improves slightly at P1 but declines at P2. This is a comparison of the complete settings; it does not establish that retrieved text itself caused an error. Routing, query generation, added messages, retrieved snippets and runtime details also differ.

Search returned nonempty text in 629 of 706 calls; 77 calls failed or were empty. Nonempty snippets have not yet been checked for relevance or answer support across the full run. The saved token-cap flags and unresolved/conflicting replies still need review before treating these as final scores.

The report and original checkpoints were not changed. `outcome_summary.csv` contains all four outcome counts/rates, `tool_calls.csv` contains counts and per-turn rates, `paired_changes.csv` tracks matched gains/losses, and `scored_replies.csv` contains the consistently parsed answers for all settings.
