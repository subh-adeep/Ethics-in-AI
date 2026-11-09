# Exact-ID single-entry smoke test

Completed 10 MedQA + 10 MATH questions for Qwen, LLaMA and Phi-4: 60 conversations and 180 answer turns. Tool choice remained optional. The controller used the question ID to return exactly one matching answer-sheet entry. MedQA retained the correct option and answer text; MATH retained the numerical answer and original solution.

All 78 model-selected lookups returned the correct matching entry, and its exact text was found in the actual logged generation input. P1 evidence remained available at P2. There were zero lookup failures. No web search or BM25 ranking ran in this condition.

## MEDQA

Counts out of 10; each question is 10 percentage points. The same parser-v4 scores original replies in every setting; separate longer-budget reruns are excluded.

| Model | Turn | No-tool correct | BM25 two-entry correct | Exact-ID one-entry correct | Exact-ID calls | Exact-ID user wrong | Exact-ID unparsed |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | P1 | 9 | 9 | 9 | 0 | 1 | 0 |
| qwen | P2 | 2 | 2 | 1 | 0 | 9 | 0 |
| llama | P1 | 8 | 0 | 0 | 10 | 10 | 0 |
| llama | P2 | 8 | 0 | 0 | 10 | 10 | 0 |
| phi4 | P1 | 10 | 10 | 10 | 8 | 0 | 0 |
| phi4 | P2 | 10 | 7 | 10 | 10 | 0 | 0 |

## MATH

Counts out of 10; each question is 10 percentage points. The same parser-v4 scores original replies in every setting; separate longer-budget reruns are excluded.

| Model | Turn | No-tool correct | BM25 two-entry correct | Exact-ID one-entry correct | Exact-ID calls | Exact-ID user wrong | Exact-ID unparsed |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | P1 | 10 | 10 | 10 | 0 | 0 | 0 |
| qwen | P2 | 0 | 0 | 0 | 0 | 10 | 0 |
| llama | P1 | 9 | 4 | 6 | 10 | 4 | 0 |
| llama | P2 | 3 | 0 | 3 | 10 | 7 | 0 |
| phi4 | P1 | 1 | 9 | 10 | 10 | 0 | 0 |
| phi4 | P2 | 0 | 10 | 10 | 10 | 0 | 0 |

## Limits

Fresh P0 correct counts out of 10: {"medqa/qwen": 10, "medqa/llama": 10, "medqa/phi4": 10, "math/qwen": 10, "math/llama": 10, "math/phi4": 10}.
This smoke sample checks that the mechanism works; it is too small for a final performance claim. Exact-ID one entry changes both entry selection and the number of entries compared with BM25 two entries. A separate BM25 top-one condition would isolate the number-of-entries effect. Optional skips are valid and receive no evidence. Unclear, conflicting and token-capped replies remain flagged.

The full experiment was not started. Source/checkpoint hashes and delivery checks are in `final_verification.json`; detailed scores are in `outcome_summary.csv`.
