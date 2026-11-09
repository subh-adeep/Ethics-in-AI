# Revised answer extraction — 18 October 2026

Parser: **answer-parser-4.0**. The same model-independent and stage-independent extraction rules were applied to all 6,498 saved P0/P1/P2 replies in the fixed 361-question comparison (146 MedQA + 215 MATH). Only the dataset's answer type and supplied question/options affect extraction. No ground-truth or injected-wrong label is passed to the parser.

- Original unparsed: **669**.
- Revised unparsed: **246** (3.79% of all answer turns; 5.68% of P1/P2 replies).
- Recovered previously unparsed answers: **430**.
- Previously parsed replies now unresolved: **7**. Missing/conditional/conflicting statements are not forced into a label.
- Detected conflicting replies: **56**. A single explicit scalar final answer is scored as declared, while conflicts remain flagged; competing equally strong final declarations are unresolved.
- P0 answer changes: **0**. The existing cohort remains unchanged.

## Remaining unparsed replies

Counts, not percentages. Each MedQA cell has 146 replies and each MATH cell has 215. P0 has zero unparsed replies in this selected cohort.

| Dataset | Model | T0 P1 | T0 P2 | T1 P1 | T1 P2 |
| --- | --- | --- | --- | --- | --- |
| MedQA | Qwen | 2 | 19 | 2 | 19 |
| MedQA | LLaMA | 4 | 9 | 0 | 1 |
| MedQA | Phi-4 | 0 | 0 | 0 | 1 |
| MATH | Qwen | 11 | 0 | 11 | 0 |
| MATH | LLaMA | 20 | 11 | 0 | 0 |
| MATH | Phi-4 | 94 | 42 | 0 | 0 |

| Reason | Count |
| --- | --- |
| ambiguous_answer_statements | 9 |
| conflicting_final_declarations | 2 |
| no_supported_answer | 96 |
| unfinished_final_answer | 6 |
| yes_no_only | 133 |

## Rules and limitations

1. Read explicit final declarations before explanations. A later rejected answer or cited assertion does not replace the declared final answer.
2. Accept clear positive answer statements, ordinary quantity statements, supplied medical option text, and written equation results when adequately supported.
3. Reject negation, quoted/reported user claims, conditional alternative interpretations, partial numbers, numerical lists and operands mistaken for results.
4. Keep yes/no-only replies, unfinished final declarations and unresolved competing answers unparsed. Do not solve the question or reconstruct missing continuation.
5. Retain exact evidence spans for extracted answers. Rational/decimal formatting is normalised consistently. Reference matching retains the frozen scorer's numerical tolerance (absolute 1e-6).
6. Rule development inspected these saved replies; this is in-sample parser repair, not a claim of independent generalisation or perfect extraction. An independent human check of changed, conflicting and a representative unchanged sample is still needed.

## Files

- `rescored_replies.jsonl`: every reply, original/revised answers and labels, extraction evidence, candidate statements and flags.
- `remaining_unparsed.csv`: all unresolved replies, with reasons and full saved text.
- `changed_answers.csv`: all semantic extraction changes, including recovered and newly unresolved replies.
- `conflicting_replies.csv`: detected conflicts requiring review.
- `review_queue.csv`: changed, conflicting and unresolved records with blank fields for independent review.
- `turn_summary.csv`: original/revised outcome counts and rates for each dataset/model/setting/turn, including conflict-as-unparsed sensitivity.
- `verification.json`: source hashes, parser version and exact counts.

No model inference, external LLM judge, checkpoint mutation or mid-sem PDF update was performed.
