# Local GPU follow-up

Gemma and the source-quote checks recovered 11 answers from the 246 original unresolved replies. The 53 selected longer-budget turns supplied 38 additional readable answers. 197 cases remain unresolved in this separate follow-up.

T0: no tool. T1: optional BM25 answer retrieval. P1: an ordinary user's wrong suggestion. P2: the same wrong suggestion with claimed expertise. The fixed question cohort contains 146 MedQA and 215 MATH questions.

## Counts

```json
{
  "initial_unparsed": 246,
  "processed_records": 246,
  "not_yet_processed": 0,
  "unique_gemma_gpu_jobs": 84,
  "unique_gemma_followup_jobs": 12,
  "gemma_nonnull_answer_extractions": 8,
  "additional_original_answers_from_literal_quote_fields": 3,
  "supported_extractions": 11,
  "unresolved_after_extraction": 235,
  "selected_cap_rerun_turns": 53,
  "completed_rerun_turns": 53,
  "unique_rerun_contexts": 39,
  "rerun_parsed": 36,
  "additional_rerun_answers_from_gemma": 2,
  "remaining_after_original_reply_extraction": 235,
  "rerun_failed_turns": 0,
  "remaining_in_separate_followup": 197
}
```

## Remaining unparsed replies

In each cell: parser-v4 count → after Gemma extraction from the original saved reply → after the separate longer-budget follow-up.

| Dataset | Model | T0 P1 | T0 P2 | T1 P1 | T1 P2 |
|---|---|---:|---:|---:|---:|
| MedQA | Qwen | 2 → 1 → 0 | 19 → 15 → 10 | 2 → 1 → 0 | 19 → 15 → 10 |
| MedQA | LLaMA | 4 → 4 → 4 | 9 → 9 → 5 | 0 → 0 → 0 | 1 → 1 → 1 |
| MedQA | Phi-4 | 0 → 0 → 0 | 0 → 0 → 0 | 0 → 0 → 0 | 1 → 1 → 1 |
| MATH | Qwen | 11 → 11 → 7 | 0 → 0 → 0 | 11 → 11 → 7 | 0 → 0 → 0 |
| MATH | LLaMA | 20 → 19 → 14 | 11 → 11 → 3 | 0 → 0 → 0 | 0 → 0 → 0 |
| MATH | Phi-4 | 94 → 94 → 94 | 42 → 42 → 41 | 0 → 0 → 0 | 0 → 0 → 0 |

## Method and limits

Gemma sees no ground-truth or injected-wrong annotation. Identical extraction prompts are cached. Only parser-v4 unresolved replies are submitted. Every accepted suggestion must have a literal source quote; capitalisation and whitespace changes are allowed. These are LLM-assisted suggestions, not independent human labels.

When Gemma marks a quote supported but leaves its answer field null, a separate literal-quote check can recover an explicit option label or stated numerical result. A bare number quote and an intermediate calculation without a chosen answer are insufficient. `quote_field_repairs.json` records these repairs separately. The rerun selection remains the 53 turns chosen before this final quote-field check.

A bare Yes/No is not converted into a numerical or option answer. Missing, conflicting and unsupported choices stay unresolved.

Reruns are restricted to still-unresolved replies at the original 160-token cap. They use the original model and reconstructed saved conversation/evidence, with 320 then, only if still capped and unparsed, 640 output tokens. No new question, P0 turn or tool-selection step is run. Logged tool results and original earlier replies are replayed.

The longer-budget replies are separate observations: the local GPU/runtime/compute precision differ from the original T4 environment. They must not be substituted into the old primary results. Original checkpoints, v4 scores and report remain unchanged.

## Files

- `extraction_checkpoint.json`: exact prompts, model outputs, quotes and validation decisions.
- `extraction_results.csv`: one row for each of the 246 original unresolved replies.
- `stage_counts.csv`: all dataset/model/setting/pressure combinations, including zeros.
- `rerun_checkpoint.json`: selected contexts, generated token IDs, stopping reasons and parse results.
- `targeted_rerun_results.csv`: new longer-budget replies and whether each begins with the old saved reply.
- `followup_records.csv`: original extraction and separate rerun results joined for the 246 target records only.
- `remaining_unparsed.csv`: cases still unresolved after this follow-up.
- `rerun_extraction_checkpoint.json`: Gemma extraction from any new rerun replies still unparsed.
- `quote_field_repairs.json`: empty Gemma answer fields repaired from literal source assertions, without reference labels.
- `rerun_verification.json`: scope, saved contexts, CUDA-only inference and unchanged source/PDF hashes.
- `environment.json`: local Python and inference library versions.
- `verification.json`: final counts, prompt scope, source hashes and remaining-reply diagnostics.
- `context_replay_verification.json`: reconstruction checked against the frozen original runner.
- `provenance.json`: source hashes and local model snapshots.
