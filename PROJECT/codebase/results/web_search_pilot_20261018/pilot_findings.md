# Web-search pilot findings

Completed the same 10 MedQA and 10 MATH questions with all three models: 60 conversations and 180 final answer turns. Qwen and LLaMA ran on the laptop GTX 1650; Phi-4 ran on a private Kaggle Tesla T4.

| Model | Tool calls | Nonempty results | Failed / empty |
|---|---:|---:|---:|
| Qwen | 20 | 20 | 0 |
| LLaMA | 34 | 32 | 2 |
| Phi-4 | 39 | 38 | 1 |

Each model had 40 opportunities to search: 20 questions at P1 and again at P2. Tool use was optional. These counts are completed-turn tool calls; repeated queries may use a saved result. They are not counts of correct answers or relevant sources.

## What passed

- All 93 tool outputs, including the three failures, were found unchanged in the saved input messages for the corresponding answer generation. Nonempty retrieved text was delivered in 90 calls.
- Every selected question received nonempty evidence in at least one model conversation. Evidence returned at P1 remained in the P2 input.
- Queries, original and cleaned query text, URLs, titles, complete provider snippets, raw results, rejected blank entries, timestamps, model replies and errors are saved.
- The fixed sample and original prompts were retained. The original result files and submitted report passed checksum checks.

## What still needs work

1. **Query problems:** Qwen changed 625681 to `62568野` in one query. It also dropped the urease breath-test clue from another query. LLaMA added unsupported calcium/PTHrP or tophaceous-gout details in two cases and reversed a base-conversion direction.
2. **Incomplete snippets:** Many MATH results are calculator landing pages. A matching title does not mean the snippet includes an answer or a usable solution. Several medical snippets are also too general.
3. **Inconsistent returned text:** One 5-choose-1 result describes choosing from 8 distinct elements. The returned snippet itself is inconsistent; the linked full page was not fetched or independently verified.
4. **Search failures:** LLaMA had two failed/empty turns on its first HIV query before quote cleanup. Phi-4 had one failed floor-function search; the same query succeeded at the next turn. These failures remain in the primary pilot records.

Useful examples include snippets explicitly naming PCR for infant HIV testing, elastase in alpha-1-antitrypsin deficiency, H. pylori with urea breath testing, and the GCD answer 56. Some other results supply a method without the numerical answer. The 55 distinct completed query bundles were reviewed by the AI assistant; these annotations are not human-validated retrieval-accuracy scores.

## Repairs and scope

Blank URL/text records are rejected. Balanced outer quotation marks are removed from generated search queries, with both versions logged. One Windows checkpoint-save error was recovered from a valid temporary file; checkpoint writes now retry sharing violations. The first cloud attempt failed during package import before model inference and was repaired. Local attention changed from eager to SDPA during the pilot, with prior turns retained and execution details recorded. Phi-4 used the same pinned model revision on Kaggle.

This is an iterative pipeline pilot, not a final comparison of model accuracy. It returns search-result snippets, not full downloaded web pages. Correctness scoring remains provisional and the pending parser work was not reopened.

## Next step

Before a full T2 run, make queries preserve the question’s numbers, conversion direction and key clues; avoid adding unsupported case details. Improve evidence extraction so the model receives useful passages rather than only calculator descriptions. Test these repairs on this same fixed 20-question sample before expanding the run.

## Main files

- `completed_tool_calls.csv`: each completed model-selected search call and its status.
- `retrieved_documents.csv` / `.json`: URLs and retrieved snippets, with phase/model/question/turn fields.
- `search_events.jsonl`, `phi4_search_events.jsonl`: complete local/cloud search events, including raw results, attempts, rejections and errors.
- `actual_relevance_review.json`: the AI source review and its limitations.
- `completed_search_failures.json`: the three failed/empty completed tool turns.
- `*_checkpoint.json`, `*_model_calls.jsonl`: answers, exact generation messages, routing probabilities and budgets.
- `final_verification.json`, `delivery_verification.json`: final counts and delivery/checksum checks.
- `protocol.json`, `implementation_revisions.json`, `cloud_environment.json`: settings, runtime changes and cloud details.
- `kaggle_download_attempt1/`: the retained cloud setup-error log; `kaggle_download_final_v3/`: complete downloaded cloud artifacts.
