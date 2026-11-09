# Web-search pilot: 10 MedQA + 10 MATH questions

Status: complete. All three models finished the fixed sample. See [pilot_findings.md](<D:/collage/7th sem/Ethics in ai/PROJECT/codebase/results/web_search_pilot_20261018/pilot_findings.md>) for the results and remaining issues.

The sample is fixed before inference (seed 20261018). All three models receive the same 20 questions from the existing shared initially-correct cohort. Each conversation generates a fresh neutral answer, then the original ordinary-user and claimed-expert challenges. Tool use remains optional and is chosen by comparing the scores for `1` and `2`. Answer budget: 160 new tokens; search-query budget: 64 new tokens. Original data and the submitted report are not edited.

The search tool returns up to three **search-result snippets**, with their titles and URLs. It does not download full web pages. A nonempty HTTP(S) URL and snippet are required for operational success. Operational success does not establish relevance, factual reliability, or correct use by the model.

## Saved files

- `manifest.json`: the 20 fixed questions, original prompts and scoring references. References are for evaluation only and are not passed to query generation or the tool observation.
- `protocol.json`, `environment.json`, `implementation_hashes.json`: settings, local model/runtime differences and source versions.
- `preflight.json`: a separate search check using controller question-text queries for all 20 questions; these are not model-selected tool calls.
- `preflight_relevance_review.json`: AI inspection of the saved preflight snippets, not a human review or a full-page fact check.
- `search_events.jsonl`: every search query, timestamp, model/turn, raw provider results, rejected entries, URLs, snippets, cache use, unsuccessful attempts and errors.
- `*_checkpoint.json`: model replies and tool records, saved after each answer turn; safe to resume.
- `*_model_calls.jsonl`: exact messages sent to local inference, action probabilities, query generations, replies, token counts and token-limit flags.
- `runtime_errors.jsonl`: runtime failures and recoveries, if any occur. Missing file means no such error was recorded.
- `delivery_verification.json`: verifies that each logged tool observation appears unchanged in the actual generation input; checks retained history and original-file hashes.
- `retrieved_documents.csv` / `.json`, `search_records.json`, `summary.json`: readable exports, generated from the saved logs.

## Resume or export

Run from the workspace using the existing Miniconda Python:

```powershell
C:/Users/subha/Miniconda3/python.exe PROJECT/codebase/src/analysis/web_search_pilot.py --phase models
C:/Users/subha/Miniconda3/python.exe PROJECT/codebase/src/analysis/review_web_search_pilot.py
```

The runner skips completed turns. It does not reselect the sample when a fresh local baseline differs from the earlier run. Parser scores are provisional; the pending review of unclear answers is outside this pilot.

## Completed pilot

All 60 conversations are saved. Qwen and LLaMA ran on the laptop; Phi-4 ran on Kaggle. There were 93 completed tool calls: 90 returned nonempty snippets and three failed or were empty. Exact tool-output delivery was checked for all 93 calls. Relevance was mixed; see the findings and source review. The submitted report and original results are unchanged.

The pilot includes documented repairs and hardware/attention differences, so it is not a final model-performance comparison. `phi4_search_events.jsonl` holds the cloud search log separately from the local log.
