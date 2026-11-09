# Full exact-ID single-entry lookup experiment

Private Kaggle job: https://www.kaggle.com/code/subhadeepsing/exact-answer-lookup-full-20261020

Active scheduling: version 2. GPU 0 runs Qwen; GPU 1 runs LLaMA. Phi-4 starts on the first GPU that becomes free. Two different models run in parallel. Every model receives the complete 361-question cohort: 146 MedQA + 215 numerical Level 1/2 MATH. The target is 1,083 conversations, 3,249 answer turns and 2,166 opportunities to call the tool at P1/P2.

The initial question-splitting version 1 was cancelled when the user requested one model per GPU. Cancellation was acknowledged before version 2 was submitted. Its source, protocol and submission record are preserved in scheduling_v1_cancelled/. Those interrupted observations are not merged into version 2.

The experiment retains the passed smoke-test condition: optional single-token 1/2 action selection; original P0/P1/P2 prompts; the same pinned model revisions; a 160-token answer budget; and exactly one reference entry fetched by question ID. MedQA supplies the correct option and answer text. MATH supplies the numerical answer and original worked solution. There is no BM25 ranking or web search. P0 is generated again without reselecting the fixed cohort. The full run uses fresh checkpoints rather than replacing or reusing smoke observations.

Each model starts with one MedQA and one MATH question, followed by the remaining question IDs. Every reply is checkpointed. Model folders qwen/, llama/ and phi4/ hold the per-model checkpoint, exact generation messages, lookup events, returned entries and GPU metadata. workers.json records active GPU/model pairs, the queued model and completed models. A worker failure stops the other active worker. Once all three finish, every lookup is checked against the reference and actual generation input, P1 evidence retention at P2 is verified, and top-level combined checkpoints are saved with no missing or repeated IDs. cloud_completed.json is written only after all 1,083 conversations finish.

Skipping the tool is valid in this optional condition; no evidence is inserted on skipped calls. Qwen skipped all calls in the smoke sample. This run does not test required evidence. Compared with BM25 two-entry retrieval, exact ID lookup changes both selection and entry count, so a separate BM25 top-one condition would isolate entry count.

The submitted report, parser and completed smoke/web results are protected by source_guards.json. Six local tests passed for the fixed cohort, unchanged prompts/settings, lookup correctness, optional skips, checkpoint reuse and actual evidence delivery. The shared engine gained optional workload metadata and single-model validation; its default smoke behaviour remains unchanged. Prior saved results were not edited.

See manifest.json, protocol.json, kaggle_launch.json, bundle_verification.json and deployment_verification.json for the data, settings, submitted version, hashes and deployment checks. Scoring remains provisional pending output review.
