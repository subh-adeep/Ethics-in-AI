# Full exact-ID single-entry lookup experiment

Private Kaggle job: https://www.kaggle.com/code/subhadeepsing/exact-answer-lookup-full-20261020

The fixed cohort contains 146 MedQA + 215 numerical Level 1/2 MATH questions, tested with Qwen, LLaMA and Phi-4. The target is 1,083 conversations and 3,249 final answer turns, with 2,166 opportunities to call the tool at P1/P2. Fresh P0 replies do not redefine the cohort.

This retains the passed smoke-test condition: optional single-token 1/2 tool selection; the original P0/P1/P2 prompts; the same pinned model revisions and 160-token answer budget; exactly one reference entry fetched using the current question ID. MedQA provides the option and answer text. MATH provides the numerical answer and original solution. No BM25 ranking or web search runs in this condition. The full run starts new checkpoints rather than replacing or reusing the smoke observations.

Two isolated workers each receive 73 MedQA questions. GPU 0 also receives 108 MATH questions (181 total); GPU 1 receives 107 (180 total). Each worker starts with one MedQA and one MATH question, then processes the rest. Both workers run all three models sequentially on their assigned question IDs. Every reply is checkpointed; exact input messages, routing choices, returned reference entries and execution metadata are logged. Missing IDs or ID/question mismatches cause a failure rather than unrelated evidence. At completion, each actual lookup is verified against the reference and actual generation input, P1 evidence retention at P2 is checked, and the shards are merged with no missing or repeated IDs. A worker failure stops its sibling.

Qwen made no calls in the smoke sample. Skipping the tool remains a valid outcome in this optional condition; no evidence is inserted when a model skips. This run does not measure required evidence. It also changes both entry selection and entry count relative to BM25 with two entries; a separate BM25 top-one experiment would isolate entry count.

The submitted report, old checkpoints, parser and completed smoke/web verification artifacts are protected by source_guards.json. Six local tests passed before submission. The shared runner gained only an optional workload label for truthful execution metadata; the default smoke behaviour is unchanged. Previous smoke numerical outputs were not changed.

See manifest.json and protocol.json for the fixed data and settings; kaggle_launch.json for the submitted version; bundle_verification.json for source hashes. Cloud outputs contain per-GPU checkpoints and logs, plus combined per-model checkpoints and cloud_completed.json after all conversations finish. Correctness scoring remains provisional until the saved outputs are reviewed.
