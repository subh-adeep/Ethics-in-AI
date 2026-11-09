# MedQA web-search run on two Kaggle GPUs

Private job: https://www.kaggle.com/code/subhadeepsing/medqa-web-search-dual-gpu-20261019

This new T2 run uses all 146 MedQA questions in the established shared initially-correct cohort, with Qwen, LLaMA and Phi-4. MATH is excluded. Each GPU receives a disjoint half of 73 questions and runs all three models. The complete target is 438 conversations and 1,314 final answer turns. There are 876 opportunities to search at P1/P2; search remains optional.

The original question and pressure prompts, separate 1/2 action selection, 160-token answer budget and 64-token query budget are retained. P0 is generated again, but the cohort is not selected again using those new answers. Model revisions are pinned. Answers and complete model inputs are checkpointed after every turn.

Before the full run, both GPUs must pass connectivity and generation checks on two questions per model. The controller verifies that actual logged generation inputs contain the retrieved observation and exclude hidden evaluation annotations. Successful smoke answers are reused when the full run continues. The controller stops both workers if either fails. A one-GPU fallback is not allowed.

The previous 10+10 pilot passed search/delivery checks, with mixed source relevance. This deployment retains that pilot's strict search adapter and query prompt. It is exploratory: generated queries may lose clues or add unsupported details, and snippets may be incomplete. A nonempty search result does not prove it answers the question. Full webpages are not fetched. Accuracy and relevance need review after the run, and unclear/conflicting-answer scoring remains provisional.

`manifest.json` records the fixed cohort and source hashes. `protocol.json` records settings and limits. `kaggle_launch.json` records the submitted version. `latest_status.json` and `monitoring/` contain status snapshots taken during deployment. Within the cloud output, `gpu0/` and `gpu1/` contain exact model calls, search events, URLs, returned snippets, raw provider results, errors and per-model checkpoints. `smoke_passed.json` is written only after both GPU checks pass; `cloud_completed.json` is written only after all 438 conversations finish and both halves are merged without missing or repeated IDs.

The submitted report, existing parser, completed pilot and original T0/T1 results are not edited by this run. `source_guards.json` records checksums of protected files.

## Deployment verified

Version 1 never received a worker and was cancelled before an identical-source version 2 was submitted. The obsolete queued version 2 of the completed Phi-4 pilot was also cancelled. Version 2 of this MedQA job is running. Verification used the visible Kaggle browser logs: two Tesla T4 workers, both smoke gates passed, 20/20 smoke search outputs delivered unchanged into actual generation inputs, and full-run questions completing beyond the smoke sample on both GPUs. This confirms operation at the time of the check; the full run is still in progress and accuracy has not been established. See `deployment_verification.json`.

## Completed

The full run completed at 3:22 AM IST on 19 October 2026. All 438 conversations and 1,314 answer turns were downloaded and checked. See `completion_summary.md` and `final_verification.json` for counts and search failures.
