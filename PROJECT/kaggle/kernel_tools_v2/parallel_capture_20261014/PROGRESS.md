# Preserved Kaggle experiment progress

Remote run was manually stopped. All nine checkpoints parsed and passed cohort, source, settings, count and metric consistency checks.

Completed: **3036/3789 dialogues**. Remaining: **753**.

| Model | T0 | T1 | T2 |
|---|---:|---:|---:|
| qwen | 421/421 | 421/421 | 421/421 |
| llama | 421/421 | 421/421 | 86/421 |
| phi4 | 421/421 | 421/421 | 3/421 |

## T1 outcomes after successful RAG calls

Stored scorer labels; no offline rescoring has been applied. These are descriptive counts, not a causal comparison with T0.

| Model | Calls | Resistant | Sycophantic | Other wrong | Unparsed |
|---|---:|---:|---:|---:|---:|
| qwen | 13 | 2 | 10 | 1 | 0 |
| llama | 842 | 100 | 734 | 4 | 4 |
| phi4 | 832 | 712 | 117 | 3 | 0 |

The original raw responses, tool records, web caches, health checks, manifests and inference source are preserved in this directory.
Do not rerun the old upload: a future resume must use these latest checkpoints.
workers.json records launcher state before interruption; its active process IDs do not mean the cancelled cloud job is still running.
