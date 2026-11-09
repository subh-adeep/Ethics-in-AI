# MedQA web-search run completed

Completed at 3:22 AM IST on 19 October 2026. Kaggle reports successful execution in 2 hours 31 minutes on two Tesla T4 GPUs.

| Model | Questions completed | Answer turns | Search calls | Nonempty results | Failed / empty |
|---|---:|---:|---:|---:|---:|
| qwen | 146/146 | 438 | 130 | 120 | 10 |
| llama | 146/146 | 438 | 292 | 263 | 29 |
| phi4 | 146/146 | 438 | 284 | 246 | 38 |

All 438 conversations and 1,314 answer turns are saved locally. Each model has the exact fixed 146 MedQA IDs, with no missing or repeated question. All 706 tool observations, including failures, were rechecked against the actual generation inputs. Of these calls, 629 returned nonempty snippets and 77 failed or returned no usable text. Nonempty text is not a relevance or correctness score.

The correctness comparison is not yet reviewed. Some replies reached the unchanged 160-token limit (Qwen 68, LLaMA 82, Phi-4 89 across P0/P1/P2), so their saved truncation flags should be included in the scoring review. Existing report and parser checksums remain unchanged.

Full downloaded files: `kaggle_download_final_v2/medqa_project/PROJECT/codebase/results/medqa_web_t2_20261019/`. Verification: `final_verification.json`.
