# Exact-answer lookup smoke test

Scope: 10 MedQA + 10 MATH questions, the same IDs used in the earlier web pilot, with all three pinned models. This private two-GPU Kaggle job stops after 60 conversations and 180 answer turns.

Kaggle: https://www.kaggle.com/code/subhadeepsing/exact-answer-lookup-smoke-20261019

The original prompts, 1/2 action selection, optional tool policy and 160-token answer budget are retained. Python uses the current question ID to fetch one exact matching reference entry. The tool does no BM25 ranking and generates no search query. MedQA supplies the correct option plus answer text; MATH supplies the answer and the original worked solution. Reference answers are deliberately provided as experimental evidence, while injected-wrong scoring fields and supports_ground_truth annotations are kept out of model inputs.

Evidence is supplied only if the model chooses the tool. Each call is checked for one entry, matching ID/question, correct stored answer, exact solution text and delivery into the actual model generation input. Evidence returned at P1 must remain in the input at P2. Missing IDs or mismatched questions stop the lookup rather than falling back to an unrelated entry. Checkpoints are saved after every answer.

This combines two changes relative to the old BM25 condition: one entry rather than two, and exact ID selection rather than ranking. Results cannot identify which change is responsible. A BM25 top-one condition can test that separately. The smoke sample is small and scores remain provisional.

The old report, original checkpoints, parser and completed web-search results are protected by hashes. Four local tests passed before submission, covering identity checks, the optional skip path, actual evidence delivery, history retention and checkpoint reuse. Final downloaded results will be verified and summarised in smoke_findings.md. No full-cohort experiment is automatically launched.

## Completed

Both GPU shards finished successfully in 543.7 seconds. All 60 conversations and 180 answer turns were downloaded and checked. All 78 actual lookups delivered the one correct matching entry. Qwen chose no tool calls; LLaMA made 40, and Phi-4 made 38. See smoke_findings.md and final_verification.json. The full run has not started.
