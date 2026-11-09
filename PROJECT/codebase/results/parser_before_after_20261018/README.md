# Before versus now: answer scoring

The two-page landscape PDF compares the same report tables and BM25 tool-call graphs side by side: MedQA on page 1 and MATH on page 2. Left = original frozen-run scoring; right = answer-parser-4.0 scoring. Changed numeric cells are blue. The fixed cohort is 361 questions (146 MedQA + 215 MATH); no model inference was rerun.

Outcome tables use all selected questions, including unparsed replies. Flip rates require that the same run's P0 answer was correct. LLaMA T1 MATH therefore retains 213 eligible questions for flips, while its outcome table uses all 215. P2 flips are relative to P0 and include cases that already flipped at P1. Tool calls are identical before/now because extraction cannot change recorded actions.

Files: `output/pdf/parser_before_after.pdf`, `figures/medqa_before_after.png`, `figures/math_before_after.png`, three comparison CSV files, and eight reusable LaTeX tabular snippets under `tables/` (requires `booktabs`). The LaTeX snippets are table bodies; add your own caption/table wrapper.

Revised scoring remains subject to independent review. Original checkpoints, report tables and the mid-sem PDF were not changed.
