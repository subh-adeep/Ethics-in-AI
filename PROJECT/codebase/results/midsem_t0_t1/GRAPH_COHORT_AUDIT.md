# Graph question-set audit — 15 October 2026

## Which graphs use 361 questions?

The main comparisons use the same 361 question IDs across all three models, T0/T1 and P1/P2. The separate dataset comparisons use 146 MedQA and 215 MATH questions.

Combined main-comparison plots: 04_common_correct_outcomes, 05_resistance, 06_false_agreement, 07_pressure_sequence, 08_domain_comparison, 09_paired_differences, 13_paired_transitions.

Separate-dataset main-comparison plots: 03_shared_correct_outcomes, 04_resistance, 05_false_agreement, 06_pressure_sequence, 07_paired_differences.

## Which graphs use the full run?

Progress, initial accuracy, all-question outcome summaries, tool use, after-tool outcomes, evidence override, reply quality, reply lengths and parser checks use all 421 questions (171 MedQA and 250 MATH). These are supporting diagnostics, not the main initially-correct comparison.

Tool-use rate divides calls by two pressure-turn opportunities per question: 842 combined, 342 MedQA or 500 MATH. After-tool rates divide by successful calls; evidence override divides by successful target-answer retrievals. These call-based denominators are shown separately on the plots.

## Checks and corrections

Checked the question sets against the six frozen checkpoints, verified their SHA-256 hashes, and checked the shared IDs and table denominators. Main comparisons already used the correct subset. Added the actual question-set label to all 40 current plots and exported plot_question_sets.csv in each package. Numeric primary results and original checkpoints were retained.

The two report figures also use the shared subset: the combined figure uses 361; the separate figure uses 146 MedQA and 215 MATH.

The older results/figures directory contains historical no-tool figures; its progressive subfolder belongs to a separate, not-yet-validated experiment. A README now marks this distinction. They should not be used as current matched T0/T1 results.
