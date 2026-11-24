# AI-use log for the final phase (input for the "Declaration about AI Assistance")

This is a factual list of what an AI assistant did in the final phase (21 November 2026), so you can write
the detailed disclosure the course asks for. Add the earlier phases from your own records and the
mid-term declaration. **Write the final disclosure text yourself.**

Tool used: Claude Code (Anthropic), model Claude Opus 5.5, in the Claude desktop app; plus helper
sessions/agents started from it.

## What the AI did

| # | Task | Where the output is | What you should check |
|---|---|---|---|
| 1 | Compared the proposal and mid-term promises with the saved results; listed what was done, missing or changed | conversation; `../../FINAL_RESULTS_README.md` | that the list matches what you remember |
| 2 | Wrote `FINAL_RESULTS_README.md` (all experiments in order, with numbers from the result files) | repo root | numbers against the result folders |
| 3 | Wrote analysis code: rescoring Exact (MATH) and Python replies with your parser v4, paired McNemar tests with bootstrap CIs and Holm correction, the metrics table, the hand-check sheet | `codebase/src/analysis/final_*.py`, `codebase/results/final_analysis_20261121/` | run the scripts again; read the code |
| 4 | **Read all 365 unparsed replies by hand and decided whether each contained an answer** (first reading: 109 recovered; strict re-check: 64 kept). The decisions were made by the AI assistant, not by you | `final_analysis_20261121/manual_unparsed_review/`, `manual_unparsed_review_strict/` (each decision has the quote and reason) | spot-check decisions in `strict_recheck.csv` |
| 5 | Built the report figures and data tables from the result files (heatmap, forest plot, tool-use/override bars, Python bars, appendix tables, probe table) | `Final report/scripts/build_figures.py`, `figures/`, `tables/` | figures against the CSVs |
| 6 | Adapted your mid-term flowchart: added the Python track (saved answers of question `math_l12_0505`), the MedQA-only note and the final scoring box; made the layout more compact | `figures/experiment_flowchart.tex` | the example answers |
| 7 | Set up the report in the course template: section files, figure/table placement, figure and table captions, Table 1 (conditions) and Table 2 (plan vs. done) contents, appendix structure, the prompts appendix (copied verbatim from the run scripts), and **point-by-point outlines** of what to write in each section | `Final report/final_report.tex`, `sections/` | rewrite every outline box in your own words; check Table 2 reasons |
| 8 | Checked every citation against its source page (an AI sub-agent with web search) and wrote the BibTeX file | `references.bib`, `CITATION_AUDIT.md` | open the papers you cite and read them |
| 9 | Found and corrected a wrong statement in the earlier notes: the Python condition **did** use the 1/2 tool menu (checked in `kaggle/kernel_math_*_215q/run_*_215.py`); confirmed the labels are not used in prompts or tool inputs | report Method section | — |

## Your prompts in this phase (in your words)

- "check are we ready or not for our final pdf … what we promise … are we ready to write report or some experiment left"
- "explain me each task what to do in simple language then I will decide"
- "create a final read me with organised result which we will put into our final report … each step in sequence"
- "start a new session and there we should do all of this thing"
- "check all the none cases manually and if there is some answer extract it … create a new table, do not overwrite the old table"
- "Yes, re-check them" (strict re-check of the recovered answers)
- "by seeing mid sem and seeing proposal file and check the prof git repo what should be the format, create a final submission file … mention all the results … if we promised any extra experiment and did not do it give justification … use graph, table, diagram … perfect citation"
- "in midterm report I already have a diagram you can check" (the flowchart was then reused and extended)

## Not done by the AI (fill in)

- The research question, the choice of models, datasets and tool conditions, running the experiments, … *(your list)*
- The report paragraphs (to be written by you from the outlines).
