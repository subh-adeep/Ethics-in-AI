# Final report — DS 307 Ethics in AI

**Can Tools Help Small Language Models Stay Correct Under User Pressure?** — Subhadeep Sing

The folder follows the course template (github.com/FLAIR-IISc/project-template): one main `.tex` file,
one file per section in `sections/`, and separate `figures/` and `tables/` folders.
Page limit: **8 pages** of main text. References, the appendix, Author Contributions and the AI declaration
do not count. The current PDF has its main text on pages 1–8.

## Build

```bash
python scripts/build_figures.py      # only if the result files changed (needs matplotlib)
make                                 # or: pdflatex, bibtex, pdflatex, pdflatex on final_report_submission
make clean                           # remove LaTeX build files
```

`build_figures.py` reads the final result files in `../codebase/results/final_analysis_20261121/`
(parser v4 + strict AI-agent review) and writes every data figure, every data table and
`tables/numbers.tex`, the macros for the numbers quoted in the text, so text and data cannot drift apart.
Figure 1 is `figures/experiment_flowchart.tex` (the mid-term flowchart, extended); after editing it,
compile `figures/experiment_flowchart_standalone.tex` inside `figures/`.

## Files

| Path | What it is |
|---|---|
| `final_report_submission.tex` / `.pdf` | main file and the submitted PDF |
| `sections/` | `abstract`, `intro`, `related`, `methods`, `results`, `discussion`, `conclusion`, `appendix` |
| `figures/` | Figure 1 flowchart (TikZ + logos) and the data figures |
| `tables/` | generated tables and `numbers.tex` |
| `references.bib` | the 28 cited papers (checked against their sources) |
| `project_template.sty`, `project_template.bst` | course template style files (unchanged) |
| `Makefile` | build commands, same targets as the course template |
| `scripts/build_figures.py` | rebuilds all figures, tables and number macros from the result files |

## Before you submit

1. Make sure the GitHub repository linked in Author Contributions is shared with the instructor and TA.
2. Read the AI declaration once more and check that it describes your work correctly.
3. Email the PDF to the instructor and TA, subject: "DS 307 Project Final Report by Subhadeep Sing".
