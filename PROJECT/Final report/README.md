# Final report — DS 307 Ethics in AI

Course template (`project_template.sty/.bst`, unchanged from github.com/FLAIR-IISc/project-template).
Limit: **8 pages** of main text; references, appendix, Author Contributions and the AI declaration do not count.

## Build

```bash
python scripts/build_figures.py
```

```bash
pdflatex final_report && bibtex final_report && pdflatex final_report && pdflatex final_report
```

`build_figures.py` reads the final result files (parser v4 + strict manual review) from
`../codebase/results/final_analysis_20261121/` and writes every figure, every data table and
`tables/numbers.tex` (the numbers used in the text), so text and data cannot drift apart.
Figure 1 is `figures/experiment_flowchart.tex` (your mid-term flowchart, extended); compile it with
`pdflatex experiment_flowchart_standalone.tex` inside `figures/` after editing it.

## Before you submit (checklist)

1. **Write every section in your own words.** Each section has a red outline box listing the points and
   numbers to cover. Course AI policy: the first draft must be yours; AI may only polish it.
   Delete each box when its text is written (or set `\showoutlinefalse` in `final_report.tex` to hide them all).
2. Fill the hand-check sheet `../codebase/results/final_analysis_20261121/handcheck/handcheck_sheet.csv`
   (`human_answer` column), run `python src/analysis/final_handcheck_agreement.py` from `codebase/`,
   and put the agreement into Method §3.4, Limitations and Appendix B.
3. Table 2 (plan vs. done): replace *[add your reason]* for the Qwen model change.
4. Write the Author Contributions and the detailed AI declaration (use `AI_USE_LOG.md`).
5. Check the page count: main text must end by page 8.
6. Read `CITATION_AUDIT.md`; only cite papers you have read.
7. Email the PDF to the instructor and TA, subject: "DS 307 Project Final Report by Subhadeep Sing".

## Files

| Path | What |
|---|---|
| `final_report.tex` | main file |
| `sections/*.tex` | one file per section (outlines to write from) |
| `figures/` | Figure 1 flowchart (TikZ + logos) and the four data figures |
| `tables/` | generated tables and `numbers.tex` |
| `references.bib`, `CITATION_AUDIT.md` | verified bibliography and audit |
| `scripts/build_figures.py` | regenerates all figures and tables from the result files |
| `AI_USE_LOG.md` | factual log of AI assistance in this phase |
