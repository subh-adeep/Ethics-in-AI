# Edited mid-semester report — 15 October 2026

The author supplied the latest draft. This revision keeps simple English, corrects grammar and incomplete sentences, checks technical claims, adds citations and future work, and uses the professor's unchanged style and bibliography style.

The main report is six pages. References, Author Contributions, the AI Assistance Declaration and appendices are outside that limit. Total: ten pages. There are exactly six tables and no graphs, as requested. Graphs may be considered later.

Edit midsem_report.tex. The six table inputs are named medqa/math_outcomes_table.tex, medqa/math_pressure_tool_calls.tex and medqa/math_flip_table.tex. The original supplied draft is preserved under student_source/ in the source package and in the parent project directory. The earlier expanded version is preserved outside this draft folder in previous_expanded_20261015/.

Build from this folder:

    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex
    bibtex midsem_report
    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex
    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex

The original build_draft.py is a legacy auto-generator; edit the .tex directly instead. No new model experiment was run for this revision. Original checkpoint replies and scores remain unchanged. The declaration describes substantial earlier AI help and the editing of the supplied draft; the author must review the report before submission.
