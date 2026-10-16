# Report edited from the new rough draft

Input: the author's attachment b2f8d83a-35e0-464d-b0c0-e384b0956179/Pasted text.txt, supplied on 15 October 2026. The supplied source is preserved as rough_original.tex. Earlier report drafts were not opened or used to construct this revision.

The report uses the professor's unchanged project_template.sty and project_template.bst. It has six main pages plus one exempt page containing contributions, AI assistance and references. It contains four tables, one experiment flowchart and two bar graphs for tool calls (MedQA and MATH). The flowchart was added at the author's request on 16 October 2026; it appears on page 2. The starting pool was corrected to 1,400 MedQA and 1,944 MATH candidates. Its text is small at report width. Editable TikZ source and logos are included under figures/.

Edit midsem_report.tex. The remaining tables are separate .tex inputs with unchanged values. The original tool-call tables are preserved in medqa_calls.tex and math_calls.tex for reference, but are not included in the PDF. Their replacements are medqa_calls_figure.tex and math_calls_figure.tex. Regenerate the two graphs with python create_tool_call_bars.py; counts and percentages match the original tables.

Build from this folder:

    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex
    bibtex midsem_report
    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex
    pdflatex -interaction=nonstopmode -halt-on-error midsem_report.tex

English editing preserves the supplied section order and discussion. A short future-work section and working references were added. The AI-use statement covers this editing and the broader known project assistance; it does not claim grammar-only AI use throughout the project.
