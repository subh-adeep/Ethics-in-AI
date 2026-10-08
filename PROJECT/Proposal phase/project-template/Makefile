all: compiletex compilebib linkbib linkreferences

.PHONY: clean

clean:
	rm *.aux *.bbl *.blg *.log *.out 2>/dev/null

compiletex:
	pdflatex project_template

compilebib:
	bibtex project_template

linkbib:
	pdflatex project_template

linkreferences:
	pdflatex project_template
