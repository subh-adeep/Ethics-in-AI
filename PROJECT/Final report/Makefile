MAIN = final_report_submission

all: compiletex compilebib linkbib linkreferences

.PHONY: clean

clean:
	rm -f *.aux *.bbl *.blg *.log *.out 2>/dev/null

compiletex:
	pdflatex $(MAIN)

compilebib:
	bibtex $(MAIN)

linkbib:
	pdflatex $(MAIN)

linkreferences:
	pdflatex $(MAIN)
