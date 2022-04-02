all: clean plots
	latexmk -r .latexmkrc
plots:
	python plots.py
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.pdf *.run.xml *.xdv *.toc *.png