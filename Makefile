all:
	latexmk -r .latexmkrc
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.pdf *.run.xml *.xdv *.toc