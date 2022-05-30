all:
	latexmk -r .latexmkrc
plots:
	python plots.py
	python plot_comp.py
	python plot_li.py
	python plot_sa.py
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out thesis.pdf *.run.xml *.xdv *.toc