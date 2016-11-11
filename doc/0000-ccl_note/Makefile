# To make the document with the default settings:
# > make
#
# To make with a specific format:
# > make <format>
#
# To tar up a flat version of a specific format:
# > make <format> tar
#
# Alex Drlica-Wagner <kadrlica@fnal.gov>
#

# Primary file names
main=main
outname=0000-ccl_note

# LATEX environment variables
export TEXINPUTS:=.:./texmf/styles/:./tables/:
export BSTINPUTS:=.:./texmf/bib/:

# Journal class switcher flags
# apj=\def\flag{apj}
# apjl=\def
# mnras=\def\flag{mnras}
# note=\def\flag{note}

# Submission flags (these need some thought)
# arxiv=\def\flag{emulateapj}
# submit=${aastex}
# draft=\def\linenums{\linenumbers}

# Files to copy when making tarball
tardir=tmp
figdir=./figures
figures=$(figdir)/*.{png,jpg,pdf}
tabdir=./tables
tables=$(tabdir)/*.tex
styles=./texmf/styles/*.{sty,cls}
bibs=./texmf/bib/*.bst
source=$(main).{tex,bbl,bib} macros.tex authors.tex acknowledgments.tex

tarfiles=$(figures) $(tables) $(styles) $(bibs) $(source)

all: export flag = ${draft}
all: main

copy: 
	cp ${main}.pdf ${outname}.pdf

touch:
	touch ${main}.tex

# 
main : 
	latexmk -g -pdf  \
	-pdflatex='pdflatex %O -interaction=nonstopmode "${flag}\input{%S}"'  \
	${main}
# 

tar : main
	mkdir -p ${tardir}
	cp ${tarfiles} ${tardir} | true
	cp ${main}.pdf ${tardir}/${main}.pdf
	cd ${tardir} && tar -czf ../${main}.tar.gz . && cd ..
	rm -rf ${tardir}

# http://stackoverflow.com/q/8028314/
TARGETS=apj apjl prd prl mnras note
$(TARGETS): export flag = \def\flag{$(@)}
$(TARGETS): 
	$(MAKE) -e main
	$(MAKE) -e tar

tidy:
	rm -f *.log *.aux *.out *.dvi *.synctex.gz *.fdb_latexmk *.fls
	rm -f *.bbl *.blg *Notes.bib

clean: tidy
	rm -f ${main}.pdf ${main}.tar.gz
