# To make the document with the default settings:
# > make
#
# To make with a specific format:
# > make <format>
#
# To tar up a flat version of a specific format:
# > make <format> tar
#
# Alex Drlica-Wagner: https://github.com/DarkEnergyScienceCollaboration/start_paper/issues/new?body=@kadrlica

# Primary file names - avoid cookiecutter variables, to enable `make
# upgrade` to cleanly over-write this Makefile...
main=main
outname=$(notdir $(shell pwd))

# LATEX environment variables
export TEXINPUTS:=./texmf/styles/:./tables/:
export BSTINPUTS:=./texmf/bib/:

# LaTeX journal class switcher flags
# apj=\def\flag{apj}
# apjl=\def
# mnras=\def\flag{mnras}

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
source=$(main).{tex,bbl,bib} lsstdesc.bib macros.tex authors.tex acknowledgments.tex

tarfiles=$(figures) $(tables) $(styles) $(bibs) $(source)

all: export flag = ${draft}
all: main copy

copy:
	cp ${main}.pdf ${outname}.pdf

touch:
	touch ${main}.tex

#http://journals.aas.org/authors/aastex/linux.html
#change the compiler call to allow a "." file
# {% raw %}
main :
	latexmk -g -pdf  \
	-pdflatex='openout_any=a pdflatex %O -interaction=nonstopmode "${flag}\input{%S}"'  \
	${main}
# {% endraw %}

tar : main
	mkdir -p ${tardir}
	cp ${tarfiles} ${tardir} | true
	cp ${outname}.pdf ${tardir}/${outname}.pdf
	cd ${tardir} && tar -czf ../${outname}.tar.gz . && cd ..
	rm -rf ${tardir}

# http://stackoverflow.com/q/8028314/
TARGETS=apj apjl prd prl mnras   tex
$(TARGETS): export flag = \def\flag{$(@)}
$(TARGETS):
	$(MAKE) -e main
	$(MAKE) -e tar
# NB. the 'tex' target doesn't actually do anything in docswitch - make
# with no target compiles PDF out of main.tex using lsstdescnote.cls
# (which is to say, by default we assume you are writing an LSST
# DESC Note in latex format).

tidy:
	rm -f *.log *.aux *.out *.dvi *.synctex.gz *.fdb_latexmk *.fls
	rm -f *.bbl *.blg *Notes.bib ${main}.pdf

clean: tidy
	rm -f ${outname}.pdf ${outname}.tar.gz

# Update the tex styles etc:

baseurl=https://raw.githubusercontent.com/DarkEnergyScienceCollaboration/start_paper/master/%7B%7Bcookiecutter.folder_name%7D%7D

UPDATES=\
texmf/bib/apj.bst \
texmf/bib/mnras.bst \
texmf/styles/aas_macros.sty \
texmf/styles/aastex.cls \
texmf/styles/aastex61.cls \
texmf/styles/aps_macros.sty \
texmf/styles/docswitch.sty \
texmf/styles/emulateapj.cls \
texmf/styles/mnras.cls \
texmf/styles/lsstdescnote.cls \
.logos/desc-logo-small.png \
.logos/desc-logo.png \
.logos/header.png \
lsstdesc.bib \
.travis.yml \
figures/example.png

.PHONY: $(UPDATES)
$(UPDATES):
	curl -s -S -o $(@) ${baseurl}/$(@)
	@echo " "

update:
	@echo "\nOver-writing LaTeX style files with the latest versions: \n"
	@mkdir -p .logos figures texmf/styles texmf/bib
	$(MAKE) $(UPDATES)

# Get fresh copies of the templates etc, for reference:

TEMPLATES=\
acknowledgments.tex \
authors.tex \
macros.tex \
main.ipynb \
main.md \
main.rst \
main.tex \
main.bib

.PHONY: $(TEMPLATES) templates
$(TEMPLATES):
	curl -s -S -o templates/$(@) ${baseurl}/$(@)
	@echo " "

templates:
	@echo "\nDownloading the latest versions of the template files, for reference: \n"
	@mkdir -p templates
	$(MAKE) $(TEMPLATES)
	$(MAKE) new
	ls templates/*

# Get a template copy of the latest Makefile, for reference:
.PHONY: new
new:
	@echo "\nDownloading the latest version of the Makefile, for reference: \n"
	@mkdir -p templates
	curl -s -S -o templates/Makefile ${baseurl}/Makefile
	@echo " "

# Over-write this Makefile with the latest version:
.PHONY: upgrade
upgrade:
	@echo "\nDownloading the latest version of the Makefile: \n"
	curl -s -S -o Makefile ${baseurl}/Makefile
	@echo "\nNow get the latest styles and templates with\n\n    make update\n    make templates\n"
