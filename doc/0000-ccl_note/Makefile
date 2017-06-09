# To make the document with the default settings:
# > make
#
# To make with a specific format:
# > make <format>
#
# To tar up a flat version of a specific format:
# > make <format> tar
#
# Alex Drlica-Wagner: https://github.com/LSSTDESC/start_paper/issues/new?body=@kadrlica

# Primary file names - avoid cookiecutter variables, to enable `make
# upgrade` to cleanly over-write this Makefile...
main=main
default=$(shell cat .metadata.json | grep 'default_format' | cut -d'"' -f4)
outname=$(notdir $(shell pwd))

ifeq ($(default), {{ cookiecutter.default_format }})
 style=tex
else
 style=${default}
endif

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
source=$(main).{tex,bbl,bib} lsstdesc.bib acknowledgments.tex authors.tex contributions.tex

tarfiles=$(figures) $(tables) $(styles) $(bibs) $(source)

# Interpret `make` with no target as `make tex` (a latex Note).
# At present, if the default_format is anything other than
# [apj|apjl|mnras|prd|prl|emulateapj], a latex Note is made.
# In future, we could think of using `make` to eg run the ipynb
# notebook and make PDF from the output, but this has not been
# implemented yet.
all: export flag = \def\flag{${style}}
all: main copy

copy:
	cp ${main}.pdf ${outname}.pdf

touch:
	touch ${main}.tex

#http://journals.aas.org/authors/aastex/linux.html
#change the compiler call to allow a "." file
# {% raw %}
main : authlist
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

authlist :
	pip install --upgrade mkauthlist
	mkauthlist -j ${style} -f -c "LSST Dark Energy Science Collaboration" \
		--cntrb contributions.tex authors.csv authors.tex

# http://stackoverflow.com/q/8028314/
TARGETS=apj apjl prd prl mnras tex aastex61 emulateapj
$(TARGETS): export style = $(@)
$(TARGETS): export flag = \def\flag{$(@)}
$(TARGETS):
	$(MAKE) -e main
	$(MAKE) -e copy
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

baseurl=https://raw.githubusercontent.com/LSSTDESC/start_paper/master/%7B%7Bcookiecutter.folder_name%7D%7D

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
texmf/styles/lsstdesc_macros.sty \
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
authors.csv \
main.ipynb \
main.md \
main.rst \
main.tex \
main.bib \
.metadata.json

.PHONY: $(TEMPLATES) templates
$(TEMPLATES):
	curl -s -S -o templates/$(@) ${baseurl}/$(@)
	@echo " "

templates:
	@echo "\nDownloading the latest versions of the template files, for reference: \n"
	@mkdir -p templates
	$(MAKE) $(TEMPLATES)
	$(MAKE) new
	ls -a templates/*

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
	@echo "\nUpgrading version of mkauthlist: \n"
	pip install mkauthlist --upgrade --no-deps
	@echo "\nNow get the latest styles and templates with\n\n    make update\n    make templates\n"
