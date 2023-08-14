# your C compiler:
CC       = gcc
#CC       = icc
#CC       = pgcc

# your tool for creating static libraries:
AR        = ar rv

# Your python interpreter. (python3 for N5K, usually)
PYTHON ?= python3

# your optimization flag
OPTFLAG = -O4 -ffast-math #-march=native
#OPTFLAG = -Ofast -ffast-math #-march=native
#OPTFLAG = -fast

# your openmp flag (comment for compiling without openmp)
OMPFLAG   = -fopenmp
#OMPFLAG   = -mp -mp=nonuma -mp=allcores -g
#OMPFLAG   = -openmp


# all other compilation flags
CCFLAG = -g -fPIC
LDFLAG = -g -fPIC

# output directory to store the bessel files in
BESSEL_DIR := ${CURDIR}
CCFLAG += -DBESSEL_DIR='"$(BESSEL_DIR)"'

all: matter libmatter.a matterlib

SOURCE = matter.o fft.o hypergeom.o arrays.o common.o

%.o:  %.c
	mkdir -p build; cd build; $(CC) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) -c ../$< -o $*.o

libmatter.a: $(SOURCE)
	$(AR)  $@ $(addprefix build/, $(SOURCE))

matter: $(SOURCE)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o class $(addprefix build/,$(notdir $^)) -lm

matterlib: libmatter.a python/mattery.pyx python/mattery.pxd
ifdef OMPFLAG
	cp python/setup.py python/autosetup.py
else
	grep -v "lgomp" python/setup.py > python/autosetup.py
endif
	cd python; export CC=$(CC); $(PYTHON) autosetup.py install || $(PYTHON) autosetup.py install --user
	echo "hello world"
	rm python/autosetup.py

clean:
	rm -f libmatter.a
	rm -f build/*
	rm -f python/classy.c
	rm -rf python/build
