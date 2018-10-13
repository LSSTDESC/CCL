########## User-definable stuff ##########
#
###Compiler and compilation options
COMP = gcc
OPTIONS = -Wall -O3
#OPTIONS += -D_HAS_OMP -fopenmp
OPTIONS += -D_DEBUG
#
###Path to libraries and headers
###If two or more of the dependencies reside in the same paths, only
###one instance is necessary.
#GSL
GSL_INC = -I/home/alonso/include
GSL_LIB = -L/home/alonso/lib
#COSMOMAD
CSM_INC =
CSM_LIB =
#
########## End of user-definable ##########

OPTIONS += $(DEFINEFLAGS)

INC_ALL = -I./src $(GSL_INC) $(CSM_INC) $(DAM_INC)
LIB_ALL = $(GSL_LIB) $(CSM_LIB) $(DAM_LIB) -lcosmomad -lgsl -lgslcblas -lm

COMMONO = src/common.o
IOO = src/io.o
COSMOO = src/cosmo.o
TRANSFERO = src/transfers.o
SPECTRAO = src/spectra.o
MAIN = src/limberjack.c
OFILES = $(COMMONO) $(IOO) $(COSMOO) $(TRANSFERO) $(SPECTRAO)

EXE = limberjack

default : $(EXE)

%.o : %.c
	$(COMP) $(OPTIONS) $(INC_ALL) -c $< -o $@

$(EXE) : $(OFILES)
	$(COMP) $(OPTIONS) $(INC_ALL) $(OFILES) $(MAIN) -o $(EXE) $(LIB_ALL)

clean :
	rm -f src/*.o

cleaner : 
	rm -f *~ src/*.o src/*~ $(EXE)
