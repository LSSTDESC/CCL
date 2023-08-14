MDIR := $(shell pwd)

CC = gcc

LIB_LINK =

GSL_DIR := $(shell echo ${GSL_DIR})
FFTW_INC := $(shell echo ${FFTW_INC})
FFTW_DIR := $(shell echo ${FFTW_DIR})

ifneq ($(GSL_DIR),)
  LIB_LINK += -I $(GSL_DIR)/include -L $(GSL_DIR)/lib
endif
ifneq ($(FFTW_DIR),)
  LIB_LINK += -I $(FFTW_INC) -L $(FFTW_DIR)
endif

LIB_LINK += -lgsl -lgslcblas -lm -lmvec -lfftw3
FLAGS = -fPIC -Wall -O3 -ffast-math
BUILD_DIR = $(MDIR)/build
LIB = libfftlogx
SRC_DIR = $(MDIR)/src
_SRC = \
	cfftlog.c \
	utils.c \
	utils_complex.c \

SRC = $(addprefix $(SRC_DIR)/,$(_SRC))


all: $(LIB).so

$(LIB).so:
	if ! [ -e $(BUILD_DIR) ]; then mkdir $(BUILD_DIR); fi;
	$(CC) -shared -o $(BUILD_DIR)/$(LIB).so $(SRC) $(LIB_LINK) $(FLAGS)

clean:
	rm $(BUILD_DIR)/*.so
