
if LOCALCLASS
 SUBDIRS = include class
else
 SUBDIRS = include
endif

EXTRA_DIST = doc sandbox README.md include/ccl_params.ini \
             tests/benchmark tests/ctest.h tests/min_code.c tests/readme.md \
             examples/ccl_sample_run.c \
             examples/example_neutrinos.c examples/example_photoz.c \
             examples/example_power.c examples/example_power_neutrinos.c \
             examples/Distance\ Calculations\ Example.ipynb \
             examples/Lensing\ angular\ power\ spectrum.ipynb \
             examples/Photo-z\ example.ipynb \
             examples/Power\ spectrum\ example.ipynb

ccldir = $(datadir)/ccl
ccl_DATA = include/ccl_params.ini

lib_LTLIBRARIES = libccl.la

libccl_la_SOURCES = src/ccl_background.c src/ccl_core.c src/ccl_error.c \
                    src/ccl_lsst_specs.c src/ccl_placeholder.c src/ccl_power.c \
                    src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c \
                    src/ccl_neutrinos.c

if LOCALCLASS
  libccl_la_LIBADD = class/libclass.la
endif
libccl_la_CPPFLAGS = -I./include/ -I$(CLASSDIR)/include -D__CCL_DATA_DIR__=$(datadir)/ccl

noinst_PROGRAMS = tests/check_ccl
TESTS = tests/check_ccl tests/check_swig_timestamp.sh
tests_check_ccl_SOURCES = tests/ccl_test.c tests/ccl_test_utils.c \
                          tests/ccl_test_params.c tests/ccl_test_cosmology.c \
                          tests/ccl_test_distances.c tests/ccl_test_growth.c \
                          tests/ccl_test_bbks.c tests/ccl_test_eh.c \
                          tests/ccl_test_class.c tests/ccl_test_sigmaM.c \
                          tests/ccl_test_massfunc.c tests/ccl_test_cls.c
tests_check_ccl_CPPFLAGS = -I./tests/ -I./include/
tests_check_ccl_LDFLAGS =  -L./ -lccl -lgsl -lgslcblas $(CLASS_LIB)

dist-hook:
	rm -f $(distdir)/clean_code.sh
