SUBDIRS = include class angpow
EXTRA_DIST = doc sandbox README.md tests/benchmark tests/ccl_example_photoz.c tests/ccl_example_specs.c tests/ccl_sample_power.c tests/ctest.h tests/min_code.c tests/readme.md include/ccl_params.ini

ccldir = $(datadir)/ccl
ccl_DATA = include/ccl_params.ini

lib_LTLIBRARIES = libccl.la
libccl_la_SOURCES = src/ccl_background.c src/ccl_core.c src/ccl_error.c src/ccl_lsst_specs.c src/ccl_placeholder.c src/ccl_power.c src/ccl_utils.c src/ccl_cls.cc src/ccl_massfunc.c
libccl_la_LIBADD = class/libclass.la angpow/libangpow.la
libccl_la_CPPFLAGS = $(OPENMP_CFLAGS) $(OPENMP_CXXFLAGS) -std=c++11 -I./include/ -I./class/include -I./angpow/inc -D__CCL_DATA_DIR__=$(datadir)/ccl

noinst_PROGRAMS = tests/check_ccl
TESTS = tests/check_ccl
tests_check_ccl_SOURCES = tests/ccl_test.c tests/ccl_test_utils.c tests/ccl_test_params.c tests/ccl_test_cosmology.c tests/ccl_test_distances.c tests/ccl_test_growth.c tests/ccl_test_bbks.c tests/ccl_test_eh.c tests/ccl_test_cls.c tests/ccl_test_sigmaM.c tests/ccl_test_massfunc.c
tests_check_ccl_CPPFLAGS = $(OPENMP_CFLAGS) $(OPENMP_CXXFLAGS) -I./tests/ -I./include/ -fopenmp
tests_check_ccl_LDFLAGS =  $(OPENMP_CFLAGS) $(OPENMP_CXXFLAGS) -L./ -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm

dist-hook:
	rm -f $(distdir)/clean_code.sh
