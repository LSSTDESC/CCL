SUBDIRS = include class
EXTRA_DIST = doc sandbox README.md tests/benchmark tests/ccl_example_photoz.c tests/ccl_example_specs.c tests/ccl_sample_power.c tests/ctest.h tests/min_code.c tests/readme.md

lib_LTLIBRARIES = libccl.la
libccl_la_SOURCES = src/ccl_background.c src/ccl_core.c src/ccl_error.c src/ccl_lsst_specs.c src/ccl_placeholder.c src/ccl_power.c src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c
libccl_la_LIBADD = class/libclass.la
libccl_la_CPPFLAGS = -I./include/ -I./class/include

noinst_PROGRAMS = tests/check_ccl
TESTS = tests/check_ccl
tests_check_ccl_SOURCES = tests/ccl_test.c tests/ccl_test_utils.c tests/ccl_test_params.c tests/ccl_test_distances.c tests/ccl_test_growth.c tests/ccl_test_bbks.c tests/ccl_test_cls.c tests/ccl_test_sigmaM.c tests/ccl_test_massfunc.c
tests_check_ccl_CPPFLAGS = -I./tests/ -I./include/
tests_check_ccl_LDFLAGS =  -L./ -lccl -lgsl -lgslcblas

dist-hook:
	rm -f $(distdir)/clean_code.sh
