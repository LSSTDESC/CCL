SUBDIRS = include class
EXTRA_DIST = doc sandbox README.md

lib_LTLIBRARIES = libccl.la
libccl_la_SOURCES = src/ccl_background.c src/ccl_core.c src/ccl_error.c src/ccl_lsst_specs.c src/ccl_placeholder.c src/ccl_power.c src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c
libccl_la_LIBADD = class/libclass.la
libccl_la_CPPFLAGS = -I./include/ -I./class/include

dist-hook:
	rm -f $(distdir)/clean_code.sh
