noinst_LTLIBRARIES = libclass.la

libclass_la_SOURCES = source/background.c source/input.c source/lensing.c source/nonlinear.c source/output.c source/perturbations.c source/primordial.c source/spectra.c source/thermodynamics.c source/transfer.c tools/arrays.c tools/common.c tools/dei_rkck.c tools/evolver_ndf15.c tools/evolver_rkck.c tools/growTable.c tools/hyperspherical.c tools/parser.c tools/quadrature.c tools/sparse.c
libclass_la_CPPFLAGS = -O4 -ffast-math -I./include/ -I./class/tools/ -D__CLASSDIR__='"$(abs_top_srcdir)/class"'

EXTRA_DIST = bbn explanatory.ini main include tools/hermite3_interpolation_csource.h tools/hermite4_interpolation_csource.h tools/hermite6_interpolation_csource.h
