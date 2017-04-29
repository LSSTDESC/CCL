noinst_LTLIBRARIES = libangpow.la

libangpow_la_SOURCES = src/angpow_bessel.cc src/angpow_chebyshevInt.cc src/angpow_clbase.cc src/angpow_cosmo.cc src/angpow_ctheta.cc src/angpow_kinteg.cc src/angpow_parameters.cc src/angpow_pk2cl.cc src/angpow_powspec.cc src/angpow_radint.cc src/angpow_tools.cc src/walltime_c.cc src/walltimer.cc
libangpow_la_CPPFLAGS = -Wall -Wpointer-arith -O3 -fPIC -std=c++11 $(OPENMP_CFLAGS) $(OPENMP_CXXFLAGS) -fno-common -march=native -ffast-math -m64 -c -I./inc/

#libclass_la_SOURCES = source/background.c source/input.c source/lensing.c source/nonlinear.c source/output.c source/perturbations.c source/primordial.c source/spectra.c source/thermodynamics.c source/transfer.c tools/arrays.c tools/common.c tools/dei_rkck.c tools/evolver_ndf15.c tools/evolver_rkck.c tools/growTable.c tools/hyperspherical.c tools/parser.c tools/quadrature.c tools/sparse.c
#libclass_la_CPPFLAGS = -O4 -ffast-math -I./include/ -I./class/tools/ -D__CLASSDIR__='"$(abs_top_srcdir)/class"'

#EXTRA_DIST = bbn explanatory.ini main include tools/hermite3_interpolation_csource.h tools/hermite4_interpolation_csource.h tools/hermite6_interpolation_csource.h


#g++ -DLinuxx   -I./inc/ -I/home/damonge/include -Wall -Wpointer-arith -O3 -fPIC -std=c++11 -fopenmp  -fno-common  -Wall -march=native -ffast-math -m64 -c  -I./inc/ -c src/angpow_bessel.cc -o Objs/angpow_bessel.o
