#!/bin/bash

CCLINCLUDE='./include/'
AngPowINCLUDE='./angpow/inc/'
AngPowLIB='./angpow/lib/'
FFTWLIB='/opt/local/lib'
GSLINCLUDE='/opt/local/include'

#echo gcc -Wall -Wpedantic -g -I$CCLINCLUDE -I$AngPowINCLUDE -std=c++11 -fPIC ccl_angpow_interface.cc -o tests/ccl_angpow_interface -lgsl -lgslcblas -lm -lccl -I/opt/local/include/

echo "g++  -Wall -Wpedantic -g  -std=c++11 -fPIC -fopenmp -I$CCLINCLUDE -I$GSLINCLUDE -I$AngPowINCLUDE tests/ccl_sample_angpow.cc -o tests/ccl_test_angpow -L$FFTWLIB -L$AngPowLIB -langpow -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm  -DHAVE_ANGPOW"
g++  -Wall -Wpedantic -g  -std=c++11 -fPIC -fopenmp -I$CCLINCLUDE -I$GSLINCLUDE -I$AngPowINCLUDE tests/ccl_sample_angpow.c -o tests/ccl_test_angpow -L$FFTWLIB -L$AngPowLIB -langpow -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm -DHAVE_ANGPOW

#g++  -Wall -Wpedantic -g  -std=c++11 -fPIC -fopenmp -I./include -I/home/damonge/include -I../AngPow-rsd/inc tests/ccl_sample_angpow.cc -o tests/ccl_test_angpow -L/home/damonge/lib -L../AngPow-rsd/lib -langpow -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm
