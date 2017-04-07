#!/bin/bash

CCLINCLUDE='./include/'
AngPowINCLUDE='../AngPow/inc/'
AngPowLIB='../AngPow/lib/'
FFTWLIB='/opt/local/lib'
GSLINCLUDE='/opt/local/include/'

#echo gcc -Wall -Wpedantic -g -I$CCLINCLUDE -I$AngPowINCLUDE -std=c++11 -fPIC ccl_angpow_interface.cc -o tests/ccl_angpow_interface -lgsl -lgslcblas -lm -lccl -I/opt/local/include/

g++  -Wall -Wpedantic -g -I$CCLINCLUDE  -I$AngPowINCLUDE  -I$GSLINCLUDE -std=c++11 -fPIC -fopenmp tests/ccl_test_angpow.cc -o tests/ccl_test_angpow -lgsl -lgslcblas -lm -lccl -L$AngPowLIB -langpow  -L$FFTWLIB -lfftw3 -lfftw3_threads 
