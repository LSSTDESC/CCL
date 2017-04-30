#!/bin/bash

CCLINCLUDE='./include/'
AngPowINCLUDE='../AngPow-rsd/inc/'
AngPowLIB='../AngPow-rsd/lib/'
FFTWLIB='/home/damonge/lib'
GSLINCLUDE='/home/damonge/include'

#echo gcc -Wall -Wpedantic -g -I$CCLINCLUDE -I$AngPowINCLUDE -std=c++11 -fPIC ccl_angpow_interface.cc -o tests/ccl_angpow_interface -lgsl -lgslcblas -lm -lccl -I/opt/local/include/

echo "g++  -Wall -Wpedantic -g  -std=c++11 -fPIC -fopenmp -I./include -I/home/damonge/include -I../AngPow-rsd/inc tests/ccl_sample_angpow.cc -o tests/ccl_test_angpow -L/home/damonge/lib -L../AngPow-rsd/lib -langpow -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm"
g++  -Wall -Wpedantic -g  -std=c++11 -fPIC -fopenmp -I./include -I/home/damonge/include -I../AngPow-rsd/inc tests/ccl_sample_angpow.cc -o tests/ccl_test_angpow -L/home/damonge/lib -L../AngPow-rsd/lib -langpow -lccl -lgsl -lgslcblas -lfftw3 -lfftw3_threads -lm
