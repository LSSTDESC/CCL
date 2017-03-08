#!/bin/bash

CCLINCLUDE='./include/'
AngPowINCLUDE='../AngPow/inc/Angpow/'

echo gcc -Wall -Wpedantic -g -I$CCLINCLUDE -I$AngPowINCLUDE -std=c++11 -fPIC ccl_angpow_interface.cc -o tests/ccl_angpow_interface -lgsl -lgslcblas -lm -lccl -I/opt/local/include/

g++  -Wall -Wpedantic -g -I$CCLINCLUDE  -I$AngPowINCLUDE -std=c++11 -fPIC ccl_angpow_interface.cc -o tests/ccl_angpow_interface -lgsl -lgslcblas -lm -lccl -I/opt/local/include/
