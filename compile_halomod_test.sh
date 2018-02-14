#!/bin/bash

gcc -Wall -Wpedantic -g -O0 -I./include -std=gnu99 -fPIC tests/ccl_test_halomod.c -o tests/ccl_test_halomod -L/usr/local/lib -lgsl -lgslcblas -lm -Lclass -lclass -lccl
