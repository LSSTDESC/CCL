#!/bin/bash

# This generates the SZ power spectrum benchmarks
git clone git@github.com:borisbolliet/class_sz.git
cd class_sz
make clean ; make -j8 class
cd ..

# Planck 2013
./class_sz/class class_sz_P13.ini
rm ../sz_cl_P13_pk_cb.dat ../sz_cl_P13_pk.dat ../sz_cl_P13_redshift_dependent_functions.txt ../sz_cl_P13_tk.dat
