#!/bin/bash

# This generates the CIB power spectrum benchmarks
git clone git@github.com:borisbolliet/class_sz.git
cd class_sz
make clean ; make -j8 class
cd ..

./class_sz/class class_sz_cib.ini
rm ../cib_class_sz_cib.txt ../cib_class_sz_pk.dat ../cib_class_sz_redshift_dependent_functions.txt ../cib_class_sz_tk.dat
