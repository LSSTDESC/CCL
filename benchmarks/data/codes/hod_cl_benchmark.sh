#!/bin/bash

# Clone and install
git clone git@github.com:damonge/pysz_gal.git
cd pysz_gal/pysz_gal/source
make clean
make
cd ../../
python setup.py install --user
cd ..
python hod_bm.py

