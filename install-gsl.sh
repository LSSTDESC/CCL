#!/bin/sh
set -ex
wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz
tar -xzvf gsl-2.4.tar.gz
cd gsl-2.4 && ./configure && make && sudo make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_RUN_PATH=$LD_RUN_PATH:/usr/local/lib
