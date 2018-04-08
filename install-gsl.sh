#!/bin/sh
set -ex
if [ -d "gsl-2.4/gsl" ]
then
    cd gsl-2.4 && sudo make install
else
    wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz
    tar -xzf gsl-2.4.tar.gz
    cd gsl-2.4 && ./configure --prefix=/usr && make && sudo make install
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
export LD_RUN_PATH=$LD_RUN_PATH:/usr/lib
