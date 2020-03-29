#!/bin/bash

git clone https://github.com/mishakb/ISiTGR
cd ISiTGR/camb/fortran
make

# at this point the make file leaves you in the fortran dir
cd Release
