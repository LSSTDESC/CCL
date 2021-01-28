#!/bin/bash

git clone https://github.com/lesgourg/class_public.git
cd class_public
git checkout v2.7.2
make -j4

# at this point the make file leaves you in the python dir
cd ..
