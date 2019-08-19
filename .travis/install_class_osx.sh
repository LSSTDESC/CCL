#!/bin/bash

git clone https://github.com/lesgourg/class_public.git
cd class_public
git checkout v2.7.2

sed 's/"CC       = gcc"/""/g' Makefile
sed 's/"AR        = ar rv"/""/g' Makefile

cat Makefile

make

# at this point the make file leaves you in the python dir
python setup.py install

cd ..
