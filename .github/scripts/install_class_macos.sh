#!/bin/bash

git clone https://github.com/lesgourg/class_public.git
cd class_public
sed -i "s/CC       = gcc/CC       = gcc-9/g" Makefile
git checkout v2.7.2
make
