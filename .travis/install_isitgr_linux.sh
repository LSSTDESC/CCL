#!/bin/bash

git clone https://github.com/mishakb/ISiTGR
cd ISiTGR/camb
python setup.py install
echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
