#!/bin/bash

CONDA_INST=Linux

echo "installing miniconda"
rm -rf $HOME/miniconda
mkdir -p $HOME/download
curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-${CONDA_INST}-x86_64.sh -o $HOME/download/miniconda.sh
bash $HOME/download/miniconda.sh -b -p $HOME/miniconda

export PATH=$HOME/miniconda/bin:$PATH

conda config --set always_yes yes --set changeps1 no
conda config --add channels defaults
conda config --add channels conda-forge
conda update -q conda
conda info -a

conda create -q -n test-environment python=3.6 pip \
      numpy nose coveralls flake8 pyyaml gsl fftw cmake swig  scipy \
      compilers pkg-config setuptools_scm pytest pandas pytest-cov \
      cython "camb>=1" isitgr traitlets fast-pt
