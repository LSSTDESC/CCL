#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    CONDA_INST=MacOSX
else
    CONDA_INST=Linux
fi

# Install some custom requirements on OS X
if test -e $HOME/miniconda/bin; then
  echo "miniconda already installed.";
else
  echo "Installing miniconda.";
  rm -rf $HOME/miniconda;
  mkdir -p $HOME/download;
  if [ "${TOXENV}" = py27 ]; then
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-${CONDA_INST}-x86_64.sh -O $HOME/download/miniconda.sh;
  else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-${CONDA_INST}-x86_64.sh -O $HOME/download/miniconda.sh;
  fi;
  bash $HOME/download/miniconda.sh -b -p $HOME/miniconda;
fi;
source $HOME/miniconda/bin/activate
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

case "${TOXENV}" in
    py27)
        conda create -q -n test-environment python=2.7 pip
        ;;
    py36)
        conda create -q -n test-environment python=3.6 pip
        ;;
esac;

source activate test-environment
conda install numpy nose coveralls flake8 pyyaml gsl fftw3 cmake swig
