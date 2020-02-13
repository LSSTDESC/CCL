#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    CONDA_INST=MacOSX
    echo "removing homewbrew"
    curl -s https://raw.githubusercontent.com/Homebrew/install/master/uninstall -o uninstall_homebrew.sh
    chmod u+x ./uninstall_homebrew.sh
    ./uninstall_homebrew.sh -f -q >& /dev/null
else
    CONDA_INST=Linux
fi

echo "installing miniconda"
rm -rf $HOME/miniconda
mkdir -p $HOME/download
if [ "${TOXENV}" = py27 ]; then
    curl -s https://repo.continuum.io/miniconda/Miniconda2-latest-${CONDA_INST}-x86_64.sh -o $HOME/download/miniconda.sh
else
    curl -s https://repo.continuum.io/miniconda/Miniconda3-latest-${CONDA_INST}-x86_64.sh -o $HOME/download/miniconda.sh
fi
bash $HOME/download/miniconda.sh -b -p $HOME/miniconda

export PATH=$HOME/miniconda/bin:$PATH

conda config --set always_yes yes --set changeps1 no
conda config --add channels defaults
conda config --add channels conda-forge
conda update -q conda
conda info -a

case "${TOXENV}" in
py27)
  conda create -q -n test-environment python=2.7 pip \
    numpy nose coveralls flake8 pyyaml gsl fftw cmake swig scipy \
    compilers pkg-config setuptools_scm pytest pandas pytest-cov \
    cython "camb>=1"
  ;;
py36)
  conda create -q -n test-environment python=3.6 pip \
    numpy nose coveralls flake8 pyyaml gsl fftw cmake swig  scipy \
    compilers pkg-config setuptools_scm pytest pandas pytest-cov \
    cython "camb>=1"
  ;;
esac;

# we have to activate the cond env before we install this
source activate test-environment
pip install https://github.com/JoeMcEwen/FAST-PT/archive/v3.0.2.tar.gz --no-deps
conda deactivate
