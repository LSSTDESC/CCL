#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    if test -e $HOME/miniconda/bin; then
      echo "miniconda already installed.";
    else
      echo "Installing miniconda.";
      rm -rf $HOME/miniconda;
      mkdir -p $HOME/download;
      if [ "${TOXENV}" = py27 ]; then
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
      else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
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
    pip install numpy nose coveralls flake8

else
    # Install some custom requirements on Linux
    echo "No specific requirements on linux"
    pip install nose coverage coveralls flake8
fi
