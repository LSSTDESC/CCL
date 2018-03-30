#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv

    case "${TOXENV}" in
        py27)
            # Install some custom Python 2.7 requirements on OS X
            echo "Native python environment"
            sudo easy_install pip
            pip install --user nose
            ;;
        py36)
            echo "Installing miniconda.";
            if test -e $HOME/miniconda/bin; then
              echo "miniconda already installed.";
            else
              rm -rf $HOME/miniconda;
              mkdir -p $HOME/download;
              wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
              bash $HOME/download/miniconda.sh -b -p $HOME/miniconda;
            fi;
            source $HOME/miniconda/bin/activate
            hash -r
            conda config --set always_yes yes --set changeps1 no
            conda update -q conda
            conda info -a
            conda create -q -n test-environment python=3.6 pip
            source activate test-environment
            pip install numpy nose
            ;;
    esac
else
    # Install some custom requirements on Linux
    echo "No specific requirements on linux"
fi
