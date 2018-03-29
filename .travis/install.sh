#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv

    case "${TOXENV}" in
        py27)
            # Install some custom Python 2.7 requirements on OS X
            echo "Native python environment"
            sudo easy_install pip
            sudo pip install nose
            ;;
        py36)
            # Install some custom Python 3.6 requirements on OS X
            brew update
            # Per the `pyenv homebrew recommendations <https://github.com/yyuu/pyenv/wiki#suggested-build-environment>`_.
            brew install openssl readline
            # See https://docs.travis-ci.com/user/osx-ci-environment/#A-note-on-upgrading-packages.
            brew outdated pyenv || brew upgrade pyenv
            # virtualenv doesn't work without pyenv knowledge. venv in Python 3.3
            # doesn't provide Pip by default. So, use `pyenv-virtualenv <https://github.com/yyuu/pyenv-virtualenv/blob/master/README.md>`_.
            brew install pyenv-virtualenv
            pyenv install 3.6.4
            export PYENV_VERSION=$PYTHON
            export PATH="/Users/travis/.pyenv/shims:${PATH}"
            pyenv-virtualenv venv
            source venv/bin/activate
            # A manual check that the correct version of Python is running.
            python --version
            pip install --user nose numpy
            ;;
    esac
else
    # Install some custom requirements on Linux
    echo "No specific requirements on linux"
fi
