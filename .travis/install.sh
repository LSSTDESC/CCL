#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv

    case "${TOXENV}" in
        py27)
            # Install some custom Python 2.7 requirements on OS X
            echo "No specific requirements on osx for now"
            easy_install --user pip
            pip install --user nose
            ;;
        py36)
            # Install some custom Python 3.6 requirements on OS X
            echo "No specific requirements on osx for now"
            brew update
            brew upgrade python
            virtualenv venv -p python3
            source venv/bin/activate
            pip install --user nose numpy
            ;;
    esac
else
    # Install some custom requirements on Linux
    echo "No specific requirements on linux"
fi
