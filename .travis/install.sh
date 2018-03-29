#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv

    case "${TOXENV}" in
        py27)
            # Install some custom Python 2.7 requirements on OS X
            echo "No specific requirements on osx for now"
            ;;
        py36)
            # Install some custom Python 3.6 requirements on OS X
            echo "No specific requirements on osx for now"
            ;;
    esac
else
    # Install some custom requirements on Linux
    echo "No specific requirements on linux"
fi
