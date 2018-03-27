#!/usr/bin/env python
from distutils.command.build import build as _build
from distutils.core import setup
from subprocess import call
import os

class build(_build):
    """Specialized Python source builder."""
    def run(self):
        call(["mkdir", "-p", "build"])
        call(["cmake", "-H.", "-Bbuild"])
        call(["cp", "build/_pyccl.so", "pyccl/"])
        call(["cp", "build/ccllib.py", "pyccl/"])
        _build.run(self)

setup(name="pyccl",
    description="Library of validated cosmological functions.",
    author="LSST DESC",
    version="0.1",
    packages=['pyccl'],
    provides=['pyccl'],
    package_data={'pyccl': ['_pyccl.so']},
    install_requires=['numpy'],
    cmdclass={'build': build},
    classifiers=[
	  'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: C',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics'
      ]
    )
