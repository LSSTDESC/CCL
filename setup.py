#!/usr/bin/env python

from distutils.core import *
from distutils import sysconfig
import os.path
import sys

# Throw error if Python 3.x is being used
if sys.version_info.major == 3:
    print("Python 3.x is not currently supported. Please use Python 2.")
    sys.exit(1)

# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# CCL extension module
_ccllib = Extension(
    "_ccllib",
    ["pyccl/ccl_wrap.c",],
    libraries = ['m', 'gsl', 'gslcblas', 'ccl'],
    include_dirs = [numpy_include, "include/", "class/include"],
    extra_compile_args=['-O4', '-std=c99'],
    swig_opts=['-threads'],
    )

# CCL setup script
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.2",
        packages     = ['pyccl'],
        ext_modules  = [_ccllib,],
        )
