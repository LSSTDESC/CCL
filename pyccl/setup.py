#!/usr/bin/env python

# System imports
from distutils.core import *
from distutils import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# CCL extension module
_ccllib = Extension("_ccllib",
                   ["ccl.i", 
                    "../src/ccl_core.c",
                    "../src/ccl_background.c",
                    "../src/ccl_cls.c",
                    "../src/ccl_error.c",
                    "../src/ccl_lsst_specs.c",
                    "../src/ccl_massfunc.c",
                    "../src/ccl_placeholder.c",
                    "../src/ccl_utils.c",
                    "../src/ccl_power_dummy.c",],
                   libraries = ['m', 'gsl', 'gslcblas',],
                   include_dirs = [numpy_include, "../include/"],
                   )

# CCL setup
setup(  name        = "CCL",
        description = "Library of validated cosmological functions.",
        author      = "Phil Bull",
        version     = "0.1",
        ext_modules = [_ccllib]
        )
