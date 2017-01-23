#!/usr/bin/env python

# System imports
from distutils.core import *
from distutils import sysconfig

# Third-party modules
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# CCL extension module
_ccllib = Extension(
            "_ccllib",
               ["pyccl/ccl.i",
                "src/ccl_core.c",
                "src/ccl_background.c",
                "src/ccl_cls.c",
                "src/ccl_error.c",
                "src/ccl_lsst_specs.c",
                "src/ccl_massfunc.c",
                "src/ccl_placeholder.c",
                "src/ccl_utils.c",
                "src/ccl_power.c",
                
                "class/source/background.c", 
                "class/source/input.c", 
                "class/source/lensing.c", 
                "class/source/nonlinear.c", 
                "class/source/output.c", 
                "class/source/perturbations.c", 
                "class/source/primordial.c", 
                "class/source/spectra.c", 
                "class/source/thermodynamics.c", 
                "class/source/transfer.c", 
                
                "class/tools/arrays.c", 
                "class/tools/common.c", 
                "class/tools/dei_rkck.c", 
                "class/tools/evolver_ndf15.c", 
                "class/tools/evolver_rkck.c", 
                "class/tools/growTable.c", 
                "class/tools/hyperspherical.c", 
                "class/tools/parser.c", 
                "class/tools/quadrature.c", 
                "class/tools/sparse.c", 
                ],
                #"src/ccl_power_dummy.c",],
               libraries = ['m', 'gsl', 'gslcblas', 'gomp'],
               include_dirs = [numpy_include, "include/", "class/include"],
               extra_compile_args=['-O4', '-fopenmp', 
                                   '-D__CLASSDIR__="./class"'],
               #extra_compile_args=['-g', '-O0', '-Q',], # For debugging
               swig_opts=['-threads'],
               )


# CCL setup
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.1",
        packages     = ['pyccl'],
        ext_modules  = [_ccllib],
        #data_files   = [('data', ['pyccl/README', ]),]
        )
