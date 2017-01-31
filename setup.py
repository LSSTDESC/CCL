#!/usr/bin/env python

from distutils.core import *
from distutils import sysconfig
from distutils.command.build_ext import build_ext
from distutils.command.install import install
import os.path

# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# CLASS source directory
CLASS_SRC_DIR = "class"

# Default root data directory for CLASS (i.e. in the source dir)
global CLASS_DATA_DIR
CLASS_DATA_DIR = "class"

# Override default 'install' and 'build_ext' behaviours so that the eventual 
# installed location of the CLASS data directory can be passed to the compiler 
# in advance. This is necessary because CLASS looks for certain datafiles in a 
# directory that is hard-coded in the variable __CLASSDIR__ at compile-time. 
# The program segfaults if this directory is not found.

class Install(install):
    
    def run(self):
        # Define global variable containing the location in which the CLASS 
        # data will be installed (chosen to be in the same root dir as the 
        # installed Python files, to keep everything together)
        global CLASS_DATA_DIR
        CLASS_DATA_DIR = os.path.join( self.install_libbase, 
                                       self.distribution.get_name(),
                                       "class" )
        # Run the usual install script
        install.run(self)
        
        # Copy datafiles from class subdirs to installed Python module dir
        class_subdirs = ['bbn/', ]
        for subdir in class_subdirs:
            
            # Define input/output dirs in OS-independent way
            in_data_dir = os.path.join('class/', subdir)
            out_data_dir = os.path.join(CLASS_DATA_DIR, subdir)
            
            # Copy datafiles
            self.copy_tree(in_data_dir, out_data_dir)

class BuildExt(build_ext):
    
    def build_extensions(self):
        # Find Extension and modify relevant compiler argument to point at the 
        # installed location of the CLASS data (uses global variable, which is 
        # a bit of a hack)
        for ext in self.extensions:
            for i in range(len(ext.extra_compile_args)):
                if "__CLASSDIR__" in ext.extra_compile_args[i]:
                    ext.extra_compile_args[i] = \
                                    "-D__CLASSDIR__=\"%s\"" % CLASS_DATA_DIR
        
        # Run the usual build_ext script
        build_ext.build_extensions(self)


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
                
                CLASS_SRC_DIR + "/source/background.c", 
                CLASS_SRC_DIR + "/source/input.c", 
                CLASS_SRC_DIR + "/source/lensing.c", 
                CLASS_SRC_DIR + "/source/nonlinear.c", 
                CLASS_SRC_DIR + "/source/output.c", 
                CLASS_SRC_DIR + "/source/perturbations.c", 
                CLASS_SRC_DIR + "/source/primordial.c", 
                CLASS_SRC_DIR + "/source/spectra.c", 
                CLASS_SRC_DIR + "/source/thermodynamics.c", 
                CLASS_SRC_DIR + "/source/transfer.c", 
                
                CLASS_SRC_DIR + "/tools/arrays.c", 
                CLASS_SRC_DIR + "/tools/common.c", 
                CLASS_SRC_DIR + "/tools/dei_rkck.c", 
                CLASS_SRC_DIR + "/tools/evolver_ndf15.c", 
                CLASS_SRC_DIR + "/tools/evolver_rkck.c", 
                CLASS_SRC_DIR + "/tools/growTable.c", 
                CLASS_SRC_DIR + "/tools/hyperspherical.c", 
                CLASS_SRC_DIR + "/tools/parser.c", 
                CLASS_SRC_DIR + "/tools/quadrature.c", 
                CLASS_SRC_DIR + "/tools/sparse.c", 
               ],
               libraries = ['m', 'gsl', 'gslcblas', 'gomp'],
               include_dirs = [numpy_include, "include/", "class/include"],
               extra_compile_args=['-O4', '-fopenmp', '-std=gnu99',
                                   '-D__CLASSDIR__="%s"' % CLASS_DATA_DIR],
               #extra_compile_args=['-g', '-O0', '-Q',], # For debugging
               swig_opts=['-threads'],
           )
           # N.B. Needs -std=gnu99 to allow for loop initial declarations *and* 
           # for M_PI to be defined in math.h (it's not deined in C99)

# CCL setup script
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.1",
        packages     = ['pyccl'],
        ext_modules  = [_ccllib,],
        cmdclass     = {'install': Install, 'build_ext': BuildExt},
        )

