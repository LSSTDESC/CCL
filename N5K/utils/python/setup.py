from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


import numpy as nm
import os
import subprocess as sbp
import os.path as osp


# Recover the gcc compiler
GCCPATH_STRING = sbp.Popen(
    ['gcc', '-print-libgcc-file-name'],
    stdout=sbp.PIPE).communicate()[0]
GCCPATH = osp.normpath(osp.dirname(GCCPATH_STRING)).decode()

liblist = ["matter"]
MVEC_STRING = sbp.Popen(
    ['gcc', '-lmvec'],
    stderr=sbp.PIPE).communicate()[1]
if b"mvec" not in MVEC_STRING:
    liblist += ["mvec","m"]

# define absolute paths
root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

print("Doing setup in folder ",root_folder)
setup(
    name='matterlib',
    version="v1.0",
    description='Python interface to the Matter module by Nils Schoeneberg',
    url='http://www.class-code.net',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension("matterlib", [os.path.join(root_folder, "python/mattery.pyx")],
                           include_dirs=[nm.get_include(), root_folder],
                           libraries=liblist,
                           library_dirs=[root_folder, GCCPATH],
                           extra_link_args=['-lgomp'],
                           )],gdb_debug=True)
)
