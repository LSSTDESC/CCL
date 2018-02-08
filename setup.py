#!/usr/bin/env python

import os
import sys
from distutils import log, ccompiler
from distutils.cmd import Command
from distutils.core import setup
from distutils.errors import CompileError, LinkError
from distutils.extension import Extension
from distutils.sysconfig import customize_compiler, get_config_var, get_config_vars
from distutils.util import get_platform
from distutils.command.install import install as DistutilsInstall
from distutils.command.build_clib import build_clib as BuildCLib
from distutils.dir_util import mkpath
import platform
import shutil
import site
import subprocess
from subprocess import check_call
import tempfile
from textwrap import dedent

# Implicit requirement - numpy must already be installed
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# This part is taken from healpy's setup.py

# Apple switched default C++ standard libraries (from gcc's libstdc++ to
# clang's libc++), but some pre-packaged Python environments such as Anaconda
# are built against the old C++ standard library. Luckily, we don't have to
# actually detect which C++ standard library was used to build the Python
# interpreter. We just have to propagate MACOSX_DEPLOYMENT_TARGET from the
# configuration variables to the environment.

if get_config_var('MACOSX_DEPLOYMENT_TARGET') and 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = get_config_var('MACOSX_DEPLOYMENT_TARGET')

#######################
class BuildExternalCLib(BuildCLib):
    """Subclass of Distutils' standard build_clib subcommand. Adds support for
    libraries that are installed externally and detected with pkg-config, with
    an optional fallback to build from a local configure-make-install style
    distribution."""

    def __init__(self, dist):
        BuildCLib.__init__(self, dist)
        self.build_args = {}

    def check_extensions(self):
        ret_val = _check_extensions()
        return ret_val

    def build_library(self,library):
        # Use a subdirectory of build_temp as the build directory.
        target_temp = os.path.realpath(os.path.join(self.build_temp, library))
        # Destination for headers and libraries is build_clib.
        target_clib = os.path.realpath(self.build_clib)


        # Create build directories if they do not yet exist.
        mkpath(target_temp)
        mkpath(target_clib)

        env = _get_build_env()

        # Run configure.
        cmd = ['/bin/sh', os.path.join(os.path.dirname(__file__), 'configure'),
            '--prefix=' + target_clib,
            '--disable-shared',
            '--with-pic',
            '--disable-maintainer-mode']

        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=env)

        # Run make install.
        cmd = ['make', 'install']
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=env)
        return target_clib

    def run(self):
        # Uncomment the line below if you want to check if the C library
        # is installed and in your path.
        # ret_val = self.check_extensions()
        build_path = self.build_library('ccl')
        BuildCLib.run(self)
        # if ret_val==None:
        #     try:
        #         self.build_library()
        #     except:
        #         pass
        # if ret_val!=None:
        #     pass


# Creating the PyTest command
class PyTest(Command):
    # This assumes that the package is located in the default prefix.
    # If not, you have to add the path where you installed the C library
    # to DYLD_LIBARY_PATH or LD_LIBRARY_PATH depending on your platform.

    lib_env = 'LD_LIBARY_PATH'
    if 'Darwin' in platform.system():
        lib_env = 'DYLD_LIBRARY_PATH'
    if lib_env not in os.environ:
        libpath = os.getenv(lib_env, os.path.join(sys.prefix, 'lib'))
        os.environ[lib_env] = libpath
        libpath2 = os.path.realpath(os.path.join(site.USER_BASE, 'lib'))
        os.environ[lib_env] += os.path.join(libpath2)
    else:
        os.environ[lib_env] += os.pathsep + os.path.join(sys.prefix, 'lib')
        libpath2 = os.path.realpath(os.path.join(site.USER_BASE, 'lib'))
        os.environ[lib_env] += os.path.join(libpath2)

    user_options = []

    def initialize_options(self):

        pass

    def finalize_options(self):

        pass

    def run(self):

        errno = subprocess.call([sys.executable, 'tests/run_tests.py'])
        raise SystemExit(errno)


class PyUninstall(DistutilsInstall):
    def __init__(self, dist):
        DistutilsInstall.__init__(self, dist)
        self.build_args = {}
        if self.record is None:
            self.record = 'install-record.txt'

    def run(self):
        print("Removing...")
        os.system("cat %s | xargs rm -rfv" % self.record)

class PyInstall(DistutilsInstall):
    def __init__(self, dist):
        DistutilsInstall.__init__(self, dist)
        self.build_args = {}
        if self.record is None:
            self.record = 'install-record.txt'

    def check_extensions(self):
        ret_val = _check_extensions()
        return ret_val

    def build_library(self, library):
        plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
        if self.user:
            self.build_temp = os.path.join(self.install_userbase, 'temp' + plat_specifier)
        else:
            self.build_temp = os.path.join(self.prefix, 'temp' + plat_specifier)

        # Use a subdirectory of build_temp as the build directory.
        target_temp = os.path.realpath(os.path.join(self.build_temp, library))
        # Destination for headers and libraries is build_clib.
        if self.user:
            target_clib = os.path.realpath(self.install_userbase)
        else:
            target_clib = os.path.realpath(self.prefix)

        # Create build directories if they do not yet exist.
        mkpath(target_temp)
        mkpath(target_clib)

        env = _get_build_env()
        # Run configure.
        cmd = ['/bin/sh', os.path.join(os.path.dirname(__file__), 'configure'),
            '--prefix=' + target_clib]
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=env)
        # Run make install.
        cmd = ['make', 'install']
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=env)
        return target_clib

    def run(self):
        # Uncomment the line below if you want to check if the C library
        # is installed and in your path.
        # ret_val = self.check_extensions()
        lib_path = self.build_library('ccl')
        DistutilsInstall.run(self)


def _get_build_env():
    env = dict(os.environ)
    cc, cxx, opt, cflags = get_config_vars('CC', 'CXX', 'OPT', 'CFLAGS')
    if 'CFLAGS' in env:
        env['CFLAGS'] = opt + ' ' + env['CFLAGS']
    env['CXXFLAGS'] = cflags
    return env

def _check_extensions():
    """check if the C module can be built by trying to compile a small
    program against ccl"""

    libraries = ['ccl']

    # write a temporary .c file to compile
    c_code = dedent("""
    #include <stdio.h>
    #define CTEST_MAIN
    #include "ctest.h"
    int main(int argc, const char *argv[])
    {
      int result = ctest_main(argc, argv);
      return result;
    }
    """)
    tmp_dir = tempfile.mkdtemp(prefix='tmp_ccl_')
    bin_file_name = os.path.join(tmp_dir, 'test_ccl')
    file_name = bin_file_name + '.c'
    with open(file_name, 'w') as fp:
        fp.write(c_code)
    # and try to compile it
    compiler = ccompiler.new_compiler()
    assert isinstance(compiler, ccompiler.CCompiler)
    customize_compiler(compiler)

    try:
        compiler.link_executable(
            compiler.compile([file_name]),
            bin_file_name,
            libraries=libraries,
        )
    except CompileError:
        print('libccl compile error')
        ret_val = None
    except LinkError:
        print('libccl link error')
        ret_val = None
    else:
        ret_val = [Extension("_ccllib",
                   ["pyccl/ccl_wrap.c"],
                   libraries=['m', 'gsl', 'gslcblas', 'ccl'],
                   include_dirs=[numpy_include, "include/", "class/include"],
                   extra_compile_args=['-O4', '-std=c99'],
                   swig_opts=['-threads'],)]
    shutil.rmtree(tmp_dir)
    return ret_val

# CCL setup script

if "--user" in sys.argv:
    libdir=os.path.realpath(os.path.join(site.USER_BASE,'lib'))
elif "--prefix" in sys.argv:
    ii = np.where(np.array(sys.argv)=="--prefix")
    libdir=os.path.realpath(os.path.join(sys.argv[ii+1],'lib'))
else:
    libdir=os.path.realpath(os.path.join(sys.prefix,'lib'))
setup(name="pyccl",
    description="Library of validated cosmological functions.",
    author="LSST DESC",
    version="0.1",
    packages=['pyccl'],
    ext_modules=[
        Extension("_ccllib",["pyccl/ccl_wrap.c"],
                  libraries=['m','gsl','gslcblas','ccl'],
                  include_dirs=[numpy_include, "include/", "class/include"],
                  library_dirs=[libdir],
                  runtime_library_dirs=[libdir],
                  extra_compile_args=['-O4', '-std=c99'],
                  swig_opts=['-threads'], 
        )
    ],
    cmdclass={
        'install': PyInstall,
        'build_clib': BuildExternalCLib,
        'test': PyTest,
        'uninstall': PyUninstall
    }
    )
