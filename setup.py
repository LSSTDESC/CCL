#!/usr/bin/env python

import os
import sys
from subprocess import check_output, CalledProcessError, check_call
from distutils.sysconfig import get_config_var, get_config_vars
from distutils.core import *
from distutils.util import get_platform
from distutils.command.install import install as DistutilsInstall
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsExecError
from distutils.dir_util import mkpath
from distutils.file_util import copy_file
from distutils import log
from distutils.command.install import install as DistutilsInstall
import numpy
from textwrap import dedent
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

#This part is taken from healpy's setup.py

# Apple switched default C++ standard libraries (from gcc's libstdc++ to
# clang's libc++), but some pre-packaged Python environments such as Anaconda
# are built against the old C++ standard library. Luckily, we don't have to
# actually detect which C++ standard library was used to build the Python
# interpreter. We just have to propagate MACOSX_DEPLOYMENT_TARGET from the
# configuration variables to the environment.

if get_config_var('MACOSX_DEPLOYMENT_TARGET') and not 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = get_config_var('MACOSX_DEPLOYMENT_TARGET')


#######################
class build_external_clib(build_clib):
    """Subclass of Distutils' standard build_clib subcommand. Adds support for
    libraries that are installed externally and detected with pkg-config, with
    an optional fallback to build from a local configure-make-install style
    distribution."""

    def __init__(self, dist):
        build_clib.__init__(self, dist)
        self.build_args = {}
    def check_extensions(self):
        """check if the C module can be built by trying to compile a small
        program against ccl"""

        import tempfile
        import shutil
        import distutils.sysconfig
        import distutils.ccompiler
        from distutils.errors import CompileError, LinkError

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
        tmp_dir = tempfile.mkdtemp(prefix = 'tmp_ccl_')
        bin_file_name = os.path.join(tmp_dir, 'test_ccl')
        file_name = bin_file_name + '.c'
        with open(file_name, 'w') as fp:
            fp.write(c_code)
        # and try to compile it
        compiler = distutils.ccompiler.new_compiler()
        assert isinstance(compiler, distutils.ccompiler.CCompiler)
        distutils.sysconfig.customize_compiler(compiler)

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
                       ["pyccl/ccl.i",],
                       libraries = ['m', 'gsl', 'gslcblas', 'ccl'],
                       include_dirs = [numpy_include, "include/", "class/include"],
                       extra_compile_args=['-O4', '-std=c99'],
                       swig_opts=['-threads'],)]
        shutil.rmtree(tmp_dir)
        return ret_val

    def build_library(self,library):
        env = dict(os.environ)
        cc, cxx, opt, cflags = get_config_vars('CC', 'CXX', 'OPT', 'CFLAGS')
        cxxflags = cflags
        if 'CC' in env:
            cc = env['CC']
        if 'CXX' in env:
            cxx = env['CXX']
        if 'CFLAGS' in env:
            cflags = opt + ' ' + env['CFLAGS']
        if 'CXXFLAGS' in env:
            cxxflags = opt + ' ' + env['CXXFLAGS']

        # Use a subdirectory of build_temp as the build directory.
        build_temp = os.path.realpath(os.path.join(self.build_temp, library))
        # Destination for headers and libraries is build_clib.
        build_clib = os.path.realpath(self.build_clib)

        # Create build directories if they do not yet exist.
        mkpath(build_temp)
        mkpath(build_clib)

        # Run configure.
        cmd = ['/bin/sh', os.path.join(os.path.dirname(__file__), 'configure'),
        '--prefix=' + build_clib,
        '--disable-shared',
        '--with-pic',
        '--disable-maintainer-mode']

        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=dict(env,
            CC=cc, CXX=cxx, CFLAGS=cflags, CXXFLAGS=cxxflags))

        # Run make install.
        cmd = ['make', 'install']
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=dict(env,
            CC=cc, CXX=cxx, CFLAGS=cflags, CXXFLAGS=cxxflags))
        return build_clib

    def run(self):
        #Uncomment the line below if you want to check if the C library
        #is installed and in your path.
        #ret_val = self.check_extensions()
        build_path = self.build_library('ccl')
        build_clib.run(self)
        #if ret_val==None:
        #    try:
        #        self.build_library()
        #    except:
        #        pass
        #if ret_val!=None:
        #    pass


#Creating the PyTest command
class PyTest(Command):
    import sys
    import platform
    import site
    #This assumes that the package is located in the default prefix.
    #If not, you have to add the path where you installed the C library
    #To DYLD_LIBRARY_PATH or LD_LIBRARY_PATH depending on your platform
    if 'Linux' in platform.system():
        if not 'LD_LIBRARY_PATH' is os.environ:
            libpath = os.getenv('LD_LIBARY_PATH',os.path.join(sys.prefix,'lib'))
            os.environ['LD_LIBRARY_PATH']=libpath
            libpath2 = os.path.realpath(os.path.join(site.USER_BASE,'lib'))
            os.environ['LD_LIBRARY_PATH']+=os.path.join(libpath2)
        else:
            os.environ['LD_LIBRARY_PATH']+= os.pathsep + os.path.join(sys.prefix,'lib')
            libpath2 = os.path.realpath(os.path.join(site.USER_BASE,'lib'))
            os.environ['LD_LIBRARY_PATH']+=os.path.join(libpath2)
    if 'Darwin' in platform.system():
        if not 'DYLD_LIBRARY_PATH' is os.environ:
            libpath = os.getenv('DYLD_LIBARY_PATH',os.path.join(sys.prefix,'lib'))
            os.environ['DYLD_LIBRARY_PATH']=libpath
            libpath2 = os.path.realpath(os.path.join(site.USER_BASE,'lib'))
            os.environ['DYLD_LIBRARY_PATH']+=os.path.join(libpath2)
        else:
            os.environ['DYLD_LIBRARY_PATH']+= os.pathsep + os.path.join(sys.prefix,'lib')
            libpath2 = os.path.realpath(os.path.join(site.USER_BASE,'lib'))
            os.environ['DYLD_LIBRARY_PATH']+=os.path.join(libpath2)

    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/run_tests.py'])
        raise SystemExit(errno)
class PyUninstall(DistutilsInstall):
    def __init__(self,dist):
        DistutilsInstall.__init__(self,dist)
        self.build_args = {}
        if self.record==None:
            self.record='install-record.txt'
    def run(self):
        print "Removing..."
        os.system("cat %s | xargs rm -rfv" % self.record)
class PyInstall(DistutilsInstall):
    def __init__(self, dist):
        DistutilsInstall.__init__(self, dist)
        self.build_args = {}
        if self.record==None:
            self.record='install-record.txt'
    def check_extensions(self):
        """check if the C module can be built by trying to compile a small
        program against ccl"""

        import tempfile
        import shutil
        import distutils.sysconfig
        import distutils.ccompiler
        from distutils.errors import CompileError, LinkError

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
        tmp_dir = tempfile.mkdtemp(prefix = 'tmp_ccl_')
        bin_file_name = os.path.join(tmp_dir, 'test_ccl')
        file_name = bin_file_name + '.c'
        with open(file_name, 'w') as fp:
            fp.write(c_code)
        # and try to compile it
        compiler = distutils.ccompiler.new_compiler()
        assert isinstance(compiler, distutils.ccompiler.CCompiler)
        distutils.sysconfig.customize_compiler(compiler)

        try:
            compiler.link_executable(
                compiler.compile([file_name]),
                bin_file_name,
                libraries=libraries,
            )
        except CompileError:
            print('libccl compile error. Building C library')
            ret_val = None
        except LinkError:
            print('libccl link error')
            ret_val = None
        else:
            ret_val = [Extension("_ccllib",
                       ["pyccl/ccl.i",],
                       libraries = ['m', 'gsl', 'gslcblas', 'ccl'],
                       include_dirs = [numpy_include, "include/", "class/include"],
                       extra_compile_args=['-O4', '-std=c99'],
                       swig_opts=['-threads'],)]
        shutil.rmtree(tmp_dir)
        return ret_val

    def build_library(self,library):
        plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
        if self.user:
            self.build_temp = os.path.join(self.install_userbase,'temp' + plat_specifier)
        else:
            self.build_temp = os.path.join(self.prefix,'temp' + plat_specifier)
        env = dict(os.environ)
        cc, cxx, opt, cflags = get_config_vars('CC', 'CXX', 'OPT', 'CFLAGS')
        cxxflags = cflags
        if 'CC' in env:
            cc = env['CC']
        if 'CXX' in env:
            cxx = env['CXX']
        if 'CFLAGS' in env:
            cflags = opt + ' ' + env['CFLAGS']
        if 'CXXFLAGS' in env:
            cxxflags = opt + ' ' + env['CXXFLAGS']
        # Use a subdirectory of build_temp as the build directory.
        build_temp = os.path.realpath(os.path.join(self.build_temp, library))
        # Destination for headers and libraries is build_clib.
        if self.user:
            build_clib = os.path.realpath(self.install_userbase)
        else:
            build_clib = os.path.realpath(self.prefix)

        # Create build directories if they do not yet exist.
        mkpath(build_temp)
        mkpath(build_clib)

        # Run configure.
        cmd = ['/bin/sh', os.path.join(os.path.dirname(__file__), 'configure'),
        '--prefix=' + build_clib]
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=dict(env,
            CC=cc, CXX=cxx, CFLAGS=cflags, CXXFLAGS=cxxflags))
        # Run make install.
        cmd = ['make', 'install']
        log.info('%s', ' '.join(cmd))
        check_call(cmd, cwd='./', env=dict(env,
            CC=cc, CXX=cxx, CFLAGS=cflags, CXXFLAGS=cxxflags))
        return build_clib
    def run(self):
        #Uncomment the line below if you want to check if the C library
        #is installed and in your path.
        #ret_val = self.check_extensions()
        lib_path = self.build_library('ccl')
        DistutilsInstall.run(self)

# CCL setup script
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.1",
        packages     = ['pyccl'],
        ext_modules = [Extension(
            "_ccllib",
               ["pyccl/ccl.i",],
               libraries = ['m', 'gsl', 'gslcblas', 'ccl'],
               include_dirs = [numpy_include, "include/", "class/include"],
               extra_compile_args=['-O4', '-std=c99'],
               swig_opts=['-threads'],
           )],
        cmdclass = {
            'install': PyInstall,
            'build_clib': build_external_clib,
            'test': PyTest,
            'uninstall': PyUninstall
        },
        )
