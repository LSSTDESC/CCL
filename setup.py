#!/usr/bin/env python
# This setup file has been adapted from healpy's setup file
import shlex
import os
import fnmatch
import errno
import sys
from subprocess import check_output, CalledProcessError, check_call
from distutils.sysconfig import get_config_var, get_config_vars
from distutils.core import *
from distutils import sysconfig
import os.path
from distutils.command.install import install as DistutilsInstall
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsExecError
from distutils.dir_util import mkpath
from distutils.file_util import copy_file
from distutils import log
from setuptools.command.build_ext import build_ext
from distutils.core import Extension
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# Apple switched default C++ standard libraries (from gcc's libstdc++ to
# clang's libc++), but some pre-packaged Python environments such as Anaconda
# are built against the old C++ standard library. Luckily, we don't have to
# actually detect which C++ standard library was used to build the Python
# interpreter. We just have to propagate MACOSX_DEPLOYMENT_TARGET from the
# configuration variables to the environment.
#
# This workaround fixes <https://github.com/healpy/healpy/issues/151>.
if get_config_var('MACOSX_DEPLOYMENT_TARGET') and not 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = get_config_var('MACOSX_DEPLOYMENT_TARGET')

class build_external_clib(build_clib):
    """Subclass of Distutils' standard build_clib subcommand. Adds support for
    libraries that are installed externally and detected with pkg-config, with
    an optional fallback to build from a local configure-make-install style
    distribution."""

    def __init__(self, dist):
        build_clib.__init__(self, dist)
        self.build_args = {}

    def autotools_path(self):
        """
        Install Autotools locally, because
        we will be building from git and need to generate the build
        system.
        """
        on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
        if not on_rtd:
            return None

        log.info("checking if autotools is installed")
        try:
            check_output(['autoreconf', '--version'])
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            log.info("autotools is not installed")
        else:
            log.info("autotools is already installed")
            return None

        urls = [
            'https://ftp.gnu.org/gnu/m4/m4-1.4.17.tar.gz',
            'https://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz',
            'https://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz',
            'https://ftp.gnu.org/gnu/automake/automake-1.15.tar.gz',
            'https://pkg-config.freedesktop.org/releases/pkg-config-0.29.1.tar.gz'
        ]

        # Use a subdirectory of build_temp as the build directory.
        build_temp = os.path.realpath(self.build_temp)
        prefix = os.path.join(build_temp, 'autotools')
        mkpath(prefix)

        env = dict(os.environ)
        path = os.path.join(prefix, 'bin')
        try:
            path += ':' + env['PATH']
        except KeyError:
            pass
        env['PATH'] = path

        # Check again if autotools is available, now that we have added the
        # temporary path.
        try:
            check_output(['autoreconf', '--version'], env=env)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            log.info("building autotools from source")
        else:
            log.info("using autotools built from source")
            return path

        # Otherwise, build from source.
        for url in urls:
            _, _, tarball = url.rpartition('/')
            pkg_version = tarball.replace('.tar.gz', '')
            log.info('downloading %s', url)
            check_call(['curl', '-O', url], cwd=build_temp)
            log.info('extracting %s', tarball)
            check_call(['tar', '-xzf', tarball], cwd=build_temp)
            cwd = os.path.join(build_temp, pkg_version)
            log.info('configuring %s', pkg_version)
            check_call(['./configure', '--prefix', prefix, '--with-internal-glib'], env=env, cwd=cwd)
            log.info('making %s', pkg_version)
            check_call(['make', 'install'], env=env, cwd=cwd)
        return path

    def env(self):
        """Construct an environment dictionary suitable for having pkg-config
        pick up .pc files in the build_clib directory."""
        # Test if pkg-config is present. If not, fall back to pykg-config.
        try:
            env = self._env
        except AttributeError:
            env = dict(os.environ)

            path = self.autotools_path()
            if path is not None:
                env['PATH'] = path

            try:
                check_output(['pkg-config', '--version'])
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
                log.warn('pkg-config is not installed, falling back to pykg-config')
                env['PKG_CONFIG'] = sys.executable + ' ' + os.path.abspath('run_pykg_config.py')
                
            else:
                env['PKG_CONFIG'] = 'pkg-config'

            build_clib = os.path.realpath(self.build_clib)
            pkg_config_path = (
                os.path.join(build_clib, 'lib64', 'pkgconfig') +
                ':' + os.path.join(build_clib, 'lib', 'pkgconfig'))
            try:
                pkg_config_path += ':' + env['PKG_CONFIG_PATH']
            except KeyError:
                pass
            env['PKG_CONFIG_PATH'] = pkg_config_path

            self._env = env
        return env
    def pkgconfig(self, *packages):
        env = self.env()
        PKG_CONFIG = tuple(shlex.split(
            env['PKG_CONFIG'], posix=(os.sep == '/')))
        kw = {} 
        index_key_flag = ()
        for index, flag, keys in index_key_flag:
            cmd = PKG_CONFIG + (flag,) + tuple(packages)
            log.debug('%s', ' '.join(cmd))
            args = [token[index:].decode() for token in check_output(cmd, env=env).split()]
            if args:
                for key in keys:
                    kw.setdefault(key, []).extend(args)
        return kw
 
    def finalize_options(self):
        """Run 'autoreconf -i' for any bundled libraries to generate the
        configure script."""
        build_clib.finalize_options(self)
        env = self.env()

        for lib_name, build_info in self.libraries:
            if 'sources' not in build_info:
                log.info("checking if configure script for library '%s' exists", lib_name)
                if not os.path.exists(os.path.join(build_info['local_source'], 'configure')):
                    log.info("running 'autoreconf -i' for library '%s'", lib_name)
                    check_call(['autoreconf', '-i'], cwd=build_info['local_source'], env=env)

    def build_library(self, library, pkg_config_name, local_source=None, supports_non_srcdir_builds=True):
        log.info("checking if library '%s' is installed", library)
        try:
            build_args = self.pkgconfig(pkg_config_name)
            log.info("found '%s' installed, using it", library)
        except CalledProcessError:

            # If local_source is not specified, then immediately fail.
            if local_source is None:
                raise DistutilsExecError("library '%s' is not installed", library)

            log.info("building library '%s' from source", library)

            env = self.env()

            # Determine which compilers we are to use, and what flags.
            # This is based on what distutils.sysconfig.customize_compiler()
            # does, but that function has a problem that it doesn't produce
            # necessary (e.g. architecture) flags for C++ compilers.
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

            if not supports_non_srcdir_builds:
                self._stage_files_recursive(local_source, build_temp)

            # Run configure.
            cmd = ['/bin/sh', os.path.join(os.path.realpath(local_source), 'configure'),
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
            check_call(cmd, cwd='./', env=env)

            build_args = self.pkgconfig(pkg_config_name)

        return build_args
        # Done!

    @staticmethod
    def _list_files_recursive(path, skip=('.*', '*.o', 'autom4te.cache')):
        """Yield paths to all of the files contained within the given path,
        following symlinks. If skip is a tuple of fnmatch()-style wildcard
        strings, skip any directory or filename matching any of the patterns in
        skip."""
        for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
            if not any(any(fnmatch.fnmatch(p, s) for s in skip) for p in dirpath.split(os.sep)):
                for filename in filenames:
                    if not any(fnmatch.fnmatch(filename, s) for s in skip):
                        yield os.path.join(dirpath, filename)

    @staticmethod
    def _stage_files_recursive(src, dest, skip=None):
        """Hard link or copy all of the files in the path src into the path dest.
        Subdirectories are created as needed, and files in dest are overwritten."""
        # Use hard links if they are supported on this system.
        if hasattr(os, 'link'):
            link='hard'
        elif hasattr(os, 'symlink'):
            link='sym'
        else:
            link=None

        for dirpath, dirnames, filenames in os.walk(src, followlinks=True):
            if not any(p.startswith('.') for p in dirpath.split(os.sep)):
                dest_dirpath = os.path.join(dest, dirpath.split(src, 1)[1].lstrip(os.sep))
                mkpath(dest_dirpath)
                for filename in filenames:
                    if not filename.startswith('.'):
                        src_path = os.path.join(dirpath, filename)
                        dest_path = os.path.join(dest_dirpath, filename)
                        if not os.path.exists(dest_path):
                            copy_file(os.path.join(dirpath, filename), os.path.join(dest_dirpath, filename))

    def get_source_files(self):
        """Copied from Distutils' own build_clib, but modified so that it is not
        an error for a build_info dictionary to lack a 'sources' key. If there
        is no 'sources' key, then all files contained within the path given by
        the 'local_sources' value are returned."""
        self.check_library_list(self.libraries)
        filenames = []
        for (lib_name, build_info) in self.libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                sources = list(self._list_files_recursive(build_info['local_source']))

            filenames.extend(sources)
        return filenames

    def build_libraries(self, libraries):
        # Build libraries that have no 'sources' key, accumulating the output
        # from pkg-config.
        for lib_name, build_info in libraries:
            if 'sources' not in build_info:
                for key, value in self.build_library(lib_name, **build_info).items():
                    if key in self.build_args:
                        self.build_args[key].extend(value)
                    else:
                        self.build_args[key] = value

        # Use parent method to build libraries that have a 'sources' key.
        build_clib.build_libraries(self, ((lib_name, build_info)
            for lib_name, build_info in libraries if 'sources' in build_info))


class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        # Make sure that Numpy is importable
        # (does the same thing as setup_requires=['numpy'])
        self.distribution.fetch_build_eggs('numpy')
        # Prevent numpy from thinking it is still in its setup process:
        # See http://stackoverflow.com/questions/19919905
        __builtins__.__NUMPY_SETUP__ = False

        # Add Numpy header search path path
        import numpy
        self.include_dirs.append(numpy.get_include())
        
    def run(self):
        # If we were asked to build any C/C++ libraries, add the directory
        # where we built them to the include path. (It's already on the library
        # path.)
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            for key, value in build_clib.build_args.items():
                for ext in self.extensions:
                    if not hasattr(ext, key) or getattr(ext, key) is None:
                        setattr(ext, key, value)
                    else:
                        getattr(ext, key).extend(value)
        build_ext.run(self)


#Creating the PyTest command
class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/run_tests.py'])
        raise SystemExit(errno)


# CCL setup script
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.1",
        packages     = ['pyccl'], 
        libraries    = [
            ('ccl',{
             'pkg_config_name': 'ccl',
             'local_source': './'})],
        ext_modules = [Extension(
            "_ccllib",
               ["pyccl/ccl.i",],
               libraries = ['m', 'gsl', 'gslcblas', 'ccl'],
               include_dirs = [numpy_include, "include/", "class/include"],
               extra_compile_args=['-O4', '-std=c99'],
               swig_opts=['-threads'],
           )],
        cmdclass = {
            'test': PyTest,
            'build_ext': custom_build_ext,
            'build_clib': build_external_clib,
        },
   
        )
