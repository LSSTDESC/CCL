import os
import sys
from subprocess import call

from setuptools import Distribution as _distribution
from setuptools import setup
from setuptools.command.build_py import build_py as _build
from setuptools.command.develop import develop as _develop


def _compile_ccl(debug=False):
    call(["mkdir", "-p", "build"])
    v = sys.version_info
    cmd = ["cmake", "-H.", "-Bbuild",
           "-DPYTHON_VERSION=%d.%d.%d" % (v.major, v.minor, v.micro)]
    if debug:
        cmd += ["-DCMAKE_BUILD_TYPE=Debug"]
    if call(cmd) != 0:
        raise Exception(
            "Could not run CMake configuration. Make sure "
            "CMake is installed !")

    if call(["make", "-Cbuild", "_ccllib"]) != 0:
        raise Exception("Could not build CCL")

    # Finds the library under its different possible names
    if os.path.exists("build/pyccl/_ccllib.so"):
        call(["cp", "build/pyccl/_ccllib.so", "pyccl/"])
    else:
        raise Exception("Could not find wrapper shared library, "
                        "compilation must have failed.")
    if call(["cp", "build/pyccl/ccllib.py", "pyccl/"]) != 0:
        raise Exception("Could not find python module, "
                        "SWIG must have failed.")


class Distribution(_distribution):
    global_options = _distribution.global_options

    global_options += [
        ("debug", None, "Debug build"),
    ]

    def __init__(self, attr=None):
        self.debug = False
        super().__init__(attr)


class Build(_build):
    """Specialized Python source builder."""

    def run(self):
        _compile_ccl(debug=self.distribution.debug)
        _build.run(self)


class Develop(_develop):
    """Specialized Python develop mode."""
    def run(self):
        _compile_ccl(debug=self.distribution.debug)
        _develop.run(self)


setup(
    distclass=Distribution,
    setup_requires=['setuptools_scm'],
    cmdclass={'build_py': Build, 'develop': Develop},
)
