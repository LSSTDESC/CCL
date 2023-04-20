# flake8: noqa E402
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError

# Set the environment variable for default config path
from os import environ, path
if environ.get("CLASS_PARAM_DIR") is None:
    environ["CLASS_PARAM_DIR"] = path.dirname(path.abspath(__file__))
del environ, path

# Patch for deprecated alias in Numpy >= 1.20.0 (used in ISiTGR & FAST-PT).
# Deprecation cycle starts in Numpy 1.20 and ends in Numpy 1.24.
from packaging.version import parse
import numpy
numpy.int = int if parse(numpy.__version__) >= parse("1.20.0") else numpy.int
del parse, numpy

DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"

from . import ccllib as lib
from .errors import *
from .base import *
from .base.parameters import *
from .cosmology import *
from .background import *
from .boltzmann import *
from .pk2d import *
from .tk3d import *
from .power import *
from .neutrinos import *
from .cells import *
from .tracers import *
from .correlations import *
from .covariances import *
from .pyutils import debug_mode, resample_array

# Deprecated & Renamed modules
from .bcm import *
from .halomodel import *
from .massfunction import *
from .haloprofile import *
from .baryons import *

def __getattr__(name):
    rename = {"core": "cosmology", "cls": "cells"}
    if name in rename:
        from .errors import CCLDeprecationWarning
        import warnings
        warnings.warn(f"Module {name} has been renamed to {rename[name]}.",
                      CCLDeprecationWarning)
        name = rename[name]
        return eval(name)
    raise AttributeError(f"No module named {name}.")
