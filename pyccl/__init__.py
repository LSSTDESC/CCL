# flake8: noqa E402
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass  # not installed
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

from . import ccllib as lib
from .errors import *
from .base import *
from .pyutils import *

from .background import *
from .power import *

from .tracers import *
from .cells import *

from .pk2d import *
from .tk3d import *

from .correlations import *
from .covariances import *

from .boltzmann import *
from .baryons import *
from .neutrinos import *

from .bcm import *           # deprecated
from .halomodel import *     # deprecated
from .massfunction import *  # deprecated
from .haloprofile import *   # deprecated

from . import halos
from . import nl_pt

from .cosmology import *


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
