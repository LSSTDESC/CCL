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
from ._core import *
from .pyutils import *

from .background import *
from .power import *

from .tracers import *
from .cells import *
from .correlations import *
from .covariances import *

from .pk2d import *
from .tk3d import *

from .boltzmann import *
from .baryons import *
from .neutrinos import *
from .emulators import *
from ._nonlimber_FKEM import *

from . import halos
from . import nl_pt

from .cosmology import *
