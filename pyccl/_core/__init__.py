"""
========================
Base (:mod:`pyccl.base`)
========================

Core functionality of CCL:
    * :mod:`parameters` - Classes with global and object-specific parameters.
    * :mod:`caching` - Hashing and caching framework.
    * :mod:`deprecations` - Control deprecations.
    * :mod:`repr_` - Custom object representations.
    * :mod:`schema` - Control the behavior of objects in the library.
"""

from .parameters import *
from .caching import *
from .schema import *
from .deprecations import *
from .repr_ import *
