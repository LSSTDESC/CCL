"""
========================
Base (:mod:`pyccl.base`)
========================

Core functionality of CCL:
    * parameters - Classes with module-level and object-specific parameters.
    * caching - Hashing and caching framework.
    * deprecations - Control deprecations.
    * repr_ - Custom object representations.
    * schema - Control the behavior of objects in the library.
"""

from .parameters import *
from .caching import *
from .schema import *
from .deprecations import *
from .repr_ import *
