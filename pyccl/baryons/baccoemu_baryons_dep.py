__all__ = ("BaccoemuBaryons",)

import numpy as np
from copy import deepcopy
from warnings import warn

from .. import Pk2D
from . import BaryonsBaccoemu

class BaccoemuBaryons(BaryonsBaccoemu):
    name = 'BaccoemuBaryons'
    def __init__(self, *args, **kwargs):
        """This throws a deprecation warning on initialization."""
        warn(f'Class {self.__class__.__name__} will be deprecated. Please use {BaryonsBaccoemu.__name__} instead.', DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
    pass