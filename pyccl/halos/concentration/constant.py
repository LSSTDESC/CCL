from __future__ import annotations

__all__ = ("ConcentrationConstant",)

from numbers import Real
from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import Concentration

if TYPE_CHECKING:
    from .. import MassDef


class ConcentrationConstant(Concentration):
    """Constant contentration-mass relation.

    Parameters
    ---------
    c
        Value of the constant concentration.
        The default is :math:`1.0`.
    mass_def
        The mass definition for this :math:`c(M)` parametrization is arbitrary
        and is not used for any calculations.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "c",)
    name = 'Constant'
    c: Real

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, c: Real = 1, *, mass_def: Union[str, MassDef] = "200c"):
        self.c = c
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return np.full_like(M, self.c)[()]
