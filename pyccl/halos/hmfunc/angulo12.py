from __future__ import annotations

__all__ = ("MassFuncAngulo12",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import MassFunc

if TYPE_CHECKING:
    from .. import MassDef


class MassFuncAngulo12(MassFunc):
    r"""Halo mass function by :footcite:t:`Angulo12`. Valid for FoF masses
    only.

    The mass function takes the form

    .. math::

        n(M, z) = A \times \left[ \frac{a}{\sigma} + 1 \right]^b
        \exp{ \left[ \frac{-c}{\sigma^2} \right]},

    where :math:`(A, a, b, c)` = (0.201, 2.08, 1.7, 1.172)` are fitted
    parameters.

    Parameters
    ----------
    mass_def
        Mass definition for this :math:`n(M)` parametrization.
    mass_def_strict
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.

    References
    ----------
    .. footbibliography::

    Attributes
    ----------
    mass_def

    mass_def_strict
    """
    name = 'Angulo12'

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[str, MassDef] = "fof",
            mass_def_strict: bool = True
    ):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * (
            np.exp(-self.c / sigM**2))
