from __future__ import annotations

__all__ = ("MassFuncJenkins01",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import MassFunc

if TYPE_CHECKING:
    from .. import MassDef


class MassFuncJenkins01(MassFunc):
    r"""Halo mass function by :footcite:t:`Jenkins01`. Valid for FoF masses
    only.

    The mass function takes the form

    .. math::

        n(M) = 0.315 \, \exp{
            \left( -\left| \sigma^{-1} + 0.61 \right|^{3.8} \right)}.

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
    name = 'Jenkins01'

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
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.abs(-np.log(sigM) + self.b)**self.q)
