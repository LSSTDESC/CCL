from __future__ import annotations

__all__ = ("MassFuncPress74",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import MassFunc

if TYPE_CHECKING:
    from .. import MassDef


class MassFuncPress74(MassFunc):
    r"""Halo mass function by :footcite:t:`Press74`. Valid for FoF masses only.

    The mass function takes the form

    .. math::

        n(M, z) = \sqrt{\frac{2}{\pi}} \, \frac{\delta_c}{\sigma} \,
        \exp{\left[ -0.5 \left( \frac{\delta_c}{\sigma} \right)^2 \right]}.

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
    """
    name = 'Press74'

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[str, MassDef] = "fof",
            mass_def_strict: bool = True
    ):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self._norm = np.sqrt(2/np.pi)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = 1.68647
        nu = delta_c/sigM
        return self._norm * nu * np.exp(-0.5 * nu**2)
