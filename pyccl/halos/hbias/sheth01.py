from __future__ import annotations

__all__ = ("HaloBiasSheth01",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import HaloBias

if TYPE_CHECKING:
    from .. import MassDef


class HaloBiasSheth01(HaloBias):
    r"""Halo bias relation by :footcite:t:`Sheth01`. Valid for FoF masses only.

    The halo bias takes the form

    .. math::

        b(M, z) = 1 + \frac{1}{\sqrt{a}\delta_{\rm c}} \left[ \sqrt{a}
        \left( a\nu^2 \right) + \sqrt{a} b \left( a\nu^2 \right)^{1-c}
        - \frac{ \left( a\nu^2 \right)^c}{ \left( a\nu^2 \right)^c
                                          + b (1-c) (1-c/2)} \right],

    where :math:`\nu(M, z) = \delta_{\rm c}(z) / \sigma(M, z)` is the peak
    height of the density field and :math:`(a,b,c) = (0.707, 0.5, 0.6)`
    are fitted parameters.

    Parameters
    ----------
    mass_def
        Mass definition for this :math:`b(M)` parametrization.
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
    name = "Sheth01"

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
        self.a = 0.707
        self.sqrta = np.sqrt(self.a)
        self.b = 0.5
        self.c = 0.6
        self.dc = 1.68647

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1 + (self.sqrta * anu2 * (1 + self.b / anu2c) -
                    anu2c / (anu2c + t1)) / (self.sqrta * self.dc)
