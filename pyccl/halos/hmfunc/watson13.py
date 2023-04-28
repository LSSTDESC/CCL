from __future__ import annotations

__all__ = ("MassFuncWatson13",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import warn_api
from . import MassFunc

if TYPE_CHECKING:
    from .. import MassDef


class MassFuncWatson13(MassFunc):
    r"""Halo mass function by :footcite:t:`Watson13`. Valid for any S.O. and
    FoF masses.

    The mass function takes the form

    .. math::

        \frac{{\rm d}n}{{\rm d}M} = \frac{\bar{\rho}_{\rm m}}{M^2} \, f(\sigma)
        \, \frac{{\rm d} \ln \sigma^{-1}}{{\rm d} \ln M}.

    where

    .. math::

        f(\sigma) = A \, \left[\left( \frac{\beta}{\sigma} \right)^\alpha + 1
        \right] \, \exp \left( -\frac{\gamma}{\sigma^2} \right),

    For FoF masses, :math:`A`, :math:`\alpha`, :math:`\beta`, :math:`\gamma`
    are fitted parameters. For S.O. masses these parameters experience a time-
    dependent modified power law evolution up to :math:`z=6`, after which they
    are fixed. The modified power law has the form

    .. math::

        X(z) = \Omega_{\rm m}(z) \left( x_1 \times (1+z)^{-x_2} + x_3 \right),

    where :math:`x_1`, :math:`x_2`, and :math:`x_3` are fitted parameters.

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
    name = 'Watson13'

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[str, MassDef] = "200m",
            mass_def_strict: bool = True
    ):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name == "vir"

    def _get_fsigma_fof(self, cosmo, sigM, a, lnM):
        pA = 0.282
        pa = 2.163
        pb = 1.406
        pc = 1.210
        return pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)

    def _get_fsigma_SO(self, cosmo, sigM, a, lnM):
        om = cosmo.omega_x(a, "matter")
        Delta_178 = self.mass_def.Delta / 178

        # TODO: this has to be vectorized with numpy
        if a == 1:
            pA = 0.194
            pa = 1.805
            pb = 2.267
            pc = 1.287
        elif a < 1/(1+6):
            pA = 0.563
            pa = 3.810
            pb = 0.874
            pc = 1.453
        else:
            pA = om * (1.097 * a**3.216 + 0.074)
            pa = om * (5.907 * a**3.058 + 2.349)
            pb = om * (3.136 * a**3.599 + 2.344)
            pc = 1.318

        f_178 = pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)
        C = np.exp(0.023 * (Delta_178 - 1.0))
        d = -0.456 * om - 0.139
        Gamma = (C * Delta_178**d *
                 np.exp(0.072 * (1.0 - Delta_178) / sigM**2.130))
        return f_178 * Gamma

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.mass_def.name == 'fof':
            return self._get_fsigma_fof(cosmo, sigM, a, lnM)
        return self._get_fsigma_SO(cosmo, sigM, a, lnM)
