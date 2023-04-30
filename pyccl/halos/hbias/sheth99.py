from __future__ import annotations

__all__ = ("HaloBiasSheth99",)

from typing import TYPE_CHECKING, Union

from ... import check, lib, warn_api
from . import HaloBias

if TYPE_CHECKING:
    from .. import MassDef


class HaloBiasSheth99(HaloBias):
    r"""Halo bias relation by :footcite:t:`Sheth99`. Valid for FoF masses only.

    The halo bias takes the form

    .. math::

        b(M, z) = 1 + \frac{a\nu - 1}{\delta_{\rm c}}
        + \frac{2p / \delta_{\rm c}}{1 + (a\nu)^p},

    where :math:`\nu(M, z) = \delta_{\rm c}(z)^2 / \sigma(M, z)^2`, and
    :math:`(a, p) = (0.707, 0.3)` are fitted parameters.

    Parameters
    ----------
    mass_def
        Mass definition for this :math:`b(M)` parametrization.
    mass_def_strict
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
    use_delta_c_fit
        If True, use the formula for :math:`\delta_{\rm c}` given by the
        fit of Nakamura & Suto (1997). If False, use
        :math:`\delta_{\rm c} \simeq 1.68647` given by spherical collapse
        theory.

    References
    ----------
    .. footbibliography::
    """
    __repr_attrs__ = __eq_attrs__ = (
        "mass_def", "mass_def_strict", "use_delta_c_fit",)
    name = "Sheth99"
    use_delta_c_fit: bool

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[str, MassDef] = "fof",
            mass_def_strict: bool = True,
            use_delta_c_fit: bool = False
    ):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.p = 0.3
        self.a = 0.707

    def _get_bsigma(self, cosmo, sigM, a):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status, cosmo=cosmo)
        else:
            delta_c = 1.68647

        nu = delta_c / sigM
        anu2 = self.a * nu**2
        return 1 + (anu2 - 1. + 2. * self.p / (1. + anu2**self.p))/delta_c
