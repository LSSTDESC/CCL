from __future__ import annotations

__all__ = ("MassFuncDespali16",)

from typing import TYPE_CHECKING, Union

import numpy as np

from ... import check, lib, warn_api
from . import MassFunc

if TYPE_CHECKING:
    from .. import MassDef


class MassFuncDespali16(MassFunc):
    r"""Halo mass function by :footcite:t:`Despali16`. Valid for any S.O.
    masses.

    The mass function takes the form

    .. math::

        f(M, z) = f(\nu) \frac{{\rm d}\nu}{{\rm d}M}

    where :math:`\nu \equiv \delta_{\rm c}^2 / \sigma^2` is the peak height
    of the density field and

    .. math::

        \nu f(\nu) = A \left( 1 + \frac{1}{\nu'^p} \right)
        \left( \frac{\nu'}{2\pi} \right)^{1/2} \exp (-\nu'/2),

    with :math:`(A, a, p)` representing the polynomials

    .. math::

        A &= A_1 x + A_0, \\
        a &= a_2 x^2 + a_1 x + a_0, \\
        p &= p_2 x^2 + p_1 x + p_0,

    where :math:`x \equiv \log_{10}(\Delta(z_0) / \Delta_{\rm vir}(z_0=0))`,
    and the polynomial coefficients are fitted parameters.

    Parameters
    ----------
    mass_def
        Mass definition for this :math:`n(M)` parametrization.
    mass_def_strict
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
    ellipsoidal
        Whether to use the fit parameters found by running an Ellipsoidal
        Overdensity finder.

    References
    ----------
    .. footbibliography::
    """
    __repr_attrs__ = __eq_attrs__ = (
        "mass_def", "mass_def_strict", "ellipsoidal",)
    name = 'Despali16'
    ellispoidal: bool

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[str, MassDef] = "200m",
            mass_def_strict: bool = True,
            ellipsoidal: bool = False
    ):
        self.ellipsoidal = ellipsoidal
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        # True for FoF since Despali16 is not defined for this mass def.
        return mass_def.Delta == "fof"

    def _setup(self):
        # key: ellipsoidal
        vals = {True: (0.3953, -0.1768, 0.7057, 0.2125, 0.3268,
                       0.2206, 0.1937, -0.04570),
                False: (0.3292, -0.1362, 0.7665, 0.2263, 0.4332,
                        0.2488, 0.2554, -0.1151)}

        A0, A1, a0, a1, a2, p0, p1, p2 = vals[self.ellipsoidal]
        coeffs = [[A1, A0], [a2, a1, a0], [p2, p2, p0]]
        self.poly_A, self.poly_a, self.poly_p = map(np.poly1d, coeffs)

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        x = np.log10(self.mass_def.get_Delta(cosmo, a) *
                     cosmo.omega_x(a, self.mass_def.rho_type) / Dv)

        A, a, p = self.poly_A(x), self.poly_a(x), self.poly_p(x)

        nu_p = a * (delta_c/sigM)**2
        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * (
            np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p))
