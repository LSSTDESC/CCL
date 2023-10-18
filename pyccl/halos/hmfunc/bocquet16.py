__all__ = ("MassFuncBocquet16",)

import numpy as np

from . import MassFunc


class MassFuncBocquet16(MassFunc):
    """Implements the mass function of `Bocquet et al. 2016
    <https://arxiv.org/abs/1502.07357>`_. This parametrization accepts
    S.O. masses with :math:`\\Delta = 200` with respect to the matter
    or critical densities, and :math:`\\Delta=500` with respect to the
    critical density.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
        hydro (:obj:`bool`): if ``False``, use the parametrization found
            using dark-matter-only simulations. Otherwise, include
            baryonic effects (default).
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict", "hydro",)
    _mass_def_strict_always = True
    name = 'Bocquet16'

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True,
                 hydro=True):
        self.hydro = hydro
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["200m", "200c", "500c"]

    def _setup(self):
        # key: (hydro, mass_def.name)
        vals = {(True, "200m"): (0.228, 2.15, 1.69, 1.30,
                                 0.285, -0.058, -0.366, -0.045),
                (False, "200m"): (0.175, 1.53, 2.55, 1.19,
                                  -0.012, -0.040, -0.194, -0.021),
                (True, "200c"): (0.202, 2.21, 2.00, 1.57,
                                 1.147, 0.375, -1.074, -0.196),
                (False, "200c"): (0.222, 1.71, 2.24, 1.46,
                                  0.269, 0.321, -0.621, -0.153),
                (True, "500c"): (0.180, 2.29, 2.44, 1.97,
                                 1.088, 0.150, -1.008, -0.322),
                (False, "500c"): (0.241, 2.18, 2.35, 2.02,
                                  0.370, 0.251, -0.698, -0.310)}

        key = (self.hydro, self.mass_def.name)
        self.A0, self.a0, self.b0, self.c0, \
            self.Az, self.az, self.bz, self.cz = vals[key]

    def _M200c_M200m(self, cosmo, a):
        # Translates the parameters of M200c to those of M200m
        # for which the base Bocquet16 model is defined.
        z = 1/a - 1
        Omega_m = cosmo.omega_x(a, "matter")
        gamma0 = 3.54E-2 + Omega_m**0.09
        gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
        gamma2 = 0.721 + 3.50E-2 / Omega_m
        gamma3 = 0.628 + 0.164 / Omega_m
        delta0 = -1.67E-2 + 2.18E-2 * Omega_m
        delta1 = 6.52E-3 - 6.86E-3 * Omega_m
        gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
        delta = delta0 + delta1 * z
        return gamma, delta

    def _M500c_M200m(self, cosmo, a):
        # Translates the parameters of M500c to those of M200m
        # for which the base Bocquet16 model is defined.
        z = 1/a - 1
        Omega_m = cosmo.omega_x(a, "matter")
        alpha0 = 0.880 + 0.329 * Omega_m
        alpha1 = 1.00 + 4.31E-2 / Omega_m
        alpha2 = -0.365 + 0.254 / Omega_m
        alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
        beta = -1.7E-2 + 3.74E-3 * Omega_m
        return alpha, beta

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        zp1 = 1./a
        AA = self.A0 * zp1**self.Az
        aa = self.a0 * zp1**self.az
        bb = self.b0 * zp1**self.bz
        cc = self.c0 * zp1**self.cz

        f = AA * ((sigM / bb)**-aa + 1.0) * np.exp(-cc / sigM**2)

        if self.mass_def.name == '200c':
            gamma, delta = self._M200c_M200m(cosmo, a)
            f *= gamma + delta * lnM
        elif self.mass_def.name == '500c':
            alpha, beta = self._M500c_M200m(cosmo, a)
            f *= alpha + beta * lnM

        return f
