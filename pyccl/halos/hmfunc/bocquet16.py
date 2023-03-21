from ..massdef import MassDef200m
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncBocquet16",)


class MassFuncBocquet16(MassFunc):
    """ Implements mass function described in arXiv:1502.07357.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            Delta = 200 (matter, critical) and 500 (critical).
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        hydro (bool): if `False`, use the parametrization found
            using dark-matter-only simulations. Otherwise, include
            baryonic effects (default).
    """
    __repr_attrs__ = ("mdef", "mass_def_strict", "hydro",)
    name = 'Bocquet16'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 hydro=True):
        self.hydro = hydro
        super(MassFuncBocquet16, self).__init__(cosmo,
                                                mass_def,
                                                mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
        if int(self.mdef.Delta) == 200:
            if self.mdef.rho_type == 'matter':
                self.mdef_type = '200m'
            elif self.mdef.rho_type == 'critical':
                self.mdef_type = '200c'
        elif int(self.mdef.Delta) == 500:
            if self.mdef.rho_type == 'critical':
                self.mdef_type = '500c'
        if self.mdef_type == '200m':
            if self.hydro:
                self.A0 = 0.228
                self.a0 = 2.15
                self.b0 = 1.69
                self.c0 = 1.30
                self.Az = 0.285
                self.az = -0.058
                self.bz = -0.366
                self.cz = -0.045
            else:
                self.A0 = 0.175
                self.a0 = 1.53
                self.b0 = 2.55
                self.c0 = 1.19
                self.Az = -0.012
                self.az = -0.040
                self.bz = -0.194
                self.cz = -0.021
        elif self.mdef_type == '200c':
            if self.hydro:
                self.A0 = 0.202
                self.a0 = 2.21
                self.b0 = 2.00
                self.c0 = 1.57
                self.Az = 1.147
                self.az = 0.375
                self.bz = -1.074
                self.cz = -0.196
            else:
                self.A0 = 0.222
                self.a0 = 1.71
                self.b0 = 2.24
                self.c0 = 1.46
                self.Az = 0.269
                self.az = 0.321
                self.bz = -0.621
                self.cz = -0.153
        elif self.mdef_type == '500c':
            if self.hydro:
                self.A0 = 0.180
                self.a0 = 2.29
                self.b0 = 2.44
                self.c0 = 1.97
                self.Az = 1.088
                self.az = 0.150
                self.bz = -1.008
                self.cz = -0.322
            else:
                self.A0 = 0.241
                self.a0 = 2.18
                self.b0 = 2.35
                self.c0 = 2.02
                self.Az = 0.370
                self.az = 0.251
                self.bz = -0.698
                self.cz = -0.310

    def _check_mdef_strict(self, mdef):
        if isinstance(mdef.Delta, str):
            return True
        elif int(mdef.Delta) == 200:
            if (mdef.rho_type != 'matter') and \
               (mdef.rho_type != 'critical'):
                return True
        elif int(mdef.Delta) == 500:
            if mdef.rho_type != 'critical':
                return True
        else:
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        zp1 = 1./a
        AA = self.A0 * zp1**self.Az
        aa = self.a0 * zp1**self.az
        bb = self.b0 * zp1**self.bz
        cc = self.c0 * zp1**self.cz

        f = AA * ((sigM / bb)**-aa + 1.0) * np.exp(-cc / sigM**2)

        if self.mdef_type == '200c':
            z = 1./a-1
            Omega_m = cosmo.omega_x(a, "matter")
            gamma0 = 3.54E-2 + Omega_m**0.09
            gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
            gamma2 = 0.721 + 3.50E-2 / Omega_m
            gamma3 = 0.628 + 0.164 / Omega_m
            delta0 = -1.67E-2 + 2.18E-2 * Omega_m
            delta1 = 6.52E-3 - 6.86E-3 * Omega_m
            gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
            delta = delta0 + delta1 * z
            M200c_M200m = gamma + delta * lnM
            f *= M200c_M200m
        elif self.mdef_type == '500c':
            z = 1./a-1
            Omega_m = cosmo.omega_x(a, "matter")
            alpha0 = 0.880 + 0.329 * Omega_m
            alpha1 = 1.00 + 4.31E-2 / Omega_m
            alpha2 = -0.365 + 0.254 / Omega_m
            alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
            beta = -1.7E-2 + 3.74E-3 * Omega_m
            M500c_M200m = alpha + beta * lnM
            f *= M500c_M200m
        return f
