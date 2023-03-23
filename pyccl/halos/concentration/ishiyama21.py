from ... import ccllib as lib
from ...base import warn_api
from ...pyutils import check
from ..massdef import MassDef
from ..halo_model_base import Concentration
import numpy as np
from scipy.optimize import brentq, root_scalar


__all__ = ("ConcentrationIshiyama21",)


class ConcentrationIshiyama21(Concentration):
    """ Concentration-mass relation by Ishiyama et al. 2021
    (arXiv:2007.14720). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-critical and 500-critical.
    By default it will be initialized for Delta = 500-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object that fixes the mass definition
            used by this c(M) parametrization.
        relaxed (bool):
            If True, use concentration for relaxed halos. Otherwise,
            use concentration for all halos. The default is False.
        Vmax (bool):
            If True, use the concentration found with the Vmax numerical
            method. Otherwise, use the concentration found with profile
            fitting. The default is False.
    """
    __repr_attrs__ = ("mass_def", "relaxed", "Vmax",)
    name = 'Ishiyama21'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=MassDef(500, 'critical'),
                 relaxed=False, Vmax=False):
        self.relaxed = relaxed
        self.Vmax = Vmax
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'vir':
            if isinstance(mass_def.Delta, str):
                return True
            elif mass_def.rho_type != 'critical':
                return True
            elif mass_def.Delta not in [200, 500]:
                return True
            elif (mass_def.Delta == 500) and self.Vmax:
                return True
        return False

    def _setup(self):
        if self.Vmax:  # use numerical method
            if self.relaxed:  # fit only relaxed halos
                if self.mass_def.Delta == 'vir':
                    self.kappa = 2.40
                    self.a0 = 2.27
                    self.a1 = 1.80
                    self.b0 = 0.56
                    self.b1 = 13.24
                    self.c_alpha = 0.079
                else:  # now it's 200c
                    self.kappa = 1.79
                    self.a0 = 2.15
                    self.a1 = 2.06
                    self.b0 = 0.88
                    self.b1 = 9.24
                    self.c_alpha = 0.51
            else:  # fit all halos
                if self.mass_def.Delta == 'vir':
                    self.kappa = 0.76
                    self.a0 = 2.34
                    self.a1 = 1.82
                    self.b0 = 1.83
                    self.b1 = 3.52
                    self.c_alpha = -0.18
                else:  # now it's 200c
                    self.kappa = 1.10
                    self.a0 = 2.30
                    self.a1 = 1.64
                    self.b0 = 1.72
                    self.b1 = 3.60
                    self.c_alpha = 0.32
        else:  # use profile fitting method
            if self.relaxed:  # fit only relaxed halos
                if self.mass_def.Delta == 'vir':
                    self.kappa = 1.22
                    self.a0 = 2.52
                    self.a1 = 1.87
                    self.b0 = 2.13
                    self.b1 = 4.19
                    self.c_alpha = -0.017
                else:  # now it's either 200c or 500c
                    if int(self.mass_def.Delta) == 200:
                        self.kappa = 0.60
                        self.a0 = 2.14
                        self.a1 = 2.63
                        self.b0 = 1.69
                        self.b1 = 6.36
                        self.c_alpha = 0.37
                    else:  # now it's 500c
                        self.kappa = 0.38
                        self.a0 = 1.44
                        self.a1 = 3.41
                        self.b0 = 2.86
                        self.b1 = 2.99
                        self.c_alpha = 0.42
            else:  # fit all halos
                if self.mass_def.Delta == 'vir':
                    self.kappa = 1.64
                    self.a0 = 2.67
                    self.a1 = 1.23
                    self.b0 = 3.92
                    self.b1 = 1.30
                    self.c_alpha = -0.19
                else:  # now it's either 200c or 500c
                    if int(self.mass_def.Delta) == 200:
                        self.kappa = 1.19
                        self.a0 = 2.54
                        self.a1 = 1.33
                        self.b0 = 4.04
                        self.b1 = 1.21
                        self.c_alpha = 0.22
                    else:  # now it's 500c
                        self.kappa = 1.83
                        self.a0 = 1.95
                        self.a1 = 1.17
                        self.b0 = 3.57
                        self.b1 = 0.91
                        self.c_alpha = 0.26

    def _dlsigmaR(self, cosmo, M, a):
        # kappa multiplies radius, so in log, 3*kappa multiplies mass
        logM = 3*np.log10(self.kappa) + np.log10(M)

        status = 0
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status, cosmo=cosmo)
        return -3/np.log(10) * dlns_dlogM

    def _G(self, x, n_eff):
        fx = np.log(1 + x) - x / (1 + x)
        G = x / fx**((5 + n_eff) / 6)
        return G

    def _G_inv(self, arg, n_eff):
        # Numerical calculation of the inverse of `_G`.
        roots = []
        for val, neff in zip(arg, n_eff):
            func = lambda x: self._G(x, neff) - val  # noqa: _G_inv Traceback
            try:
                rt = brentq(func, a=0.05, b=200)
            except ValueError:
                # No root in [0.05, 200] (rare, but it may happen).
                rt = root_scalar(func, x0=1, x1=2).root.item()
            roots.append(rt)
        return np.asarray(roots)

    def _concentration(self, cosmo, M, a):
        M_use = np.atleast_1d(M)

        nu = 1.686 / cosmo.sigmaM(M_use, a)
        n_eff = -2 * self._dlsigmaR(cosmo, M_use, a) - 3
        alpha_eff = cosmo.growth_rate(a)

        A = self.a0 * (1 + self.a1 * (n_eff + 3))
        B = self.b0 * (1 + self.b1 * (n_eff + 3))
        C = 1 - self.c_alpha * (1 - alpha_eff)
        arg = A / nu * (1 + nu**2 / B)
        G = self._G_inv(arg, n_eff)
        c = C * G

        if np.ndim(M) == 0:
            c = c[0]
        return c
