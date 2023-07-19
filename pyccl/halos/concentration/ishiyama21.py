__all__ = ("ConcentrationIshiyama21",)

import numpy as np
from scipy.optimize import brentq, root_scalar

from ... import lib
from ... import check
from . import Concentration, get_delta_c


class ConcentrationIshiyama21(Concentration):
    """Concentration-mass relation by `Ishiyama et al. 2021
    <http://arxiv.org/abs/2007.14720>`_. This parametrization is only
    valid for S.O. masses with :math:`\\Delta = \\Delta_{\\rm vir}`, or
    :math:`\\Delta=200` or :math:`500` times the critical density.
    By default it will be initialized for :math:`M_{500c}`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object or a name string.
        relaxed (:obj:`bool`):
            If ``True``, use concentration for relaxed halos. Otherwise,
            use concentration for all halos. Default: ``False``.
        Vmax (:obj:`bool`):
            If ``True``, use the concentration found with the "Vmax"
            numerical method. Otherwise, use the concentration found with
            profile fitting. Default:  ``False``.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "relaxed", "Vmax",)
    name = 'Ishiyama21'

    def __init__(self, *, mass_def="500c",
                 relaxed=False, Vmax=False):
        self.relaxed = relaxed
        self.Vmax = Vmax
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        is_500Vmax = mass_def.Delta == 500 and self.Vmax
        return mass_def.name not in ["vir", "200c", "500c"] or is_500Vmax

    def _setup(self):
        # key: (Vmax, relaxed, Delta)
        vals = {(True, True, 200): (1.79, 2.15, 2.06, 0.88, 9.24, 0.51),
                (True, False, 200): (1.10, 2.30, 1.64, 1.72, 3.60, 0.32),
                (False, True, 200): (0.60, 2.14, 2.63, 1.69, 6.36, 0.37),
                (False, False, 200): (1.19, 2.54, 1.33, 4.04, 1.21, 0.22),
                (True, True, "vir"): (2.40, 2.27, 1.80, 0.56, 13.24, 0.079),
                (True, False, "vir"): (0.76, 2.34, 1.82, 1.83, 3.52, -0.18),
                (False, True, "vir"): (1.22, 2.52, 1.87, 2.13, 4.19, -0.017),
                (False, False, "vir"): (1.64, 2.67, 1.23, 3.92, 1.30, -0.19),
                (False, True, 500): (0.38, 1.44, 3.41, 2.86, 2.99, 0.42),
                (False, False, 500): (1.83, 1.95, 1.17, 3.57, 0.91, 0.26)}

        key = (self.Vmax, self.relaxed, self.mass_def.Delta)
        self.kappa, self.a0, self.a1, \
            self.b0, self.b1, self.c_alpha = vals[key]

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
        nu = get_delta_c(cosmo, a, 'EdS_approx') / cosmo.sigmaM(M, a)
        n_eff = -2 * self._dlsigmaR(cosmo, M, a) - 3
        alpha_eff = cosmo.growth_rate(a)

        A = self.a0 * (1 + self.a1 * (n_eff + 3))
        B = self.b0 * (1 + self.b1 * (n_eff + 3))
        C = 1 - self.c_alpha * (1 - alpha_eff)
        arg = A / nu * (1 + nu**2 / B)
        G = self._G_inv(arg, n_eff)
        return C * G
