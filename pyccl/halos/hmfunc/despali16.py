from ... import ccllib as lib
from ...pyutils import check
from ..massdef import MassDef200m
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncDespali16",)


class MassFuncDespali16(MassFunc):
    """ Implements mass function described in arXiv:1507.05627.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts any SO masses.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    __repr_attrs__ = __eq_attrs__ = ("mdef", "mass_def_strict", "ellipsoidal",)
    name = 'Despali16'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 ellipsoidal=False):
        super(MassFuncDespali16, self).__init__(cosmo,
                                                mass_def,
                                                mass_def_strict)
        self.ellipsoidal = ellipsoidal

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
        pass

    def _check_mdef_strict(self, mdef):
        if mdef.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        x = np.log10(self.mdef.get_Delta(cosmo, a) *
                     cosmo.omega_x(a, self.mdef.rho_type) / Dv)

        if self.ellipsoidal:
            A = -0.1768 * x + 0.3953
            a = 0.3268 * x**2 + 0.2125 * x + 0.7057
            p = -0.04570 * x**2 + 0.1937 * x + 0.2206
        else:
            A = -0.1362 * x + 0.3292
            a = 0.4332 * x**2 + 0.2263 * x + 0.7665
            p = -0.1151 * x**2 + 0.2554 * x + 0.2488

        nu = delta_c/sigM
        nu_p = a * nu**2

        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * \
            np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)
