from ...base import warn_api
from ... import ccllib as lib
from ...pyutils import check
from ..massdef import MassDef200m
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncDespali16",)


class MassFuncDespali16(MassFunc):
    """ Implements mass function described in arXiv:1507.05627.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts any SO masses.
            The default is '200m'.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        ellipsoidal (bool): use the ellipsoidal parametrization.
    """
    __repr_attrs__ = ("mass_def", "mass_def_strict", "ellipsoidal",)
    name = 'Despali16'

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef200m(),
                 mass_def_strict=True,
                 ellipsoidal=False):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self.ellipsoidal = ellipsoidal

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        x = np.log10(self.mass_def.get_Delta(cosmo, a) *
                     cosmo.omega_x(a, self.mass_def.rho_type) / Dv)

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
