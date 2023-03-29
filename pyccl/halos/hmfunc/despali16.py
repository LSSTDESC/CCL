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
        return mass_def.Delta == "fof"

    def _setup(self):
        # key: (ellipsoidal)
        vals = {True: (0.3953, -0.1768, 0.7057, 0.2125, 0.3268,
                       0.2206, 0.1937, -0.04570),
                False: (0.3292, -0.1362, 0.7665, 0.2263, 0.4332,
                        0.2488, 0.2554, -0.1151)}
        self.A0, self.A1, self.a0, self.a1, self.a1, \
            self.p0, self.p1, self.p2 = vals[self.ellipsoidal]

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        x = np.log10(self.mass_def.get_Delta(cosmo, a) *
                     cosmo.omega_x(a, self.mass_def.rho_type) / Dv)

        A = self.A1 * x + self.A0
        a = self.a2 * x**2 + self.a1 * x + self.a0
        p = self.p2 * x**2 + self.p1 * x + self.p0

        nu_p = a * (delta_c/sigM)**2
        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * (
            np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p))
