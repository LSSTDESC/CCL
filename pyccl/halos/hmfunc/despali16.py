from ...base import warn_api
from ... import ccllib as lib
from ...pyutils import check
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncDespali16",)


class MassFuncDespali16(MassFunc):#
    r"""Halo mass function by Despali et al. (2016) :arXiv:1507.05627.
    Valid for any S.O. masses.

    The mass function takes the form

    .. math::

        1 + 1 = 2

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`200{\rm m}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
    ellipsoidal : bool, optional
        Whether to use the fit parameters found by running an Ellipsoidal
        Overdensity finder. The default is True.
    """
    __repr_attrs__ = ("mass_def", "mass_def_strict", "ellipsoidal",)
    name = 'Despali16'

    @warn_api
    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True,
                 ellipsoidal=False):
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
