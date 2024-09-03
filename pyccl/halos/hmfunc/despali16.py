__all__ = ("MassFuncDespali16",)

import numpy as np

from . import MassFunc, get_delta_c


class MassFuncDespali16(MassFunc):
    """Implements the mass function of `Despali et al. 2016
    <https://arxiv.org/abs/1507.05627>`_. This parametrization accepts
    any S.O. masses.


    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
        ellipsoidal (:obj:`bool`): use the ellipsoidal parametrization.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",
                                     "ellipsoidal",)
    name = 'Despali16'

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
        coeffs = [[A1, A0], [a2, a1, a0], [p2, p1, p0]]
        self.poly_A, self.poly_a, self.poly_p = map(np.poly1d, coeffs)

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = get_delta_c(cosmo, a, 'NakamuraSuto97')

        Dv = self.mass_def.get_Delta_vir(cosmo, a)
        x = np.log10(self.mass_def.get_Delta(cosmo, a) / Dv)

        A, a, p = self.poly_A(x), self.poly_a(x), self.poly_p(x)

        nu_p = a * (delta_c/sigM)**2
        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * (
            np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p))
