__all__ = ("HaloBiasSheth99",)

from . import HaloBias, get_delta_c


class HaloBiasSheth99(HaloBias):
    """Implements halo bias as described in `Sheth & Tormen 1999
    <https://arxiv.org/abs/astro-ph/9901122>`_.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
        use_delta_c_fit (:obj:`bool`): if ``True``, use the fit to the
            critical overdensity :math:`\\delta_c` by
            `Nakamura & Suto 1997
            <https://arxiv.org/abs/astro-ph/9612074>`_. Otherwise use
            :math:`\\delta_c = 1.68647`.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",
                                     "use_delta_c_fit",)
    name = "Sheth99"

    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.p = 0.3
        self.a = 0.707

    def _get_bsigma(self, cosmo, sigM, a):
        if self.use_delta_c_fit:
            delta_c = get_delta_c(cosmo, a, kind='NakamuraSuto97')
        else:
            delta_c = get_delta_c(cosmo, a, kind='EdS')

        nu = delta_c / sigM
        anu2 = self.a * nu**2
        return 1 + (anu2 - 1. + 2. * self.p / (1. + anu2**self.p))/delta_c
