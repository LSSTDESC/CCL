__all__ = ("HaloBiasSheth01",)

from . import HaloBias, get_delta_c


class HaloBiasSheth01(HaloBias):
    """Implements halo bias as described in `Sheth et al. 2001
    <https://arxiv.org/abs/astro-ph/9907024>`_. This
    parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = "Sheth01"

    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a = 0.707
        self.sqrta = 0.84083292038
        self.b = 0.5
        self.c = 0.6
        self.dc = get_delta_c(None, None, kind='EdS')

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1 + (self.sqrta * anu2 * (1 + self.b / anu2c) -
                    anu2c / (anu2c + t1)) / (self.sqrta * self.dc)
