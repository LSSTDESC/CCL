__all__ = ("HaloBiasBhattacharya11",)

from . import HaloBias, get_delta_c


class HaloBiasBhattacharya11(HaloBias):
    """ Implements halo bias as described in `Bhattacharya et al. 2011
    <https://arxiv.org/abs/1005.2239>`_. This parametrization is only
    valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = "Bhattacharya11"

    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = get_delta_c(None, None, kind='EdS')

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1 + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc
