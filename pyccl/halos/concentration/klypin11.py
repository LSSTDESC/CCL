__all__ = ("ConcentrationKlypin11",)

from . import Concentration


class ConcentrationKlypin11(Concentration):
    """Concentration-mass relation by `Klypin et al. 2011
    <https://arxiv.org/abs/1002.3660>`_. This parametrization is only
    valid for S.O. masses with :math:`\\Delta = \\Delta_{\\rm vir}`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
            definition object or a name string.
    """
    name = 'Klypin11'

    def __init__(self, *, mass_def="vir"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "vir"

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo["h"] * 1E-12
        return 9.6 * (M * M_pivot_inv)**(-0.075)
