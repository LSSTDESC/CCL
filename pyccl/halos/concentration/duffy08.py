__all__ = ("ConcentrationDuffy08",)

from . import Concentration


class ConcentrationDuffy08(Concentration):
    """Concentration-mass relation by `Duffy et al. 2008
    <https://arxiv.org/abs/0804.2486>`_. This parametrization is only
    valid for S.O. masses with :math:`\\Delta = \\Delta_{\\rm vir}`,
    of :math:`\\Delta=200` times the matter or critical density.
    By default it will be initialized for :math:`M_{200c}`.

    Args:
         fc_bar (:obj:`float'): an optional constant that multiplies
            the Duffy. et al. relation to mimic the impact of baryons
            (default value is set to 1). See Amon+, 2202.07440
            and Viola+, 1507.00735 (Eq 32).
         mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
            definition object, or a name string.
    """
    name = 'Duffy08'

    def __init__(self, fc_bar=1, *, mass_def="200c"):
        self.fc_bar = fc_bar
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["vir", "200m", "200c"]

    def _setup(self):
        vals = {"vir": (7.85, -0.081, -0.71),
                "200m": (10.14, -0.081, -1.01),
                "200c": (5.71, -0.084, -0.47)}

        self.A, self.B, self.C = vals[self.mass_def.name]

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo["h"] * 5E-13
        return self.fc_bar * self.A * (M * M_pivot_inv)**self.B * a**(-self.C)
