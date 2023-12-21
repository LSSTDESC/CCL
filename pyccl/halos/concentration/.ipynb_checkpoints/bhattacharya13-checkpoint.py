<<<<<<< HEAD
from ... import ccllib as lib
from ...base import warn_api
from ..halo_model_base import Concentration


__all__ = ("ConcentrationBhattacharya13",)


class ConcentrationBhattacharya13(Concentration):
    """ Concentration-mass relation by Bhattacharya et al. 2013
    (arXiv:1112.5479). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or str): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization, or a name string.
    """
    name = 'Bhattacharya13'

    @warn_api(pairs=[("mdef", "mass_def")])
=======
__all__ = ("ConcentrationBhattacharya13",)

from . import Concentration, get_delta_c


class ConcentrationBhattacharya13(Concentration):
    """Concentration-mass relation by `Bhattacharya et al. 2013
    <https://arxiv.org/abs/1112.5479>`_. This parametrization is only valid for
    S.O. masses with :math:`\\Delta = \\Delta_{\\rm vir}`, or
    :math:`\\Delta=200` times the critical or matter densities. By default it
    will be initialized for :math:`M_{200c}`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
            definition object or name string.
    """
    name = 'Bhattacharya13'

>>>>>>> master
    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["vir", "200m", "200c"]

    def _setup(self):
        vals = {"vir": (7.7, 0.9, -0.29),
                "200m": (9.0, 1.15, -0.29),
                "200c": (5.9, 0.54, -0.35)}

        self.A, self.B, self.C = vals[self.mass_def.name]

    def _concentration(self, cosmo, M, a):
        gz = cosmo.growth_factor(a)
<<<<<<< HEAD
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
=======
        delta_c = get_delta_c(cosmo, a, kind='NakamuraSuto97')
>>>>>>> master
        sig = cosmo.sigmaM(M, a)
        nu = delta_c / sig
        return self.A * gz**self.B * nu**self.C
