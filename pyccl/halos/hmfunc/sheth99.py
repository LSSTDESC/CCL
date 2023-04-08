from ... import ccllib as lib
from ...base import warn_api
from ...base.parameters import physical_constants as const
from ...pyutils import check
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncSheth99",)


class MassFuncSheth99(MassFunc):
    """ Implements mass function described in arXiv:astro-ph/9901122
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or str):
            a mass definition object, or a name string.
            This parametrization accepts FoF masses only.
            The default is 'fof'.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        use_delta_c_fit (bool): if True, use delta_c given by
            the fit of Nakamura & Suto 1997. Otherwise use
            delta_c = 1.68647.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",
                                     "use_delta_c_fit",)
    name = 'Sheth99'

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status, cosmo=cosmo)
        else:
            delta_c = const.DELTA_C

        nu = delta_c / sigM
        return nu * self.A * (1. + (self.a * nu**2)**(-self.p)) * (
            np.exp(-self.a * nu**2/2.))
