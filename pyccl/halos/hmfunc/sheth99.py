from ... import ccllib as lib
from ...pyutils import check
from ..massdef import MassDef
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncSheth99",)


class MassFuncSheth99(MassFunc):
    """ Implements mass function described in arXiv:astro-ph/9901122
    This parametrization is only valid for 'fof' masses.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        use_delta_c_fit (bool): if True, use delta_crit given by
            the fit of Nakamura & Suto 1997. Otherwise use
            delta_crit = 1.68647.
    """
    __repr_attrs__ = ("mdef", "mass_def_strict", "use_delta_c_fit",)
    name = 'Sheth99'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super(MassFuncSheth99, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status, cosmo=cosmo)
        else:
            delta_c = 1.68647

        nu = delta_c / sigM
        return nu * self.A * (1. + (self.a * nu**2)**(-self.p)) * \
            np.exp(-self.a * nu**2/2.)
