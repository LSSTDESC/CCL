__all__ = ("MassFuncJenkins01",)

import numpy as np

from ... import warn_api
from . import MassFunc


class MassFuncJenkins01(MassFunc):
    r"""Halo mass function by `Jenkins et al. (2001)
    <https://arxiv.org/abs/astro-ph/0005260>`_. Valid for FoF masses only.

    The mass function takes the form

    .. math::

        n(M) = 0.315 \, \exp{
            \left( -\left| \sigma^{-1} + 0.61 \right|^{3.8} \right)}.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`{\rm FoF}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
    """
    name = 'Jenkins01'

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.abs(-np.log(sigM) + self.b)**self.q)
