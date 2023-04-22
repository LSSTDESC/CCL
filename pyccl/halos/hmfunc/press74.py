__all__ = ("MassFuncPress74",)

import numpy as np

from ... import warn_api
from . import MassFunc


class MassFuncPress74(MassFunc):
    r"""Halo mass function by Press et al. (1974) 1974ApJ...187..425P.
    Valid for FoF masses only.

    The mass function takes the form

    .. math::

        n(M, z) = \sqrt{\frac{2}{\pi}} \, \frac{\delta_c}{\sigma} \,
        \exp{\left[ -0.5 \left( \frac{\delta_c}{\sigma} \right)^2 \right]}.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`{\rm FoF}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
    """
    name = 'Press74'

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self._norm = np.sqrt(2/np.pi)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = 1.68647
        nu = delta_c/sigM
        return self._norm * nu * np.exp(-0.5 * nu**2)
