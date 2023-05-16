__all__ = ("MassFuncPress74",)

import numpy as np

from ... import warn_api
from . import MassFunc


class MassFuncPress74(MassFunc):
    """Implements the mass function of `Press & Schechter 1974
    <https://ui.adsabs.harvard.edu/abs/1974ApJ...187..425P/abstract>`_.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
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
