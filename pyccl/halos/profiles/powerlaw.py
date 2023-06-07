from __future__ import annotations

__all__ = ("HaloProfilePowerLaw",)

from numbers import Real
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from numpy.typing import NDArray

from ... import warn_api, deprecated
from . import HaloProfile

if TYPE_CHECKING:
    from ... import Cosmology
    from .. import MassDef

    FuncSignature = Callable[[Cosmology,
                              Union[Real, NDArray[Real]],
                              Real],
                             NDArray[float]]


class HaloProfilePowerLaw(HaloProfile):
    r"""Power-law halo profile.

    .. math::

        \rho(r) = (r/r_s)^\alpha

    .. deprecated:: 2.8.0

        This profile will be removed in the next major release.

    Parameters
    ----------
    r_scale
        Correlation length of the profile.
    tilt
        Power law index of the profile.
    mass_def
        Halo mass definition.

        .. versionadded:: 2.8.0
    """
    __repr_attrs__ = __eq_attrs__ = ("r_scale", "tilt", "mass_def",
                                     "precision_fftlog",)

    @deprecated
    @warn_api
    def __init__(
            self,
            *,
            r_scale: FuncSignature,
            tilt: FuncSignature,
            mass_def: Union[str, MassDef] = None
    ):
        self.r_scale = r_scale
        self.tilt = tilt
        super().__init__(mass_def=mass_def)

    def _get_plaw_fourier(self, cosmo, a):
        # This is the optimal value for a pure power law
        # profile.
        return self.tilt(cosmo, a)

    def _get_plaw_projected(self, cosmo, a):
        # This is the optimal value for a pure power law
        # profile.
        return -3 - self.tilt(cosmo, a)

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_scale(cosmo, M_use, a)
        tilt = self.tilt(cosmo, a)
        # Form factor
        prof = (r_use[None, :] / rs[:, None])**tilt

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
