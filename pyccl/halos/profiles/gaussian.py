from __future__ import annotations

__all__ = ("HaloProfileGaussian",)

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


class HaloProfileGaussian(HaloProfile):
    r""" Gaussian profile

    .. math::

        \rho(r) = \rho_0 \, e^{-(r/r_s)^2}

    .. deprecated:: 2.8.0

        This profile will be removed in the next major release.

    Parameters
    ----------
    r_scale
        The width of the profile.
    rho0
        The amplitude of the profile.
    mass_def
        Halo mass definition.

        .. versionadded:: 2.8.0
    """
    __repr_attrs__ = __eq_attrs__ = ("r_scale", "rho_0", "mass_def",
                                     "precision_fftlog",)

    @deprecated
    @warn_api
    def __init__(
            self,
            *,
            r_scale: FuncSignature,
            rho0: FuncSignature,
            mass_def: Union[str, MassDef] = None
    ):
        self.rho_0 = rho0
        self.r_scale = r_scale
        super().__init__(mass_def=mass_def)
        self.update_precision_fftlog(padding_lo_fftlog=0.01,
                                     padding_hi_fftlog=100.,
                                     n_per_decade=10000)

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_scale(cosmo, M_use, a)
        # Compute normalization
        rho0 = self.rho_0(cosmo, M_use, a)
        # Form factor
        prof = np.exp(-(r_use[None, :] / rs[:, None])**2)
        prof = prof * rho0[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
