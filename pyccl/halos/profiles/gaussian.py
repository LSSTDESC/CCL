from __future__ import annotations

__all__ = ("HaloProfileGaussian",)

from numbers import Real
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import numpy.typing as npt

from ... import warn_api, deprecated
from . import HaloProfile

if TYPE_CHECKING:
    from ... import Cosmology
    from .. import MassDef

    FuncSignature = Callable[
        [Cosmology, Union[Real, npt.NDArray], Real],
        npt.NDArray]


@deprecated
class HaloProfileGaussian(HaloProfile):
    """ Gaussian profile

    .. math::
        \\rho(r) = \\rho_0\\, e^{-(r/r_s)^2}

    Args:
        r_scale (:obj:`function`): the width of the profile.
            The signature of this function should be
            `f(cosmo, M, a)`, where `cosmo` is a
            :class:`~pyccl.cosmology.Cosmology` object, `M` is a halo mass in
            units of M_sun, and `a` is the scale factor.
        rho0 (:obj:`function`): the amplitude of the profile.
            It should have the same signature as `r_scale`.
    """
    __repr_attrs__ = __eq_attrs__ = ("r_scale", "rho_0", "mass_def",
                                     "precision_fftlog",)

    @warn_api
    def __init__(
            self,
            *,
            r_scale: FuncSignature,
            rho0: FuncSignature,
            mass_def: Union[str, MassDef, None] = None
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
