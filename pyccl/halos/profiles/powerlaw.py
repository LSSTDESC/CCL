from __future__ import annotations

__all__ = ("HaloProfilePowerLaw",)

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
class HaloProfilePowerLaw(HaloProfile):
    """ Power-law profile

    .. math::
        \\rho(r) = (r/r_s)^\\alpha

    Args:
        r_scale (:obj:`function`): the correlation length of
            the profile. The signature of this function
            should be `f(cosmo, M, a)`, where `cosmo`
            is a :class:`~pyccl.cosmology.Cosmology` object, `M` is a halo mass
            in units of M_sun, and `a` is the scale factor.
        tilt (:obj:`function`): the power law index of the
            profile. The signature of this function should
            be `f(cosmo, a)`.
    """
    __repr_attrs__ = __eq_attrs__ = ("r_scale", "tilt", "mass_def",
                                     "precision_fftlog",)

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
