from ...base import warn_api
from .profile_base import HaloProfile
import numpy as np

__all__ = ("HaloProfilePowerLaw",)


class HaloProfilePowerLaw(HaloProfile):
    """ Power-law profile

    .. math::
        \\rho(r) = (r/r_s)^\\alpha

    Args:
        r_scale (:obj:`function`): the correlation length of
            the profile. The signature of this function
            should be `f(cosmo, M, a, mass_def)`, where `cosmo`
            is a :class:`~pyccl.core.Cosmology` object, `M` is a halo mass
            in units of M_sun, `a` is the scale factor and
            `mass_def` is a :class:`~pyccl.halos.massdef.MassDef` object.
        tilt (:obj:`function`): the power law index of the
            profile. The signature of this function should
            be `f(cosmo, a)`.
    """
    __repr_attrs__ = ("r_scale", "tilt", "precision_fftlog", "normprof",)
    normprof = False

    @warn_api
    def __init__(self, *, r_scale, tilt):
        self.r_scale = r_scale
        self.tilt = tilt
        super().__init__()

    def _get_plaw_fourier(self, cosmo, a):
        # This is the optimal value for a pure power law
        # profile.
        return self.tilt(cosmo, a)

    def _get_plaw_projected(self, cosmo, a):
        # This is the optimal value for a pure power law
        # profile.
        return -3 - self.tilt(cosmo, a)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_scale(cosmo, M_use, a, mass_def)
        tilt = self.tilt(cosmo, a)
        # Form factor
        prof = (r_use[None, :] / rs[:, None])**tilt

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
