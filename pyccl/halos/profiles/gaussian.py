from ...base import warn_api
from .profile_base import HaloProfile
import numpy as np


__all__ = ("HaloProfileGaussian",)


class HaloProfileGaussian(HaloProfile):
    """ Gaussian profile

    .. math::
        \\rho(r) = \\rho_0\\, e^{-(r/r_s)^2}

    Args:
        r_scale (:obj:`function`): the width of the profile.
            The signature of this function should be
            `f(cosmo, M, a, mass_def)`, where `cosmo` is a
            :class:`~pyccl.core.Cosmology` object, `M` is a halo mass in
            units of M_sun, `a` is the scale factor and `mass_def`
            is a :class:`~pyccl.halos.massdef.MassDef` object.
        rho0 (:obj:`function`): the amplitude of the profile.
            It should have the same signature as `r_scale`.
    """
    __repr_attrs__ = ("r_scale", "rho_0", "precision_fftlog", "normprof",)
    normprof = False

    @warn_api
    def __init__(self, *, r_scale, rho0):
        self.rho_0 = rho0
        self.r_scale = r_scale
        super().__init__()
        self.update_precision_fftlog(padding_lo_fftlog=0.01,
                                     padding_hi_fftlog=100.,
                                     n_per_decade=10000)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_scale(cosmo, M_use, a, mass_def)
        # Compute normalization
        rho0 = self.rho_0(cosmo, M_use, a, mass_def)
        # Form factor
        prof = np.exp(-(r_use[None, :] / rs[:, None])**2)
        prof = prof * rho0[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
