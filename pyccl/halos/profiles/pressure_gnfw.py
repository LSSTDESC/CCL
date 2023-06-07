from __future__ import annotations

__all__ = ("HaloProfilePressureGNFW",)

from numbers import Real
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

from ... import warn_api
from . import HaloProfilePressure

if TYPE_CHECKING:
    from .. import MassDef


class HaloProfilePressureGNFW(HaloProfilePressure):
    r"""Generalized NFW pressure profile by :footcite:t:`Arnaud10`.

    The parametrization is:

    .. math::

       P_e(r) = C \times P_0 \, h_{70}^E \, (c_{500} x)^{-\gamma} \,
       [1+(c_{500}x)^\alpha]^{(\gamma-\beta)/\alpha},

    where

    .. math::

       C = 1.65 \, h_{70}^2 \left(\frac{H(z)}{H_0}\right)^{8/3}
       \left[\frac{h_{70}\tilde{M}_{500}}{3\times10^{14} \, {\rm M_\odot}}
       \right]^{2/3 + \alpha_{\rm P}},

    :math:`x = r/\tilde{r}_{500}`, :math:`h_{70}=h/0.7`, and the exponent
    :math:`E` is :math:`-1` for SZ-based profile normalizations and
    :math:`-1.5` for X-ray-based normalizations. The biased mass
    :math:`\tilde{M}_{500}` is related to the true overdensity mass
    :math:`M_{500}` via the mass bias parameter :math:`(1-b)` as
    :math:`\tilde{M}_{500} = (1-b) \, M_{500}`. :math:`\tilde{r}_{500}` is the
    overdensity halo radius associated with :math:`\tilde{M}_{500}` (note the
    intentional tilde!), and the profile is defined for a halo overdensity
    :math:`\Delta_{\rm 500c}`.

    The default parameters (other than `mass_bias`), correspond to those
    used in in Planck Intermediate Results (V) :footcite:p:`Planck13V_inter`.
    The profile is computed in physical (non-comoving) units of
    :math:`\rm eV / cm^3`.

    Parameters
    ----------
    mass_bias
        The mass bias parameter :math:`1-b`.
    P0
        Profile normalization.
    c500
        Concentration parameter.
    alpha, beta, gamma
        Profile shape parameter.
    alpha_P
        Additional mass dependence exponent
    P0_hexp
        Power of :math:`h` with which the normalization parameter scales.
        Equal to :math:`-1` for SZ-based normalizations,
        and :math:`-3/2` for X-ray-based normalizations.
    qrange
        Limits of integration used when computing the Fourier-space
        profile template, in units of :math:`R_{\\mathrm{vir}}`.
    nq
        Number of sampling points of the Fourier-space profile template.
    x_out
        Profile threshold, in units of :math:`R_{\rm 500c}`.
    mass_def
        Halo mass definition.

        .. versionadded:: 2.8.0

    References
    ----------
    .. footbibliography::
    """
    __repr_attrs__ = __eq_attrs__ = (
        "mass_bias", "P0", "c500", "alpha", "alpha_P", "beta", "gamma",
        "P0_hexp", "qrange", "nq", "x_out", "mass_def", "precision_fftlog",)

    @warn_api
    def __init__(
            self,
            *,
            mass_bias: Real = 0.8,
            P0: Real = 6.41,
            c500: Real = 1.81,
            alpha: Real = 1.33,
            alpha_P: Real = 0.12,
            beta: Real = 4.13,
            gamma: Real = 0.31,
            P0_hexp: Real = -1,
            qrange: Sequence[Real, Real] = (1e-3, 1e3),
            nq: int = 128,
            x_out: Real = np.inf,
            mass_def: Union[str, MassDef] = None
    ):
        self.qrange = qrange
        self.nq = nq
        self.mass_bias = mass_bias
        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.alpha_P = alpha_P
        self.beta = beta
        self.gamma = gamma
        self.P0_hexp = P0_hexp
        self.x_out = x_out

        # Interpolator for dimensionless Fourier-space profile
        self._fourier_interp = None
        super().__init__(mass_def=mass_def)

    # TODO: Uncomment for CCLv3.
    # @update(names=["mass_bias", "alpha_P", "P0", "P0_hexp"])
    # def update_parameters(self, *, alpha=None, beta=None, gamma=None,
    #                       c500=None, x_out=None) -> None:
    @warn_api
    def update_parameters(
            self, *, mass_bias=None, P0=None, c500=None, alpha=None, beta=None,
            gamma=None, alpha_P=None, P0_hexp=None, x_out=None):
        """Update the profile parameters. All numerical parameters except those
        related to the Fourier interpolation settings (``qrange, nq``) are
        updatable.
        """
        if mass_bias is not None:
            self.mass_bias = mass_bias
        if alpha_P is not None:
            self.alpha_P = alpha_P
        if P0 is not None:
            self.P0 = P0
        if P0_hexp is not None:
            self.P0_hexp = P0_hexp

        # Check if we need to recompute the Fourier profile.
        re_fourier = False
        if alpha is not None and alpha != self.alpha:
            re_fourier = True
            self.alpha = alpha
        if beta is not None and beta != self.beta:
            re_fourier = True
            self.beta = beta
        if gamma is not None and gamma != self.gamma:
            re_fourier = True
            self.gamma = gamma
        if c500 is not None and c500 != self.c500:
            re_fourier = True
            self.c500 = c500
        if x_out is not None and x_out != self.x_out:
            re_fourier = True
            self.x_out = x_out

        if re_fourier and (self._fourier_interp is not None):
            self._fourier_interp = self._integ_interp()

    def _form_factor(self, x):
        # Scale-dependent factor of the GNFW profile.
        f1 = (self.c500*x)**(-self.gamma)
        exponent = -(self.beta-self.gamma)/self.alpha
        f2 = (1+(self.c500*x)**self.alpha)**exponent
        return f1*f2

    def _integ_interp(self):
        # Precomputes the Fourier transform of the profile in terms
        # of the scaled radius x and creates a spline interpolator
        # for it.
        from scipy.interpolate import interp1d
        from scipy.integrate import quad

        def integrand(x):
            return self._form_factor(x)*x

        q_arr = np.geomspace(self.qrange[0], self.qrange[1], self.nq)
        # We use the `weight` feature of quad to quickly estimate
        # the Fourier transform. We could use the existing FFTLog
        # framework, but this is a lot less of a kerfuffle.
        f_arr = np.array([quad(integrand,
                               a=1e-4, b=self.x_out,  # limits of integration
                               weight="sin",  # fourier sine weight
                               wvar=q)[0] / q
                          for q in q_arr])
        Fq = interp1d(np.log(q_arr), f_arr,
                      fill_value="extrapolate",
                      bounds_error=False)
        return Fq

    def _norm(self, cosmo, M, a, mb):
        # Computes the normalization factor of the GNFW profile.
        # Normalization factor is given in units of eV/cm^3.
        # (Bolliet et al. 2017).
        h70 = cosmo["h"]/0.7
        C0 = 1.65*h70**2
        CM = (h70*M*mb/3E14)**(2/3+self.alpha_P)   # M dependence
        Cz = cosmo.h_over_h0(a)**(8/3)  # z dependence
        P0_corr = self.P0 * h70**self.P0_hexp  # h-corrected P_0
        return P0_corr * C0 * CM * Cz

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        # (1-b)
        mb = self.mass_bias
        # R_Delta*(1+z)
        R = self.mass_def.get_radius(cosmo, M_use * mb, a) / a

        nn = self._norm(cosmo, M_use, a, mb)
        prof = self._form_factor(r_use[None, :] / R[:, None])
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a):
        # Fourier-space profile.
        # Output in units of eV * Mpc^3 / cm^3.

        # Tabulate if not done yet
        if self._fourier_interp is None:
            with self.unlock():
                self._fourier_interp = self._integ_interp()

        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        mb = self.mass_bias
        # R_Delta*(1+z)
        R = self.mass_def.get_radius(cosmo, M_use*mb, a) / a

        ff = self._fourier_interp(np.log(k_use[None, :] * R[:, None]))
        nn = self._norm(cosmo, M_use, a, mb)

        prof = (4*np.pi*R**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
