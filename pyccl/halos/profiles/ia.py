from __future__ import annotations

__all__ = ("SatelliteShearHOD",)

import warnings
from numbers import Real
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.special import binom, gamma, spherical_jn

from ... import CCLWarning, update
from ...pyutils import _spline_integrate
from . import HaloProfileHOD

if TYPE_CHECKING:
    from ... import Cosmology
    from .. import Concentration, HMCalculator, MassDef


class SatelliteShearHOD(HaloProfileHOD):
    r"""HOD profile that calculates the satellite galaxy intrinsic shear field,
    following :footcite:t:`Fortuna21`.

    It can be used to compute halo model intrinsic alignment (angular) power
    spectra. The satellite intrinsic shear profile in real space is

    .. math::

        \gamma^I(r)=a_{1 \rm h} \left(\frac{r}{r_{\rm vir}} \right)^b
        \sin^b \theta,

    where :math:`a_{1 \rm h}` is the amplitude of intrinsic alignments on the
    1-halo scale, :math:`b` the index defining the radial dependence, and
    :math:`\theta` is the angle defining the projection of the semi-major axis
    of the galaxy along the line of sight.

    Parameters
    ----------
    concentration
        Concentration-mass relation. If a string, `mass_def` must be specified.
    a1h
        Amplitude of the satellite intrinsic shear profile.
    b
        Power-law index of the satellite intrinsic shear profile.
        :math:`0` indicates a constant profile inside the halo.
    l_max
        Maximum multipole to be summed in the plane-wave expansion
        (Eq. C1 of :footcite:t:`Fortuna21`).
    log10Mmin_0, log10Mmin_0
        :math:`\log_{10}M_{\rm min}`.
    siglnM_0, siglnM_p
        :math:`\sigma_{{\rm ln}M}`.
    log10M0_0, log10M0_p
        :math:`\log_{10}M_0`.
    log10M1_0, log10M1_p
        :math:`\log_{10}M_1`.
    alpha_0, alpha_p
        :math:`\alpha`.
    bg_0, bg_p
        :math:`\beta_g`.
    bmax_0, bmax_p
        :math:`\beta_{\rm max}`.
    a_pivot
        Pivot scale factor :math:`a_*`.
    ns_independent
        If True, relax the requirement to only form satellites when centrals
        are present.
    mass_def
        Halo mass definition. If `concentration` is instantiated, this
        parameter is optional.
    integration_method
        Fourier integration method.
    r_min:
        For ``'simpson'`` or ``'spline'`` integration, minimum value of the
        physical radius to integrate the profile (in :math:`\rm Mpc`).
    N_r:
        For ``'simpson'`` or `spline` integration, number of sampling points
        of the radial integral, in log-space.
    N_jn:
        For ``'simpson'`` or ``'spline'`` integration, number of sampling
        points for the spherical Bessel functions.

    References
    ----------
    .. footbibliography::
    """
    __repr_attrs__ = __eq_attrs__ = (
        "a1h", "b", "l_max", "integration_method",
        "log10Mmin_0", "log10Mmin_p", "siglnM_0", "siglnM_p", "log10M0_0",
        "log10M0_p", "log10M1_0", "log10M1_p", "alpha_0", "alpha_p",
        "bg_0", "bg_p", "bmax_0", "bmax_p", "a_pivot", "ns_independent",
        "r_min", "N_r", "N_jn", "concentration", "integration_method",
        "precision_fftlog",)

    def __init__(
            self,
            *,
            concentration: Concentration,
            a1h: Real = 0.001,
            b: Real = -2,
            l_max: int = 6,
            log10Mmin_0: Real = 12, log10Mmin_p: Real = 0,
            siglnM_0: Real = 0.4, siglnM_p: Real = 0,
            log10M0_0: Real = 7, log10M0_p: Real = 0,
            log10M1_0: Real = 13.3, log10M1_p: Real = 0,
            alpha_0: Real = 1, alpha_p: Real = 0,
            bg_0: Real = 1, bg_p: Real = 0,
            bmax_0: Real = 1, bmax_p: Real = 0,
            a_pivot: Real = 1,
            ns_independent: bool = False,
            mass_def: Optional[MassDef] = None,
            integration_method: Literal[
                "FFTLog", "simpson", "spline"] = 'FFTLog',
            r_min: Real = 0.001,
            N_r: int = 512,
            N_jn: int = 10000
    ):
        if not 2 < l_max < 13:  # clip l_max
            warnings.warn("l_max out of bounds. Clipping to [2, 13].",
                          CCLWarning)
            l_max = np.clip(l_max, 2, 13)
        if l_max % 2 != 0:  # odd ell contributions are zero
            l_max -= 1

        # Hard-code for most common cases (b=0, b=-2) for speed.
        if b == 0:
            self._angular_fl = np.array([2.77582637, -0.19276603,
                                         0.04743899, -0.01779024,
                                         0.00832446, -0.00447308])[:, None]
        elif b == -2:
            self._angular_fl = np.array([4.71238898, -2.61799389,
                                         2.06167032, -1.76714666,
                                         1.57488973, -1.43581368])[:, None]
        else:
            self._angular_fl = np.array(
                [self._fl(l, b=b)
                 for l in range(2, l_max+1, 2)]
            )[:, None]

        self.concentration = concentration
        self.r_min = r_min
        self.N_r = N_r
        self.N_jn = N_jn
        self.a1h = a1h
        self.b = b
        self.integration_method = integration_method
        self.l_max = l_max

        super().__init__(
            concentration=concentration, mass_def=mass_def,
            log10Mmin_0=log10Mmin_0, log10Mmin_p=log10Mmin_p,
            siglnM_0=siglnM_0, siglnM_p=siglnM_p,
            log10M0_0=log10M0_0, log10M0_p=log10M0_p,
            log10M1_0=log10M1_0, log10M1_p=log10M1_p,
            fc_0=0.0, fc_p=0.0,
            alpha_0=alpha_0, alpha_p=alpha_p,
            bg_0=bg_0, bg_p=bg_p,
            bmax_0=bmax_0, bmax_p=bmax_p,
            a_pivot=a_pivot, ns_independent=ns_independent)

        self.update_precision_fftlog(padding_lo_fftlog=1E-2,
                                     padding_hi_fftlog=1E3,
                                     n_per_decade=350,
                                     plaw_fourier=-3.7)

    @update(names=[
        "a1h", "b", "l_max", "r_min", "N_r", "N_jn",
        "log10Mmin_0", "log10Mmin_p", "siglnM_0", "siglnM_p", "log10M0_0",
        "log10M0_p", "log10M1_0", "log10M1_p", "alpha_0", "alpha_p", "bg_0",
        "bg_p", "bmax_0", "bmax_p", "a_pivot", "ns_independent"])
    def update_parameters(self) -> None:
        """Update the profile parameters. All numerical parameters in
        :meth:`__init__`, as well as `ns_independent`, are updatable.
        """

    def _I_integral(self, a, b):
        r"""Compute the integral

        .. math::

            I(a,b) = \int_{-1}^1 {\rm d}x (1-x^2)^{a/2} x^b =
            \frac{((-1)^b + 1) \, \Gamma(a+1) \, \Gamma\left(
            \frac{b+1}{2} \right) \,
            2 \Gamma\left( a + \frac{b}{2} + \frac{3}{2} \right).
        """
        return (1+(-1)**b)*gamma(a/2+1)*gamma((b+1)/2)/(2*gamma(a/2+b/2+3/2))

    def _fl(self, l, thk=np.pi/2, phik=None, b=-2):
        """Compute the angular part of the satellite intrinsic shear field
        (Eq. C8 of :footcite:t:`Fortuna21`).
        """
        gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32,
                       0, 7 * np.pi / 16, 0, 105 * np.pi / 256, 0,
                       99 * np.pi / 256])
        l_sum = 0.
        if b == 0:
            var1_add = 1
        else:
            var1_add = b

        for m in range(0, l+1):
            m_sum = 0
            for j in range(0, m+1):
                m_sum += (binom(m, j) * gj[j] *
                          self._I_integral(j + var1_add, m - j) *
                          np.sin(thk)**j * np.cos(thk)**(m - j))
            l_sum += binom(l, m) * binom((l + m - 1) / 2, l) * m_sum
        if phik is not None:
            l_sum *= np.exp(1j * 2 * phik)
        return 2**l * l_sum

    def get_normalization(
            self,
            cosmo: Cosmology,
            a: Real,
            hmc: HMCalculator
    ) -> float:
        """Compute the normalization of this profile, which is the mean galaxy
        number density.
        """
        def integ(M):
            Nc = self._Nc(M, a)
            Ns = self._Ns(M, a)
            if self.ns_independent:
                return Nc+Ns
            return Nc*(1+Ns)
        return hmc.integrate_over_massfunc(integ, cosmo, a)

    def gamma_I(
            self,
            r: Union[Real, NDArray[Real]],
            r_vir: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Compute the intrinsic satellite shear,

        .. math::

            \gamma^I(r) = a_{1{\rm h}}
            \left( \frac{r}{r_{\rm vir}} \right)^b.

        If :math:`b` is 0, only the value of the amplitude :math:`a_{\rm 1h}`
        is returned. Following :footcite:t:`Fortuna21`, we clip :math:`r` at
        :math:`0.06 \, \rm Mpc` and :math:`\gamma^I(r)` at :math:`0.3`.

        Arguments
        ---------
        r : array_like (nr,)
            Radius in :math:`\rm Mpc`.
        r_vir : array_like (nr,)
            Virial radius in :math:`\rm Mpc`.

        Returns
        -------
        array_like (nr,)
            Intinsic satellite shear.
        """
        if self.b == 0:
            return self.a1h

        r_use = np.copy(np.atleast_1d(r))
        r_use[r_use < 0.06] = 0.06
        if np.ndim(r_vir == 1):
            r_vir = r_vir.reshape(len(r_vir), 1)
        # Do not output value higher than 0.3
        gamma_out = self.a1h * (r_use/r_vir)**self.b
        gamma_out[gamma_out > 0.3] = 0.3
        return gamma_out

    def _real(self, cosmo, r, M, a):
        r"""Compute the real part of the satellite intrinsic shear field,

        .. math::

            \gamma^I(r) u(r|M),

        where :math:`u` is the halo density profile divided by its mass.
        Assume an NFW halo profile.
        """
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a
        # Density profile from HOD class - truncated NFW
        u = self._usat_real(cosmo, r_use, M_use, a)
        prof = self.gamma_I(r_use, rvir) * u

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _usat_fourier(self, cosmo, k, M, a):
        """Compute the Fourier transform of the satellite intrinsic shear
        field.

        The density profile of the halo is a truncated NFW profile and the
        radial integral is evaluated up to the virial radius if the integration
        method is ``'simpson'`` or ``'spline'``.
        """
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        l_arr = np.arange(2, self.l_max+1, 2, dtype=int)
        if self.integration_method in ["simpson", "spline"]:
            # Define sampling for r-integral and spherical Bessel function. The
            # Bessel function is sampled and interpolated for speed.
            r_use = np.linspace(self.r_min,
                                self.mass_def.get_radius(cosmo, M_use, a) / a,
                                self.N_r).T
            x_jn = np.geomspace(k_use.min() * r_use.min(),
                                k_use.max() * r_use.max(),
                                self.N_jn)
            jn = np.empty(shape=(len(l_arr), len(x_jn)))

        prof = np.zeros(shape=(len(M_use), len(k_use)))
        for j, l in enumerate(l_arr):  # loop over all multipoles
            prefac = (1j**l).real * (2*l + 1) * self._angular_fl[j]

            if self.integration_method == 'FFTLog':
                prof += self._fftlog_wrap(
                    cosmo, k_use, M_use, a,
                    ell=int(l), fourier_out=True) / (4 * np.pi) * prefac

            else:
                jn[j] = spherical_jn(l_arr[j], x_jn)
                k_dot_r = np.multiply.outer(k_use, r_use)
                jn_interp = np.interp(k_dot_r, x_jn, jn[j])
                integrand = (r_use**2 * jn_interp *
                             self._real(cosmo, r_use, M_use, a))

                if self.integration_method == 'simpson':
                    for i, M_i in enumerate(M_use):
                        prof[i] += simpson(integrand[:, i], r_use[i]).T

                elif self.integration_method == 'spline':
                    for i, M_i in enumerate(M_use):
                        prof[i] += _spline_integrate(
                            r_use[i], integrand[:, i],
                            r_use[i, 0], r_use[i, -1]).T

                prof[i] *= prefac

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
