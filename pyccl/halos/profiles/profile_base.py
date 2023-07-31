__all__ = ("HaloProfile", "HaloProfileMatter",
           "HaloProfilePressure", "HaloProfileCIB",)

import functools
from typing import Callable

import numpy as np

from ... import CCLAutoRepr, FFTLogParams, unlock_instance
from ... import physical_constants as const
from ...pyutils import resample_array, _fftlog_transform
from .. import MassDef


class HaloProfile(CCLAutoRepr):
    """ This class implements functionality associated to
    halo profiles. You should not use this class directly.
    Instead, use one of the subclasses implemented in CCL
    for specific halo profiles, or write your own subclass.
    ``HaloProfile`` classes contain methods to compute halo
    profiles in real (3D) and Fourier spaces, as well as other
    projected (2D) quantities.

    A minimal implementation of a ``HaloProfile`` subclass
    should contain a method ``_real`` that returns the
    real-space profile as a function of cosmology,
    comoving radius, mass and scale factor. The default
    functionality in the base ``HaloProfile`` class will then
    allow the automatic calculation of the Fourier-space
    and projected profiles, as well as the cumulative
    mass density, based on the real-space profile using
    FFTLog to carry out fast Hankel transforms. See the
    CCL note for details. Alternatively, you can implement
    a ``_fourier`` method for the Fourier-space profile, and
    all other quantities will be computed from it. It is
    also possible to implement specific versions of any
    of these quantities if one wants to avoid the FFTLog
    calculation.
    """

    def __init__(self, *, mass_def, concentration=None,
                 is_number_counts=False):
        # Verify that profile can be initialized.
        if not (hasattr(self, "_real") or hasattr(self, "_fourier")):
            name = type(self).__name__
            raise TypeError(f"Can't instantiate {name} with no "
                            "_real or _fourier implementation.")

        # Initialize FFTLog.
        self.precision_fftlog = FFTLogParams()

        self._is_number_counts = is_number_counts

        # Initialize mass_def and concentration.
        self.mass_def, *out = MassDef.from_specs(
            mass_def, concentration=concentration)
        if out:
            self.concentration = out[0]

    @property
    def is_number_counts(self):
        """If ``True``, this profile represents source-count overdensities,
        normalised by the mean number density within their survey window
        function. This must be propagated when estimated super-sample
        effects in the covariance matrix.
        """
        return self._is_number_counts

    @is_number_counts.setter
    @unlock_instance
    def is_number_counts(self, value):
        self._is_number_counts = value

    def get_normalization(self, cosmo=None, a=None, *, hmc=None):
        """Profiles may be normalized by an overall function of redshift
        (or scale factor). This function may be cosmology-dependent and
        often comes from integrating certain halo properties over mass.
        This method returns this normalizing factor. For example,
        to get the normalized profile in real space, one would call
        the :meth:`real` method, and then **divide** the result by the value
        returned by this method.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.
            hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
                model calculator object.

        Returns:
            float: normalization factor of this profile.
        """
        return 1.0

    @unlock_instance(mutate=True)
    @functools.wraps(FFTLogParams.update_parameters)
    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update_parameters(**kwargs)

    def _get_plaw_fourier(self, cosmo, a):
        """ This controls the value of `plaw_fourier` to be used
        as a function of cosmology and scale factor.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_fourier']

    def _get_plaw_projected(self, cosmo, a):
        """ This controls the value of `plaw_projected` to be
        used as a function of cosmology and scale factor.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_projected']

    _real: Callable       # implementation of the real profile

    _fourier: Callable    # implementation of the Fourier profile

    _projected: Callable  # implementation of the projected profile

    _cumul2d: Callable    # implementation of the cumulative surface density

    def real(self, cosmo, r, M, a):
        """
        real(cosmo, r, M, a)
        Returns the 3D real-space value of the profile as a
        function of cosmology, radius, halo mass and scale factor.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            r (:obj:`float` or `array`): comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): halo profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively. If `r` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, "_real", None):
            return self._real(cosmo, r, M, a)
        return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)

    def fourier(self, cosmo, k, M, a):
        """
        fourier(cosmo, k, M, a)
        Returns the Fourier-space value of the profile as a
        function of cosmology, wavenumber, halo mass and
        scale factor.

        .. math::
           \\rho(k)=\\frac{1}{2\\pi^2} \\int dr\\, r^2\\,
           \\rho(r)\\, j_0(k r)

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber (in :math:`Mpc^{-1}`).
            M (:obj:`float` or `array`): halo mass.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): Fourier-space profile. The shape of the
            output will be ``(N_M, N_k)`` where ``N_k`` and ``N_m`` are
            the sizes of ``k`` and ``M`` respectively. If ``k`` or ``M``
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """ # noqa
        if getattr(self, "_fourier", None):
            return self._fourier(cosmo, k, M, a)
        return self._fftlog_wrap(cosmo, k, M, a, fourier_out=True)

    def projected(self, cosmo, r_t, M, a):
        """
        projected(cosmo, r_t, M, a)
        Returns the 2D projected profile as a function of
        cosmology, radius, halo mass and scale factor.

        .. math::
           \\Sigma(R)= \\int dr_\\parallel\\,
           \\rho(\\sqrt{r_\\parallel^2 + R^2})

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            r_t (:obj:`float` or `array`): transverse comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): projected profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively. If `r` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, "_projected", None):
            return self._projected(cosmo, r_t, M, a)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, is_cumul2d=False)

    def cumul2d(self, cosmo, r_t, M, a):
        """
        cumul2d(cosmo, r_t, M, a)
        Returns the 2D cumulative surface density as a
        function of cosmology, radius, halo mass and scale
        factor.

        .. math::
           \\Sigma(<R)= \\frac{2}{R^2} \\int dR'\\, R'\\,
           \\Sigma(R')

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            r_t (:obj:`float` or `array`): transverse comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): cumulative surface density. The shape of the
            output will be ``(N_M, N_r)`` where ``N_r`` and ``N_m`` are
            the sizes of ``r`` and ``M`` respectively. If ``r`` or ``M``
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """ # noqa
        if getattr(self, "_cumul2d", None):
            return self._cumul2d(cosmo, r_t, M, a)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, is_cumul2d=True)

    def convergence(self, cosmo, r, M, *, a_lens, a_source):
        """
        convergence(cosmo, r, M, *, a_lens, a_source)
        Returns the convergence as a function of cosmology,
        radius, halo mass and the scale factors of the source
        and the lens.

        .. math::
           \\kappa(R) = \\frac{\\Sigma(R)}{\\Sigma_{\\rm crit}},

        where :math:`\\Sigma(R)` is the 2D projected surface mass density
        (see :meth:`projected`), and :math:`\\Sigma_{\\rm crit}` is
        the critical surface density (see
        :meth:`~pyccl.background.sigma_critical`).

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            r (:obj:`float` or `array`): comoving radius.
            M (:obj:`float` or `array`): halo mass.
            a_lens (:obj:`float`): scale factor of lens.
            a_source (:obj:`float` or `array`): scale factor of source.
                If an array, it must have the same shape as ``r``.

        Returns:
            (:obj:`float` or `array`): convergence :math:`\\kappa`. The
            shape of the output will be ``(N_M, N_r)`` where ``N_r`` and
            ``N_m`` are the sizes of ``r`` and ``M`` respectively. If
            ``r`` or ``M`` are scalars, the corresponding dimension will
            be squeezed out on output.
        """
        Sigma = self.projected(cosmo, r, M, a_lens)
        Sigma /= a_lens**2
        Sigma_crit = cosmo.sigma_critical(a_lens=a_lens, a_source=a_source)
        return Sigma / Sigma_crit

    def shear(self, cosmo, r, M, *, a_lens, a_source):
        """
        shear(cosmo, r, M, *, a_lens, a_source)
        Returns the shear (tangential) as a function of cosmology,
        radius, halo mass and the scale factors of the
        source and the lens.

        .. math::
           \\gamma(R) = \\frac{\\Delta\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}} =
           \\frac{\\overline{\\Sigma}(< R) -
           \\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},

        where :math:`\\overline{\\Sigma}(< R)` is cumulative surface density
        (see :meth:`cumul2d`), :math:`\\Sigma(R)` is the projected 2D density
        (see :meth:`projected`), and :math:`\\Sigma_{\\rm crit}` is the
        critical surface density (see
        :meth:`~pyccl.background.sigma_critical`).

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            r (:obj:`float` or `array`): comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a_lens (:obj:`float`): scale factor of lens.
            a_source (:obj:`float` or `array`): source's scale factor.
                If array, it must have the same shape as `r`.

        Returns:
            (:obj:`float` or `array`): shear :math:`\\gamma`. The shape of the
            output will be ``(N_M, N_r)`` where ``N_r`` and ``N_m`` are
            the sizes of ``r`` and ``M`` respectively. If ``r`` or ``M``
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        Sigma = self.projected(cosmo, r, M, a_lens)
        Sigma_bar = self.cumul2d(cosmo, r, M, a_lens)
        Sigma_crit = cosmo.sigma_critical(a_lens=a_lens, a_source=a_source)
        return (Sigma_bar - Sigma) / (Sigma_crit * a_lens**2)

    def reduced_shear(self, cosmo, r, M, *, a_lens, a_source):
        """
        reduced_shear(cosmo, r, M, *, a_lens, a_source)
        Returns the reduced shear as a function of cosmology,
        radius, halo mass and the scale factors of the
        source and the lens.

        .. math::
           g_t (R) = \\frac{\\gamma(R)}{(1 - \\kappa(R))},

        where :math:`\\gamma(R)` is the shear and :math:`\\kappa(R)` is the
        convergence.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            r (:obj:`float` or `array`): comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a_lens (:obj:`float`): scale factor of lens.
            a_source (:obj:`float` or `array`): source's scale factor.
                If array, it must have the same shape as `r`.

        Returns:
            (:obj:`float` or `array`): reduced shear :math:`g_t`. The shape
            of the output will be ``(N_M, N_r)`` where ``N_r`` and ``N_m``
            are the sizes of ``r`` and ``M`` respectively. If ``r`` or
            ``M`` are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        convergence = self.convergence(cosmo, r, M, a_lens=a_lens,
                                       a_source=a_source)
        shear = self.shear(cosmo, r, M, a_lens=a_lens, a_source=a_source)
        return shear / (1.0 - convergence)

    def magnification(self, cosmo, r, M, *, a_lens, a_source):
        """
        magnification(cosmo, r, M, a_lens, a_source)
        Returns the magnification for input parameters.

        .. math::
           \\mu(R) = \\frac{1}{\\left[(1 - \\kappa(R))^2 -
           \\vert \\gamma(R) \\vert^2 \\right]]},

        where :math:`\\gamma(R)` is the shear and :math:`\\kappa(R)` is the
        convergence (see :meth:`shear` and :meth:`convergence`).

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            r (:obj:`float` or `array`): comoving radius in Mpc.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a_lens (:obj:`float`): scale factor of lens.
            a_source (:obj:`float` or `array`): source's scale factor.
                If array, it must have the same shape as `r`.

        Returns:
            (:obj:`float` or `array`): magnification :math:`\\mu`. The shape
            of the output will be ``(N_M, N_r)`` where ``N_r`` and ``N_m``
            are the sizes of ``r`` and ``M`` respectively. If ``r`` or ``M``
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        convergence = self.convergence(cosmo, r, M, a_lens=a_lens,
                                       a_source=a_source)
        shear = self.shear(cosmo, r, M, a_lens=a_lens, a_source=a_source)

        return 1.0 / ((1.0 - convergence)**2 - np.abs(shear)**2)

    def _fftlog_wrap(self, cosmo, k, M, a,
                     fourier_out=False,
                     large_padding=True, ell=0):
        # This computes the 3D Hankel transform
        #  \rho(k) = 4\pi \int dr r^2 \rho(r) j_ell(k r)
        # if fourier_out == True, and
        #  \rho(r) = \frac{1}{2\pi^2} \int dk k^2 \rho(k) j_ell(k r)
        # otherwise.

        # Select which profile should be the input
        if fourier_out:
            p_func = self._real
        else:
            p_func = self._fourier
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)
        nM = len(M_use)

        # k/r ranges to be used with FFTLog and its sampling.
        if large_padding:
            k_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(k_use)
        else:
            k_min = self.precision_fftlog['padding_lo_extra'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_extra'] * np.amax(k_use)
        n_k = (int(np.log10(k_max / k_min)) *
               self.precision_fftlog['n_per_decade'])
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([nM, k_use.size])
        # Compute real profile values
        p_real_M = p_func(cosmo, r_arr, M_use, a)
        # Power-law index to pass to FFTLog.
        plaw_index = self._get_plaw_fourier(cosmo, a)

        # Compute Fourier profile through fftlog
        k_arr, p_fourier_M = _fftlog_transform(r_arr, p_real_M,
                                               3, ell, plaw_index)
        lk_arr = np.log(k_arr)

        for im, p_k_arr in enumerate(p_fourier_M):
            # Resample into input k values
            p_fourier = resample_array(lk_arr, p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier
        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, is_cumul2d=False):
        # This computes Sigma(R) from the Fourier-space profile as:
        # Sigma(R) = \frac{1}{2\pi} \int dk k J_0(k R) \rho(k)
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        nM = len(M_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([nM, r_t_use.size])
        # Compute Fourier-space profile
        if getattr(self, "_fourier", None):
            # Compute from `_fourier` if available.
            p_fourier = self._fourier(cosmo, k_arr, M_use, a)
        else:
            # Compute with FFTLog otherwise.
            lpad = self.precision_fftlog['large_padding_2D']
            p_fourier = self._fftlog_wrap(cosmo, k_arr, M_use, a,
                                          fourier_out=True, large_padding=lpad)
        if is_cumul2d:
            # The cumulative profile involves a factor 1/(k R) in
            # the integrand.
            p_fourier *= 2 / k_arr[None, :]

        # Power-law index to pass to FFTLog.
        if is_cumul2d:
            i_bessel = 1
            plaw_index = self._get_plaw_projected(cosmo, a) - 1
        else:
            i_bessel = 0
            plaw_index = self._get_plaw_projected(cosmo, a)

        # Compute projected profile through fftlog
        r_t_arr, sig_r_t_M = _fftlog_transform(k_arr, p_fourier,
                                               2, i_bessel,
                                               plaw_index)
        lr_t_arr = np.log(r_t_arr)

        if is_cumul2d:
            sig_r_t_M /= r_t_arr[None, :]
        for im, sig_r_t_arr in enumerate(sig_r_t_M):
            # Resample into input r_t values
            sig_r_t = resample_array(lr_t_arr, sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t

        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out


class HaloProfileMatter(HaloProfile):
    """Base for matter halo profiles."""

    def get_normalization(self, cosmo, a, *, hmc=None):
        """Returns the normalization of all matter overdensity
        profiles, which is simply the comoving matter density.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.
            hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
                model calculator object.

        Returns:
            :obj:`float`: normalization factor of this profile.
        """
        return const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2


class HaloProfilePressure(HaloProfile):
    """Base for pressure halo profiles."""


class HaloProfileCIB(HaloProfile):
    """Base for CIB halo profiles."""
