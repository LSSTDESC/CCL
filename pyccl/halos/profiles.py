import warnings
import pyccl
from ..background import h_over_h0, sigma_critical
from ..power import sigmaM
from ..pyutils import resample_array, _fftlog_transform
from .concentration import Concentration
from .massdef import MassDef
import numpy as np
from scipy.special import sici, erf, gamma, gammainc


class HaloProfile(object):
    """ This class implements functionality associated to
    halo profiles. You should not use this class directly.
    Instead, use one of the subclasses implemented in CCL
    for specific halo profiles, or write your own subclass.
    `HaloProfile` classes contain methods to compute halo
    profiles in real (3D) and Fourier spaces, as well as the
    projected (2D) profile and the cumulative mean surface
    density.

    A minimal implementation of a `HaloProfile` subclass
    should contain a method `_real` that returns the
    real-space profile as a function of cosmology,
    comoving radius, mass and scale factor. The default
    functionality in the base `HaloProfile` class will then
    allow the automatic calculation of the Fourier-space
    and projected profiles, as well as the cumulative
    mass density, based on the real-space profile using
    FFTLog to carry out fast Hankel transforms. See the
    CCL note for details. Alternatively, you can implement
    a `_fourier` method for the Fourier-space profile, and
    all other quantities will be computed from it. It is
    also possible to implement specific versions of any
    of these quantities if one wants to avoid the FFTLog
    calculation.
    """
    name = 'default'
    is_number_counts = False

    def __init__(self):
        self.precision_fftlog = {'padding_lo_fftlog': 0.1,
                                 'padding_lo_extra': 0.1,
                                 'padding_hi_fftlog': 10.,
                                 'padding_hi_extra': 10.,
                                 'large_padding_2D': False,
                                 'n_per_decade': 100,
                                 'extrapol': 'linx_liny',
                                 'plaw_fourier': -1.5,
                                 'plaw_projected': -1.}

    def update_precision_fftlog(self, **kwargs):
        """ Update any of the precision parameters used by
        FFTLog to compute Hankel transforms. The available
        parameters are:

        Args:
            padding_lo_fftlog (float): when computing a Hankel
                transform we often need to extend the range of the
                input (e.g. the r-range for the real-space profile
                when computing the Fourier-space one) to avoid
                aliasing and boundary effects. This parameter
                controls the factor by which we multiply the lower
                end of the range (e.g. a value of 0.1 implies that
                we will extend the range by one decade on the
                left). Note that FFTLog works in logarithmic
                space. Default value: 0.1.
            padding_hi_fftlog (float): same as `padding_lo_fftlog`
                for the upper end of the range (e.g. a value of
                10 implies extending the range by one decade on
                the right). Default value: 10.
            n_per_decade (float): number of samples of the
                profile taken per decade when computing Hankel
                transforms.
            padding_lo_extra (float): when computing the projected
                2D profile or the 2D cumulative density,
                sometimes two Hankel transforms are needed (from
                3D real-space to Fourier, then from Fourier to
                2D real-space). This parameter controls the k range
                of the intermediate transform. The logic here is to
                avoid the range twice by `padding_lo_fftlog` (which
                can be overkill and slow down the calculation).
                Default value: 0.1.
            padding_hi_extra (float): same as `padding_lo_extra`
                for the upper end of the range. Default value: 10.
                large_padding_2D (bool): if set to `True`, the
                intermediate Hankel transform in the calculation of
                the 2D projected profile and cumulative mass
                density will use `padding_lo_fftlog` and
                `padding_hi_fftlog` instead of `padding_lo_extra`
                and `padding_hi_extra` to extend the range of the
                intermediate Hankel transform.
            extrapol (string): type of extrapolation used in the
                uncommon scenario that FFTLog returns a profile on a
                range that does not cover the intended output range.
                Pass `linx_liny` if you want to extrapolate linearly
                in the profile and `linx_logy` if you want to
                extrapolate linearly in its logarithm.
                Default value: `linx_liny`.
            plaw_fourier (float): FFTLog is able to perform more
                accurate Hankel transforms by prewhitening its arguments
                (essentially making them flatter over the range of
                integration to avoid aliasing). This parameter
                corresponds to a guess of what the tilt of the profile
                is (i.e. profile(r) = r^tilt), which FFTLog uses to
                prewhiten it. This parameter is used when computing the
                real <-> Fourier transforms. The methods
                `_get_plaw_fourier` allows finer control over this
                parameter. The default value allows for a slightly faster
                (but potentially less accurate) FFTLog transform. Some
                level of experimentation with this parameter is
                recommended when implementing a new profile.
                Default value: -1.5.
            plaw_projected (float): same as `plaw_fourier` for the
                calculation of the 2D projected and cumulative density
                profiles. Finer control can be achieved with the
                `_get_plaw_projected`. The default value allows for a
                slightly faster (but potentially less accurate) FFTLog
                transform.  Some level of experimentation with this
                parameter is recommended when implementing a new profile.
                Default value: -1.
        """
        self.precision_fftlog.update(kwargs)

    def _get_plaw_fourier(self, cosmo, a):
        """ This controls the value of `plaw_fourier` to be used
        as a function of cosmology and scale factor.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            a (float): scale factor.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_fourier']

    def _get_plaw_projected(self, cosmo, a):
        """ This controls the value of `plaw_projected` to be
        used as a function of cosmology and scale factor.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            a (float): scale factor.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_projected']

    def _get_prefactor(self, cosmo, a, hmc):
        """
        Add a pre-factor for the total power spectra that depends
        only on the scale factor.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            a (float): scale factor.
            hmc (:class:`~pyccl.halos.HMCalculator`): halo
                model calculator object.

        Returns:
            float or array_like: pre-factor value.
        """
        return 1.

    def real(self, cosmo, r, M, a, mass_def):
        """ Returns the 3D  real-space value of the profile as a
        function of cosmology, radius, halo mass and scale factor.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively. If `r` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, '_real', None):
            f_r = self._real(cosmo, r, M, a, mass_def)
        elif getattr(self, '_fourier', None):
            f_r = self._fftlog_wrap(cosmo, r, M, a, mass_def,
                                    fourier_out=False)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _real or a "
                                      " _fourier method.")
        return f_r

    def fourier(self, cosmo, k, M, a, mass_def):
        """ Returns the Fourier-space value of the profile as a
        function of cosmology, wavenumber, halo mass and
        scale factor.

        .. math::
           \\rho(k)=\\frac{1}{2\\pi^2} \\int dr\\, r^2\\,
           \\rho(r)\\, j_0(k r)

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_M, N_k)` where `N_k` and `N_m` are
            the sizes of `k` and `M` respectively. If `k` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, '_fourier', None):
            f_k = self._fourier(cosmo, k, M, a, mass_def)
        elif getattr(self, '_real', None):
            f_k = self._fftlog_wrap(cosmo, k, M, a, mass_def,
                                    fourier_out=True)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _real or a "
                                      " _fourier method.")
        return f_k

    def projected(self, cosmo, r_t, M, a, mass_def):
        """ Returns the 2D projected profile as a function of
        cosmology, radius, halo mass and scale factor.

        .. math::
           \\Sigma(R)= \\int dr_\\parallel\\,
           \\rho(\\sqrt{r_\\parallel^2 + R^2})

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively. If `r` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, '_projected', None):
            s_r_t = self._projected(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._projected_fftlog_wrap(cosmo, r_t, M,
                                                a, mass_def,
                                                is_cumul2d=False)
        return s_r_t

    def cumul2d(self, cosmo, r_t, M, a, mass_def):
        """ Returns the 2D cumulative surface density as a
        function of cosmology, radius, halo mass and scale
        factor.

        .. math::
           \\Sigma(<R)= \\frac{2}{R^2} \\int dR'\\, R'\\,
           \\Sigma(R')

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively. If `r` or `M`
            are scalars, the corresponding dimension will be
            squeezed out on output.
        """
        if getattr(self, '_cumul2d', None):
            s_r_t = self._cumul2d(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._projected_fftlog_wrap(cosmo, r_t, M,
                                                a, mass_def,
                                                is_cumul2d=True)
        return s_r_t

    def convergence(self, cosmo, r, M, a_lens, a_source, mass_def):
        """ Returns the convergence as a function of cosmology,
        radius, halo mass and the scale factors of the source
        and the lens.

        .. math::
           \\kappa(R) = \\frac{\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},\\

        where :math:`\\Sigma(R)` is the 2D projected surface mass density.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a_lens (float): scale factor of lens.
            a_source (float or array_like): scale factor of source.
                If array_like, it must have the same shape as `r`.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: convergence \
                :math:`\\kappa`
        """
        Sigma = self.projected(cosmo, r, M, a_lens, mass_def) / a_lens**2
        Sigma_crit = sigma_critical(cosmo, a_lens, a_source)
        return Sigma / Sigma_crit

    def shear(self, cosmo, r, M, a_lens, a_source, mass_def):
        """ Returns the shear (tangential) as a function of cosmology,
        radius, halo mass and the scale factors of the
        source and the lens.

        .. math::
           \\gamma(R) = \\frac{\\Delta\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}} =
           \\frac{\\overline{\\Sigma}(< R) -
           \\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},\\

        where :math:`\\overline{\\Sigma}(< R)` is the average surface density
        within R.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a_lens (float): scale factor of lens.
            a_source (float or array_like): source's scale factor.
                If array_like, it must have the same shape as `r`.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: shear \
                :math:`\\gamma`
        """
        Sigma = self.projected(cosmo, r, M, a_lens, mass_def)
        Sigma_bar = self.cumul2d(cosmo, r, M, a_lens, mass_def)
        Sigma_crit = sigma_critical(cosmo, a_lens, a_source)
        return (Sigma_bar - Sigma) / (Sigma_crit * a_lens**2)

    def reduced_shear(self, cosmo, r, M, a_lens, a_source, mass_def):
        """ Returns the reduced shear as a function of cosmology,
        radius, halo mass and the scale factors of the
        source and the lens.

        .. math::
           g_t (R) = \\frac{\\gamma(R)}{(1 - \\kappa(R))},\\

        where :math: `\\gamma(R)` is the shear and `\\kappa(R)` is the
                convergence.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a_lens (float): scale factor of lens.
            a_source (float or array_like): source's scale factor.
                If array_like, it must have the same shape as `r`.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: reduced shear \
                :math:`g_t`
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source, mass_def)
        shear = self.shear(cosmo, r, M, a_lens, a_source, mass_def)
        return shear / (1.0 - convergence)

    def magnification(self, cosmo, r, M, a_lens, a_source, mass_def):
        """ Returns the magnification for input parameters.

        .. math::
           \\mu (R) = \\frac{1}{\\left[(1 - \\kappa(R))^2 -
           \\vert \\gamma(R) \\vert^2 \\right]]},\\

        where :math: `\\gamma(R)` is the shear and `\\kappa(R)` is the
        convergence.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a_lens (float): scale factor of lens.
            a_source (float or array_like): source's scale factor.
                If array_like, it must have the same shape as `r`.
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: magnification\
                :math:`\\mu`
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source, mass_def)
        shear = self.shear(cosmo, r, M, a_lens, a_source, mass_def)

        return 1.0 / ((1.0 - convergence)**2 - np.abs(shear)**2)

    def _fftlog_wrap(self, cosmo, k, M, a, mass_def,
                     l=0,
                     fourier_out=False,
                     large_padding=True):
        # This computes the 3D Hankel transform
        #  \rho(k) = 4\pi \int dr r^2 \rho(r) j_l(k r)
        # if fourier_out == True, and
        #  \rho(r) = \frac{1}{2\pi^2} \int dk k^2 \rho(k) j_l(k r)
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
        p_real_M = p_func(cosmo, r_arr, M_use, a, mass_def)
        # Power-law index to pass to FFTLog.
        plaw_index = self._get_plaw_fourier(cosmo, a)

        # Compute Fourier profile through fftlog
        k_arr, p_fourier_M = _fftlog_transform(r_arr, p_real_M,
                                               3, l, plaw_index)
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

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, mass_def,
                               is_cumul2d=False):
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
        if getattr(self, '_fourier', None):
            # Compute from `_fourier` if available.
            p_fourier = self._fourier(cosmo, k_arr, M_use,
                                      a, mass_def)
        else:
            # Compute with FFTLog otherwise.
            lpad = self.precision_fftlog['large_padding_2D']
            p_fourier = self._fftlog_wrap(cosmo,
                                          k_arr,
                                          M_use, a,
                                          mass_def,
                                          fourier_out=True,
                                          large_padding=lpad)
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


class HaloProfileGaussian(HaloProfile):
    """ Gaussian profile

    .. math::
        \\rho(r) = \\rho_0\\, e^{-(r/r_s)^2}

    Args:
        r_scale (:obj:`function`): the width of the profile.
            The signature of this function should be
            `f(cosmo, M, a, mdef)`, where `cosmo` is a
            :class:`~pyccl.core.Cosmology` object, `M` is a halo mass in
            units of M_sun, `a` is the scale factor and `mdef`
            is a :class:`~pyccl.halos.massdef.MassDef` object.
        rho0 (:obj:`function`): the amplitude of the profile.
            It should have the same signature as `r_scale`.
    """
    name = 'Gaussian'

    def __init__(self, r_scale, rho0):
        self.rho_0 = rho0
        self.r_s = r_scale
        super(HaloProfileGaussian, self).__init__()
        self.update_precision_fftlog(padding_lo_fftlog=0.01,
                                     padding_hi_fftlog=100.,
                                     n_per_decade=10000)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_s(cosmo, M_use, a, mass_def)
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


class HaloProfilePowerLaw(HaloProfile):
    """ Power-law profile

    .. math::
        \\rho(r) = (r/r_s)^\\alpha

    Args:
        r_scale (:obj:`function`): the correlation length of
            the profile. The signature of this function
            should be `f(cosmo, M, a, mdef)`, where `cosmo`
            is a :class:`~pyccl.core.Cosmology` object, `M` is a halo mass
            in units of M_sun, `a` is the scale factor and
            `mdef` is a :class:`~pyccl.halos.massdef.MassDef` object.
        tilt (:obj:`function`): the power law index of the
            profile. The signature of this function should
            be `f(cosmo, a)`.
    """
    name = 'PowerLaw'

    def __init__(self, r_scale, tilt):
        self.r_s = r_scale
        self.tilt = tilt
        super(HaloProfilePowerLaw, self).__init__()

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
        rs = self.r_s(cosmo, M_use, a, mass_def)
        tilt = self.tilt(cosmo, a)
        # Form factor
        prof = (r_use[None, :] / rs[:, None])**tilt

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileNFW(HaloProfile):
    """ Navarro-Frenk-White (astro-ph:astro-ph/9508025) profile.

    .. math::
       \\rho(r) = \\frac{\\rho_0}
       {\\frac{r}{r_s}\\left(1+\\frac{r}{r_s}\\right)^2}

    where :math:`r_s` is related to the spherical overdensity
    halo radius :math:`R_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       R_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is

    .. math::
       \\rho_0 = \\frac{M}{4\\pi\\,r_s^3\\,[\\log(1+c) - c/(1+c)]}

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Args:
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        fourier_analytic (bool): set to `True` if you want to compute
            the Fourier profile analytically (and not through FFTLog).
            Default: `True`.
        projected_analytic (bool): set to `True` if you want to
            compute the 2D projected profile analytically (and not
            through FFTLog). Default: `False`.
        cumul2d_analytic (bool): set to `True` if you want to
            compute the 2D cumulative surface density analytically
            (and not through FFTLog). Default: `False`.
        truncated (bool): set to `True` if the profile should be
            truncated at :math:`r = R_\\Delta` (i.e. zero at larger
            radii.
    """
    name = 'NFW'

    def __init__(self, c_M_relation,
                 fourier_analytic=True,
                 projected_analytic=False,
                 cumul2d_analytic=False,
                 truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        if fourier_analytic:
            self._fourier = self._fourier_analytic
        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported "
                                 "for truncated NFW. Set `truncated` or "
                                 "`projected_analytic` to `False`.")
            self._projected = self._projected_analytic
        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumuative 2d profile not supported "
                                 "for truncated NFW. Set `truncated` or "
                                 "`cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic
        super(HaloProfileNFW, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _norm(self, M, Rs, c):
        # NFW normalization from mass, radius and concentration
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = 1./(x * (1 + x)**2)
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        norm = self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_projected(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return 1 / x2m1 + np.arccosh(1 / xx) / np.fabs(x2m1)**1.5

        def f2(xx):
            x2m1 = xx * xx - 1
            return 1 / x2m1 - np.arccos(1 / xx) / np.fabs(x2m1)**1.5

        xf = x.flatten()
        return np.piecewise(xf,
                            [xf < 1, xf > 1],
                            [f1, f2, 1./3.]).reshape(x.shape)

    def _projected_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_cumul2d(self, x):

        def f1(xx):
            sqx2m1 = np.sqrt(np.fabs(xx * xx - 1))
            return np.log(0.5 * xx) + np.arccosh(1 / xx) / sqx2m1

        def f2(xx):
            sqx2m1 = np.sqrt(np.fabs(xx * xx - 1))
            return np.log(0.5 * xx) + np.arccos(1 / xx) / sqx2m1

        xf = x.flatten()
        omln2 = 0.3068528194400547  # 1-Log[2]
        f = np.piecewise(xf,
                         [xf < 1, xf > 1],
                         [f1, f2, omln2]).reshape(x.shape)
        return 2 * f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        P1 = M_use / (np.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[:, None]) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
            prof = P1[:, None] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[:, None] * P2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileEinasto(HaloProfile):
    """ Einasto profile (1965TrAlm...5...87E).

    .. math::
       \\rho(r) = \\rho_0\\,\\exp(-2 ((r/r_s)^\\alpha-1) / \\alpha)

    where :math:`r_s` is related to the spherical overdensity
    halo radius :math:`R_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       R_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is the mean density
    within the :math:`R_\\Delta(M)` of the halo. The index
    :math:`\\alpha` depends on halo mass and redshift, and we
    use the parameterization of Diemer & Kravtsov
    (arXiv:1401.1216).

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Args:
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        truncated (bool): set to `True` if the profile should be
            truncated at :math:`r = R_\\Delta` (i.e. zero at larger
            radii.
        alpha (float, 'cosmo'): Set the Einasto alpha parameter or set to
            'cosmo' to calculate the value from cosmology. Default: 'cosmo'
    """
    name = 'Einasto'

    def __init__(self, c_M_relation, truncated=True, alpha='cosmo'):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        self.alpha = alpha
        super(HaloProfileEinasto, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def update_parameters(self, alpha=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.

        Arguments
        ---------
        alpha : float, 'cosmo'
            Profile shape parameter. Set to
            'cosmo' to calculate the value from cosmology
        """
        if alpha is not None and alpha != self.alpha:
            self.alpha = alpha

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _get_alpha(self, cosmo, M, a, mdef):
        if self.alpha == 'cosmo':
            mdef_vir = MassDef('vir', 'matter')
            Mvir = mdef.translate_mass(cosmo, M, a, mdef_vir)
            sM = sigmaM(cosmo, Mvir, a)
            nu = 1.686 / sM
            alpha = 0.155 + 0.0095 * nu * nu
        else:
            alpha = np.full_like(M, self.alpha)
        return alpha

    def _norm(self, M, Rs, c, alpha):
        # Einasto normalization from mass, radius, concentration and alpha
        return M / (np.pi * Rs**3 * 2**(2-3/alpha) * alpha**(-1+3/alpha)
                    * np.exp(2/alpha)
                    * gamma(3/alpha) * gammainc(3/alpha, 2/alpha*c**alpha))

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, M_use, a, mass_def)

        norm = self._norm(M_use, R_s, c_M, alpha)

        x = r_use[None, :] / R_s[:, None]
        prof = norm[:, None] * np.exp(-2. * (x**alpha[:, None] - 1) /
                                      alpha[:, None])
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileHernquist(HaloProfile):
    """ Hernquist (1990ApJ...356..359H).

    .. math::
       \\rho(r) = \\frac{\\rho_0}
       {\\frac{r}{r_s}\\left(1+\\frac{r}{r_s}\\right)^3}

    where :math:`r_s` is related to the spherical overdensity
    halo radius :math:`R_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       R_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is the mean density
    within the :math:`R_\\Delta(M)` of the halo.

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Args:
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        fourier_analytic (bool): set to `True` if you want to compute
            the Fourier profile analytically (and not through FFTLog).
            Default: `False`.
        projected_analytic (bool): set to `True` if you want to
            compute the 2D projected profile analytically (and not
            through FFTLog). Default: `False`.
        cumul2d_analytic (bool): set to `True` if you want to
            compute the 2D cumulative surface density analytically
            (and not through FFTLog). Default: `False`.
        truncated (bool): set to `True` if the profile should be
            truncated at :math:`r = R_\\Delta` (i.e. zero at larger
            radii.
    """
    name = 'Hernquist'

    def __init__(self, c_M_relation,
                 truncated=True,
                 fourier_analytic=False,
                 projected_analytic=False,
                 cumul2d_analytic=False):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        if fourier_analytic:
            self._fourier = self._fourier_analytic
        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported "
                                 "for truncated Hernquist. Set `truncated` or "
                                 "`projected_analytic` to `False`.")
            self._projected = self._projected_analytic
        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumuative 2d profile not supported "
                                 "for truncated Hernquist. Set `truncated` or "
                                 "`cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic
        super(HaloProfileHernquist, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-4,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _norm(self, M, Rs, c):
        # Hernquist normalization from mass, radius and concentration
        return M / (2 * np.pi * Rs**3 * (c / (1 + c))**2)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        norm = self._norm(M_use, R_s, c_M)

        x = r_use[None, :] / R_s[:, None]
        prof = norm[:, None] / (x * (1 + x)**3)
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_projected(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return (-3 / 2 / x2m1**2
                    + (x2m1+3) * np.arccosh(1 / xx) / 2 / np.fabs(x2m1)**2.5)

        def f2(xx):
            x2m1 = xx * xx - 1
            return (-3 / 2 / x2m1**2
                    + (x2m1+3) * np.arccos(1 / xx) / 2 / np.fabs(x2m1)**2.5)

        xf = x.flatten()
        return np.piecewise(xf,
                            [xf < 1, xf > 1],
                            [f1, f2, 2./15.]).reshape(x.shape)

    def _projected_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_cumul2d(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return (1 + 1 / x2m1
                    + (x2m1 + 1) * np.arccosh(1 / xx) / np.fabs(x2m1)**1.5)

        def f2(xx):
            x2m1 = xx * xx - 1
            return (1 + 1 / x2m1
                    - (x2m1 + 1) * np.arccos(1 / xx) / np.fabs(x2m1)**1.5)

        xf = x.flatten()
        f = np.piecewise(xf,
                         [xf < 1, xf > 1],
                         [f1, f2, 1./3.]).reshape(x.shape)

        return f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        P1 = M / ((c_M / (c_M + 1))**2 / 2)
        c_Mp1 = c_M[:, None] + 1
        if self.truncated:
            Si1, Ci1 = sici(c_Mp1 * x)
            P2 = x * np.sin(x) * (Ci1 - Ci2) - x * np.cos(x) * (Si1 - Si2)
            P3 = (-1 + np.sin(c_M[:, None] * x) / (c_Mp1**2 * x)
                  + c_Mp1 * np.cos(c_M[:, None] * x) / (c_Mp1**2))
            prof = P1[:, None] * (P2 - P3) / 2
        else:
            P2 = (-x * (2 * np.sin(x) * Ci2 + np.pi * np.cos(x))
                  + 2 * x * np.cos(x) * Si2 + 2) / 4
            prof = P1[:, None] * P2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfilePressureGNFW(HaloProfile):
    """ Generalized NFW pressure profile by Arnaud et al.
    (2010A&A...517A..92A).

    The parametrization is:

    .. math::

       P_e(r) = C\\times P_0 h_{70}^E (c_{500} x)^{-\\gamma}
       [1+(c_{500}x)^\\alpha]^{(\\gamma-\\beta)/\\alpha},

    where

    .. math::

       C = 1.65\\,h_{70}^2\\left(\\frac{H(z)}{H_0}\\right)^{8/3}
       \\left[\\frac{h_{70}\\tilde{M}_{500}}
       {3\\times10^{14}\\,M_\\odot}\\right]^{2/3+\\alpha_{\\mathrm{P}}},

    :math:`x = r/\\tilde{r}_{500}`, :math:`h_{70}=h/0.7`, and the
    exponent :math:`E` is -1 for SZ-based profile normalizations
    and -1.5 for X-ray-based normalizations. The biased mass
    :math:`\\tilde{M}_{500}` is related to the true overdensity
    mass :math:`M_{500}` via the mass bias parameter :math:`(1-b)`
    as :math:`\\tilde{M}_{500}=(1-b)M_{500}`. :math:`\\tilde{r}_{500}`
    is the overdensity halo radius associated with :math:`\\tilde{M}_{500}`
    (note the intentional tilde!), and the profile is defined for
    a halo overdensity :math:`\\Delta=500` with respect to the
    critical density.

    The default arguments (other than ``mass_bias``), correspond to the
    profile parameters used in the Planck 2013 (V) paper. The profile is
    calculated in physical (non-comoving) units of :math:`\\mathrm{eV/cm^3}`.

    Parameters
    ----------
    mass_bias : float
        The mass bias parameter :math:`1-b`.
    P0 : float
        Profile normalization.
    c500 : float
        Concentration parameter.
    alpha, beta, gamma : float
        Profile shape parameters.
    alpha_P : float
        Additional mass dependence exponent
    P0_hexp : float
        Power of :math:`h` with which the normalization parameter scales.
        Equal to :math:`-1` for SZ-based normalizations,
        and :math:`-3/2` for X-ray-based normalizations.
    qrange : 2-sequence
        Limits of integration used when computing the Fourier-space
        profile template, in units of :math:`R_{\\mathrm{vir}}`.
    nq : int
        Number of sampling points of the Fourier-space profile template.
    x_out : float
        Profile threshold, in units of :math:`R_{\\mathrm{500c}}`.
        Defaults to :math:`+\\infty`.
    """
    name = 'GNFW'

    def __init__(self, mass_bias=0.8, P0=6.41,
                 c500=1.81, alpha=1.33, alpha_P=0.12,
                 beta=4.13, gamma=0.31, P0_hexp=-1.,
                 qrange=(1e-3, 1e3), nq=128, x_out=np.inf):
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
        super(HaloProfilePressureGNFW, self).__init__()

    def update_parameters(self, mass_bias=None, P0=None,
                          c500=None, alpha=None, beta=None, gamma=None,
                          alpha_P=None, P0_hexp=None, x_out=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.

        .. note::

            A change in ``alpha``, ``beta``, ``gamma``, ``c500``, or ``x_out``
            recomputes the Fourier-space template, which may be slow.

        Arguments
        ---------
        mass_bias : float
            The mass bias parameter :math:`1-b`.
        P0 : float
            Profile normalization.
        c500 : float
            Concentration parameter.
        alpha, beta, gamma : float
            Profile shape parameter.
        alpha_P : float
            Additional mass-dependence exponent.
        P0_hexp : float
            Power of ``h`` with which the normalization scales.
            SZ-based normalizations: -1. X-ray-based normalizations: -3/2.
        x_out : float
            Profile threshold (as a fraction of r500c).
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
        # Computes the normalisation factor of the GNFW profile.
        # Normalisation factor is given in units of eV/cm^3.
        # (Bolliet et al. 2017).
        h70 = cosmo["h"]/0.7
        C0 = 1.65*h70**2
        CM = (h70*M*mb/3E14)**(2/3+self.alpha_P)   # M dependence
        Cz = h_over_h0(cosmo, a)**(8/3)  # z dependence
        P0_corr = self.P0 * h70**self.P0_hexp  # h-corrected P_0
        return P0_corr * C0 * CM * Cz

    def _real(self, cosmo, r, M, a, mass_def):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        # (1-b)
        mb = self.mass_bias
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * mb, a) / a

        nn = self._norm(cosmo, M_use, a, mb)
        prof = self._form_factor(r_use[None, :] / R[:, None])
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        # Fourier-space profile.
        # Output in units of eV * Mpc^3 / cm^3.

        # Tabulate if not done yet
        if self._fourier_interp is None:
            self._fourier_interp = self._integ_interp()

        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # hydrostatic bias
        mb = self.mass_bias
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * mb, a) / a

        ff = self._fourier_interp(np.log(k_use[None, :] * R[:, None]))
        nn = self._norm(cosmo, M_use, a, mb)

        prof = (4*np.pi*R**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileHOD(HaloProfile):
    """ A generic halo occupation distribution (HOD)
    profile describing the number density of galaxies
    as a function of halo mass.

    The parametrization for the mean profile is:

    .. math::
       \\langle n_g(r)|M,a\\rangle = \\bar{N}_c(M,a)
       \\left[f_c(a)+\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)\\right]

    where :math:`\\bar{N}_c` and :math:`\\bar{N}_s` are the
    mean number of central and satellite galaxies respectively,
    :math:`f_c` is the observed fraction of central galaxies, and
    :math:`u_{\\rm sat}(r|M,a)` is the distribution of satellites
    as a function of distance to the halo centre.

    These quantities are parametrized as follows:

    .. math::
       \\bar{N}_c(M,a)=\\frac{1}{2}\\left[1+{\\rm erf}
       \\left(\\frac{\\log(M/M_{\\rm min})}{\\sigma_{{\\rm ln}M}}
       \\right)\\right]

    .. math::
       \\bar{N}_s(M,a)=\\Theta(M-M_0)\\left(\\frac{M-M_0}{M_1}
       \\right)^\\alpha

    .. math::
       u_s(r|M,a)\\propto\\frac{\\Theta(r_{\\rm max}-r)}
       {(r/r_g)(1+r/r_g)^2}

    Where :math:`\\Theta(x)` is the Heaviside step function,
    and the proportionality constant in the last equation is
    such that the volume integral of :math:`u_s` is 1. The
    radius :math:`r_g` is related to the NFW scale radius :math:`r_s`
    through :math:`r_g=\\beta_g\\,r_s`, and the radius
    :math:`r_{\\rm max}` is related to the overdensity radius
    :math:`r_\\Delta` as :math:`r_{\\rm max}=\\beta_{\\rm max}r_\\Delta`.
    The scale radius is related to the comoving overdensity halo
    radius via :math:`R_\\Delta(M) = c(M)\\,r_s`.

    All the quantities :math:`\\log_{10}M_{\\rm min}`,
    :math:`\\log_{10}M_0`, :math:`\\log_{10}M_1`,
    :math:`\\sigma_{{\\rm ln}M}`, :math:`f_c`, :math:`\\alpha`,
    :math:`\\beta_g` and :math:`\\beta_{\\rm max}` are
    time-dependent via a linear expansion around a pivot scale
    factor :math:`a_*` with an offset (:math:`X_0`) and a tilt
    parameter (:math:`X_p`):

    .. math::
       X(a) = X_0 + X_p\\,(a-a_*).

    This definition of the HOD profile draws from several papers
    in the literature, including: astro-ph/0408564, arXiv:1706.05422
    and arXiv:1912.08209. The default values used here are roughly
    compatible with those found in the latter paper.

    See :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`) for a
    description of the Fourier-space two-point correlator of the
    HOD profile.

    Args:
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        lMmin_0 (float): offset parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        lMmin_p (float): tilt parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        siglM_0 (float): offset parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        siglM_p (float): tilt parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        lM0_0 (float): offset parameter for
            :math:`\\log_{10}M_0`.
        lM0_p (float): tilt parameter for
            :math:`\\log_{10}M_0`.
        lM1_0 (float): offset parameter for
            :math:`\\log_{10}M_1`.
        lM1_p (float): tilt parameter for
            :math:`\\log_{10}M_1`.
        alpha_0 (float): offset parameter for
            :math:`\\alpha`.
        alpha_p (float): tilt parameter for
            :math:`\\alpha`.
        fc_0 (float): offset parameter for
            :math:`f_c`.
        fc_p (float): tilt parameter for
            :math:`f_c`.
        bg_0 (float): offset parameter for
            :math:`\\beta_g`.
        bg_p (float): tilt parameter for
            :math:`\\beta_g`.
        bmax_0 (float): offset parameter for
            :math:`\\beta_{\\rm max}`.
        bmax_p (float): tilt parameter for
            :math:`\\beta_{\\rm max}`.
        a_pivot (float): pivot scale factor :math:`a_*`.
        ns_independent (bool): drop requirement to only form
            satellites when centrals are present.
        """
    name = 'HOD'
    is_number_counts = True

    def __init__(self, c_M_relation,
                 lMmin_0=12., lMmin_p=0., siglM_0=0.4,
                 siglM_p=0., lM0_0=7., lM0_p=0.,
                 lM1_0=13.3, lM1_p=0., alpha_0=1.,
                 alpha_p=0., fc_0=1., fc_p=0.,
                 bg_0=1., bg_p=0., bmax_0=1., bmax_p=0.,
                 a_pivot=1., ns_independent=False):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.lMmin_0 = lMmin_0
        self.lMmin_p = lMmin_p
        self.lM0_0 = lM0_0
        self.lM0_p = lM0_p
        self.lM1_0 = lM1_0
        self.lM1_p = lM1_p
        self.siglM_0 = siglM_0
        self.siglM_p = siglM_p
        self.alpha_0 = alpha_0
        self.alpha_p = alpha_p
        self.fc_0 = fc_0
        self.fc_p = fc_p
        self.bg_0 = bg_0
        self.bg_p = bg_p
        self.bmax_0 = bmax_0
        self.bmax_p = bmax_p
        self.a_pivot = a_pivot
        self.ns_independent = ns_independent
        super(HaloProfileHOD, self).__init__()

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def update_parameters(self, lMmin_0=None, lMmin_p=None,
                          siglM_0=None, siglM_p=None,
                          lM0_0=None, lM0_p=None,
                          lM1_0=None, lM1_p=None,
                          alpha_0=None, alpha_p=None,
                          fc_0=None, fc_p=None,
                          bg_0=None, bg_p=None,
                          bmax_0=None, bmax_p=None,
                          a_pivot=None,
                          ns_independent=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            lMmin_0 (float): offset parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            lMmin_p (float): tilt parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            siglM_0 (float): offset parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            siglM_p (float): tilt parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            lM0_0 (float): offset parameter for
                :math:`\\log_{10}M_0`.
            lM0_p (float): tilt parameter for
                :math:`\\log_{10}M_0`.
            lM1_0 (float): offset parameter for
                :math:`\\log_{10}M_1`.
            lM1_p (float): tilt parameter for
                :math:`\\log_{10}M_1`.
            alpha_0 (float): offset parameter for
                :math:`\\alpha`.
            alpha_p (float): tilt parameter for
                :math:`\\alpha`.
            fc_0 (float): offset parameter for
                :math:`f_c`.
            fc_p (float): tilt parameter for
                :math:`f_c`.
            bg_0 (float): offset parameter for
                :math:`\\beta_g`.
            bg_p (float): tilt parameter for
                :math:`\\beta_g`.
            bmax_0 (float): offset parameter for
                :math:`\\beta_{\\rm max}`.
            bmax_p (float): tilt parameter for
                :math:`\\beta_{\\rm max}`.
            a_pivot (float): pivot scale factor :math:`a_*`.
            ns_independent (bool): drop requirement to only form
                satellites when centrals are present
        """
        if lMmin_0 is not None:
            self.lMmin_0 = lMmin_0
        if lMmin_p is not None:
            self.lMmin_p = lMmin_p
        if lM0_0 is not None:
            self.lM0_0 = lM0_0
        if lM0_p is not None:
            self.lM0_p = lM0_p
        if lM1_0 is not None:
            self.lM1_0 = lM1_0
        if lM1_p is not None:
            self.lM1_p = lM1_p
        if siglM_0 is not None:
            self.siglM_0 = siglM_0
        if siglM_p is not None:
            self.siglM_p = siglM_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
        if fc_0 is not None:
            self.fc_0 = fc_0
        if fc_p is not None:
            self.fc_p = fc_p
        if bg_0 is not None:
            self.bg_0 = bg_0
        if bg_p is not None:
            self.bg_p = bg_p
        if bmax_0 is not None:
            self.bmax_0 = bmax_0
        if bmax_p is not None:
            self.bmax_p = bmax_p
        if a_pivot is not None:
            self.a_pivot = a_pivot
        if ns_independent is not None:
            self.ns_independent = ns_independent

    def _usat_real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = r_use[None, :] / (R_s[:, None] * bg)
        prof = 1./(x * (1 + x)**2)
        # Truncate
        prof[r_use[None, :] > R_M[:, None]*bmax] = 0

        norm = 1. / (4 * np.pi * (bg*R_s)**3 * (np.log(1+c_M) - c_M/(1+c_M)))
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _usat_fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = k_use[None, :] * R_s[:, None] * bg
        Si1, Ci1 = sici((1 + c_M[:, None]) * x)
        Si2, Ci2 = sici(x)

        P1 = 1. / (np.log(1+c_M) - c_M/(1+c_M))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
        prof = P1[:, None] * (P2 - P3)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        ur = self._usat_real(cosmo, r_use, M_use, a, mass_def)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * ur
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * ur)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a, mass_def)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * uk
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a, mass_def)

        prof = Ns[:, None] * uk
        if self.ns_independent:
            prof = 2 * Nc[:, None] * fc * prof + prof**2
        else:
            prof = Nc[:, None] * (2 * fc * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fc(self, a):
        # Observed fraction of centrals
        return self.fc_0 + self.fc_p * (a - self.a_pivot)

    def _Nc(self, M, a):
        # Number of centrals
        Mmin = 10.**(self.lMmin_0 + self.lMmin_p * (a - self.a_pivot))
        siglM = self.siglM_0 + self.siglM_p * (a - self.a_pivot)
        return 0.5 * (1 + erf(np.log(M/Mmin)/siglM))

    def _Ns(self, M, a):
        # Number of satellites
        M0 = 10.**(self.lM0_0 + self.lM0_p * (a - self.a_pivot))
        M1 = 10.**(self.lM1_0 + self.lM1_p * (a - self.a_pivot))
        alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
        return np.heaviside(M-M0, 1) * (np.fabs(M-M0) / M1)**alpha


class SatelliteShearHOD(HaloProfileHOD):
    """ Halo HOD class that calculates the satellite galaxy intrinsic shear
    field in real and fourier space, according to Fortuna et al. 2021.
    Can be used to compute halo model-based intrinsic alignment
    (angular) power spectra.

    The satellite intrinsic shear profile in real space is assumed to be
    .. math::
        \\gamma^I(r)=a_{1\\mathrm{h}}\\left(\\frac{r}{r_\\mathrm{vir}}
        \\right)^b \\sin^b\\theta,

    where :math:`a_{1\\mathrm{h}}` is the amplitude of intrinsic alignments on
    the 1-halo scale, :math:`b` the index defining the radial dependence
    and :math:`\\theta` the angle defining the projection of the
    semi-major axis of the galaxy along the line of sight.

    Args:
        c_M_relation: concentration-mass relation for the halo model.
        a1h: Amplitude of the satellite intrinsic shear profile.
        b: Power-law index of the satellite intrinsic shear profile.
            If zero, the profile is assumed to be constant inside the halo.
        lmax: Maximum multipole to be summed in the plane-wave expansion
            (Eq. (C1) in Fortuna et al. 2021, default=6).
            integration_method: Method used to obtain the fourier transform
            of the profile. Can be 'FFTLog', 'simps' or 'spline'. The FFTLog
            parameters can be updated using the self.update_precision_fftlog
            function, to attempt reducing ringing and aliasing.

    Parameters that define the numerical integration in the case of
    'simps' or 'spline' integration can also be changed using the
    self.update_r_integral_params function.
    """
    def __init__(self, c_M_relation, a1h=0.001, b=-2,
                 lmax=6, integration_method='FFTLog',
                 **kwargs):
        if lmax > 13:
            lmax = 12
            warnings.warn(
                'Maximum l provided too high. Using lmax=12.',
                category=pyccl.CCLWarning)
        elif lmax < 2:
            lmax = 2
            warnings.warn(
                'Maximum l provided too low. Using lmax=2.',
                category=pyccl.CCLWarning)
        self.a1h = a1h
        self.b = b
        self.integration_method = integration_method
        if integration_method not in ['FFTLog',
                                      'simps',
                                      'spline']:
            raise ValueError(
                'Integration method provided not supported. Use '
                '"FFTLog", "simps", or "spline".'
            )
        # If lmax is odd, make it even number (odd l contributions are zero).
        if not (lmax % 2 == 0):
            lmax = lmax-1
        self.lmax = lmax
        # Hard-code for most common cases (b=0, b=-2) to gain speed (~1.3sec).
        if self.b == 0:
            self._angular_fl = np.array([2.77582637, -0.19276603,
                                         0.04743899, -0.01779024,
                                         0.00832446, -0.00447308])\
                .reshape((6, 1))
        elif self.b == -2:
            self._angular_fl = np.array([4.71238898, -2.61799389,
                                         2.06167032, -1.76714666,
                                         1.57488973, -1.43581368])\
                .reshape((6, 1))
        else:
            self._angular_fl = np.array([self._fl(l, b=self.b)
                                         for l in range(2, self.lmax+1, 2)])\
                .reshape(self.lmax//2, 1)
        self.cM = c_M_relation
        if 'fc_0' in kwargs.keys() or 'fc_p' in kwargs.keys():
            warnings.warn(
                'Fractions for central galaxies were provided but'
                'are not tweakable and will be set to zero (fc_0=fc_p=0).',
                category=pyccl.CCLWarning)
        kwargs['fc_0'] = 0.0
        kwargs['fc_p'] = 0.0
        super(SatelliteShearHOD, self).__init__(c_M_relation,
                                                ns_independent=True,
                                                **kwargs)
        self.update_precision_fftlog(padding_lo_fftlog=1E-2,
                                     padding_hi_fftlog=1E3,
                                     n_per_decade=350,
                                     plaw_fourier=-3.7)
        self.update_r_integral_params()

    def update_r_integral_params(self, rmin=0.001, N_r=512, N_jn=10000):
        '''
        Update the parameters used in the case of Simpson's or
        spline integration.

        Args:
            rmin: Minimum value of physical radius used to carry out the
                radial integral (in Mpc). Default=0.001.
            N_r: Number of points to be used when sampling the radial integral
                (in logarithmic space). Default=512.
            N_jn: Number of points to be used when sampling the spherical
                bessel functions, that are later used to interpolate.
                Interpolating the bessel functions increases the speed of the
                computations compared to explicitly evaluating them, without
                significant loss of accuracy. Default=10000.
        '''
        self._rmin = rmin
        self._N_r = N_r
        self._N_jn = N_jn

    def _get_prefactor(self, cosmo, a, hmc):
        '''
        Compute the prefactor due to the satellite intrinsic shear
        halo model, which is the satellite galaxy fraction f_s=ns/(nc+ns)
        divided by the satellite galaxy number density ns, making it
        just 1/ngal.
        '''
        from scipy.integrate import simps
        M_use = hmc._mass
        ngal = simps(self._Nc(M_use, a) * (1+self._Ns(M_use, a)) *
                     hmc._massfunc.get_mass_function(cosmo, a=a, M=M_use),
                     hmc._lmass)
        return 1/ngal

    def _I_integral(self, a, b):
        '''
        Computes the integral
        .. math::
            I(a,b) = \\int_{-1}^1 \\mathrm{d}x (1-x^2)^{a/2}x^b =
            \\frac{((-1)^b+1)\\Gamma(a+1)\\Gamma\\left
            \\frac{b+1}{2}\\right}
            {2\\Gamma\\left(a+\\frac{b}{2}+\\frac{3}{2}\\right}.
        '''
        from scipy.special import gamma
        return (1+(-1)**b)*gamma(a/2+1)*gamma((b+1)/2)/(2*gamma(a/2+b/2+3/2))

    def _fl(self, l, thk=np.pi / 2, phik=None, b=-2):
        '''
        Computes the angular part of the satellite intrinsic shear field,
        Eq. (C8) in Fortuna et al. 2021.
        '''
        from scipy.special import binom
        gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32,
                       0, 7 * np.pi / 16, 0, 105 * np.pi / 256, 0,
                       99 * np.pi / 256])
        l_sum = 0.
        if b == 0:
            var1_add = 1
        else:
            var1_add = b
        for m in range(0, l + 1):
            m_sum = 0.
            for j in range(0, m + 1):
                m_sum += (binom(m, j) * gj[j] *
                          self._I_integral(j + var1_add, m - j) *
                          np.sin(thk) ** j * np.cos(thk) ** (m - j))
            l_sum += binom(l, m) * binom((l + m - 1) / 2, l) * m_sum
        if phik is not None:
            l_sum *= np.exp(1j * 2 * phik)
        return 2 ** l * l_sum

    def gamma_I(self, r, r_vir):
        '''
        Returns the intrinsic satellite shear,
        .. math::
            \\gamma^I(r)=a_{1\\mathrm{h}}
            \\left(\\frac{r}{r_\\mathrm{vir}}\\right)^b.
        If :math:`b` is 0, then only the value of the amplitude
        :math:`a_\\mathrm{1h}` is returned. In addition, according to
        Fortuna et al. 2021, we use a constant value of 0.06 Mpc for
        :math:`r<0.06` Mpc and set a maximum of 0.3 for :math:`\\gamma^I(r)`.
        '''
        if self.b == 0:
            return self.a1h
        else:
            r_use = np.copy(np.atleast_1d(r))
            r_use[r_use < 0.06] = 0.06
            if np.ndim(r_vir == 1):
                r_vir = r_vir.reshape(len(r_vir), 1)
            # Do not output value higher than 0.3
            gamma_out = self.a1h * (r_use/r_vir) ** self.b
            gamma_out[gamma_out > 0.3] = 0.3
            return gamma_out

    def _rvir(self, cosmo, M, a, mdef):
        """
        This translates a halo mass into a comoving radius,
        returns the comoving virial radius (in Mpc).
        """
        return mdef.get_radius(cosmo, M, a) / a

    def _real(self, cosmo, r, M, a, mass_def):
        '''
        Returns the real part of the satellite intrinsic shear field,
        .. math::
            \\gamma^I(r) u(r|M),
        with :math:`u` being the halo density profile divided by its mass.
        For now, it assumes a NFW profile.
        '''
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        rvir = self._rvir(cosmo, M_use, a, mass_def)
        # Density profile from HOD class - truncated NFW
        u = self._usat_real(cosmo, r_use, M_use, a, mass_def)
        prof = self.gamma_I(r_use, rvir) * u

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

    def _usat_fourier(self, cosmo, k, M, a, mass_def):
        '''
        Returns the fourier transform of the satellite intrinsic shear field.
        The density profile of the halo is assumed to be a truncated NFW
        profile and the radial integral (in the case of 'simps' or 'spline'
        method) is evaluated up to the virial radius.
        '''
        # TODO: Do not re-compute if already computed (to be
        #  updated with the caching CCL update).
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        l_arr = np.arange(2, self.lmax+1, 2, dtype='int32')
        if not self.integration_method == 'FFTLog':
            # Define the r-integral sampling and the sph. bessel
            # function sampling. The bessel function will be sampled
            # and interpolated to gain speed.
            r_use = np.linspace(self._rmin,
                                self._rvir(cosmo, M_use, a, mass_def),
                                self._N_r).T
            x_jn = np.geomspace(k_use.min() * r_use.min(),
                                k_use.max() * r_use.max(),
                                self._N_jn)
            jn = np.empty(shape=(len(l_arr), len(x_jn)))
        prof = np.zeros(shape=(len(M_use), len(k_use)))
        # Loop over all multipoles:
        for j, l in enumerate(l_arr):
            if self.integration_method == 'FFTLog':
                prof += (self._fftlog_wrap(cosmo, k_use,
                                           M_use, a, mass_def,
                                           l=int(l), fourier_out=True)
                         * (1j**l).real * (2*l+1) * self._angular_fl[j]
                         / (4*np.pi))
            else:
                from scipy.special import spherical_jn
                jn[j] = spherical_jn(l_arr[j], x_jn)
                k_dot_r = np.multiply.outer(k_use, r_use)
                jn_interp = np.interp(k_dot_r, x_jn, jn[j])
                integrand = r_use ** 2 * self._real(cosmo, r_use, M_use,
                                                    a, mass_def) * jn_interp
                if self.integration_method == 'simps':
                    from scipy.integrate import simps
                    for i, M_i in enumerate(M_use):
                        prof[i, :] += (simps(integrand[:, i, :],
                                             r_use[i, :]).T
                                       * (1j**l).real * (2*l+1)
                                       * self._angular_fl[j])
                elif self.integration_method == 'spline':
                    from pyccl.pyutils import _spline_integrate
                    for i, M_i in enumerate(M_use):
                        prof[i, :] += (_spline_integrate(r_use[i, :],
                                                         integrand[:, i, :],
                                                         r_use[i, 0],
                                                         r_use[i, -1]).T
                                       * (1j**l).real*(2*l+1)
                                       * self._angular_fl[j])

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
