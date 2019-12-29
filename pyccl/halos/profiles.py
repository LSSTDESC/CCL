from .. import ccllib as lib
from ..core import check
from ..power import sigmaM
from ..pyutils import resample_array
from .concentration import Concentration
from .massdef import MassDef
import numpy as np
from scipy.special import sici


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

        * `padding_lo_fftlog`: when computing a Hankel
          transform we often need to extend the range of the
          input (e.g. the r-range for the real-space profile
          when computing the Fourier-space one) to avoid
          aliasing and boundary effects. This parameter
          controls the factor by which we multiply the lower
          end of the range (e.g. a value of 0.1 implies that
          we will extend the range by one decade on the
          left). Note that FFTLog works in logarithmic
          space. Default value: 0.1.
        * `padding_hi_fftlog`: same as `padding_lo_fftlog`
          for the upper end of the range (e.g. a value of
          10 implies extending the range by one decade on
          the right). Default value: 10.
        * `n_per_decade`: number of samples of the profile
          taken per decade when computing Hankel transforms.
        * `padding_lo_extra`: when computing the projected
          2D profile or the 2D cumulative density,
          sometimes two Hankel transforms are needed (from
          3D real-space to Fourier, then from Fourier to
          2D real-space). This parameter controls the k range
          of the intermediate transform. The logic here is to
          avoid the range twice by `padding_lo_fftlog` (which
          can be overkill and slow down the calculation).
          Default value: 0.1.
        * `padding_hi_extra`: same as `padding_lo_extra` for
          the upper end of the range. Default value: 10.
        * `large_padding_2D`: if set to `True`, the
          intermediate Hankel transform in the calculation of
          the 2D projected profile and cumulative mass
          density will use `padding_lo_fftlog` and
          `padding_hi_fftlog` instead of `padding_lo_extra`
          and `padding_hi_extra` to extend the range of the
          intermediate Hankel transform.
        * `extrapol`: type of extrapolation used in the uncommon
          scenario that FFTLog returns a profile on a range that
          does not cover the intended output range. Pass
          `linx_liny` if you want to extrapolate linearly in the
          profile and `linx_logy` if you want to extrapolate
          linearly in its logarithm. Default value: `linx_liny`.
        * `plaw_fourier`: FFTLog is able to perform more
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
        * `plaw_projected`: same as `plaw_fourier` for the
          calculation of the 2D projected and cumulative density
          profiles. Finer control can be achieved with the
          `_get_plaw_projected`. The default value allows for a
          slightly faster (but potentially less accurate) FFTLog
          transform.  Some level of experimentation with this
          parameter is recommended when implementing a new profile.
          Default value: -1.
        """
        self.precision_fftlog.update(kwargs)

    def _get_plaw_fourier(self, cosmo, M, a, mass_def):
        """ This controls the value of `plaw_fourier` to be used
        as a function of cosmology, halo mass, and scale factor.

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            M (float): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_fourier']

    def _get_plaw_projected(self, cosmo, M, a, mass_def):
        """ This controls the value of `plaw_projected` to be
        used as a function of cosmology, halo mass, and scale
        factor.

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            M (float): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float: power law index to be used with FFTLog.
        """
        return self.precision_fftlog['plaw_projected']

    def real(self, cosmo, r, M, a, mass_def=None):
        """ Returns the 3D  real-space value of the profile as a
        function of cosmology, radius, halo mass and scale factor.

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_r, N_M)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively.
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

    def fourier(self, cosmo, k, M, a, mass_def=None):
        """ Returns the Fourier-space value of the profile as a
        function of cosmology, wavenumber, halo mass and
        scale factor.

        .. math::
           \\rho(k)=\\frac{1}{2\pi^2} \\int dr\\, r^2\\,
            \\rho(r)\\, j_0(k r)

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_k, N_M)` where `N_k` and `N_m` are
            the sizes of `k` and `M` respectively.
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

    def projected(self, cosmo, r_t, M, a, mass_def=None):
        """ Returns the 2D projected profile as a function of
        cosmology, radius, halo mass and scale factor.

        .. math::
           \\Sigma(R)= \\int dr_\\parallel\\,
           \\rho(\\sqrt{r_\\parallel^2 + R^2})

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_r, N_M)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively.
        """
        if getattr(self, '_projected', None):
            s_r_t = self._projected(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._projected_fftlog_wrap(cosmo, r_t, M,
                                                a, mass_def)
        return s_r_t

    def cumul2d(self, cosmo, r_t, M, a, mass_def=None):
        """ Returns the 2D cumulative surface density as a
        function of cosmology, radius, halo mass and scale
        factor.

        .. math::
           \\Sigma(<R)= \\frac{2}{R^2} \\int dR'\\, R'\\,
           \\Sigma(R')

        Args:
            cosmo (:obj:`Cosmology`): a Cosmology object.
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def (:obj:`MassDef`): a mass definition object.

        Returns:
            float or array_like: halo profile. The shape of the
            output will be `(N_r, N_M)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively.
        """
        if getattr(self, '_cumul2d', None):
            s_r_t = self._cumul2d(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._cumul2d_fftlog_wrap(cosmo, r_t, M,
                                              a, mass_def)
        return s_r_t

    def _fftlog_wrap(self, cosmo, k, M, a, mass_def,
                     fourier_out=False,
                     large_padding=True):
        # This computes the 3D Hankel transform
        #  \rho(k) = 4\pi \int dr r^2 \rho(r) j_0(k r)
        # if fourier_out == False, and
        #  \rho(r) = \frac{1}{2\pi^2} \int dk k^2 \rho(k) j_0(k r)
        # otherwise.

        # Select which profile should be the input
        if fourier_out:
            p_func = self._real
        else:
            p_func = self._fourier
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)

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

        p_k_out = np.zeros([M_use.size, k_use.size])
        for im, mass in enumerate(M_use):
            # Compute real profile values
            p_real = p_func(cosmo, r_arr, mass, a, mass_def)

            # Power-law index to pass to FFTLog.
            plaw_index = self._get_plaw_fourier(cosmo, mass,
                                                a, mass_def)
            # Compute Fourier profile through fftlog
            status = 0
            result, status = lib.fftlog_transform(r_arr, p_real,
                                                  3, 0, plaw_index,
                                                  2 * r_arr.size, status)
            check(status)
            k_arr, p_k_arr = result.reshape([2, r_arr.size])

            # Resample into input k values
            p_fourier = resample_array(np.log(k_arr), p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier

        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        p_k_out = p_k_out.T
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, mass_def):
        # This computes Sigma(<R) from the Fourier-space profile as:
        # Sigma(R) = \frac{1}{2\pi} \int dk k J_0(k R) \rho(k)
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([M_use.size, r_t_use.size])
        for im, mass in enumerate(M_use):
            # Compute Fourier-space profile
            if getattr(self, '_fourier', None):
                # Compute from `_fourier` if available.
                p_fourier = self._fourier(cosmo, k_arr, mass,
                                          a, mass_def)
            else:
                # Compute with FFTLog otherwise.
                lpad = self.precision_fftlog['large_padding_2D']
                p_fourier = self._fftlog_wrap(cosmo,
                                              k_arr,
                                              mass, a,
                                              mass_def,
                                              fourier_out=True,
                                              large_padding=lpad)

            # Power-law index to pass to FFTLog.
            plaw_index = self._get_plaw_projected(cosmo, mass,
                                                  a, mass_def)
            # Compute projected profile through fftlog
            status = 0
            result, status = lib.fftlog_transform(k_arr, p_fourier,
                                                  2, 0, plaw_index,
                                                  2 * k_arr.size, status)
            check(status)
            r_t_arr, sig_r_t_arr = result.reshape([2, k_arr.size])

            # Resample into input r_t values
            sig_r_t = resample_array(np.log(r_t_arr), sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t
        sig_r_t_out = sig_r_t_out.T

        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out

    def _cumul2d_fftlog_wrap(self, cosmo, r_t, M, a, mass_def):
        # This computes Sigma(<R) from the Fourier-space profile as:
        # Sigma(<R) = \frac{1}{2\pi} \int dk k 2 J_1(k R)/(k R) \rho(k)
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([M_use.size, r_t_use.size])
        for im, mass in enumerate(M_use):
            # Compute Fourier-space profile
            if getattr(self, '_fourier', None):
                # Compute from `_fourier` if available.
                p_fourier = self._fourier(cosmo, k_arr, mass,
                                          a, mass_def)
            else:
                # Compute with FFTLog otherwise.
                lpad = self.precision_fftlog['large_padding_2D']
                p_fourier = self._fftlog_wrap(cosmo,
                                              k_arr,
                                              mass, a,
                                              mass_def,
                                              fourier_out=True,
                                              large_padding=lpad)
            # The cumulative profile involves a factor 1/(k R) in
            # the integrand.
            p_fourier *= 2 / k_arr

            # Power-law index to pass to FFTLog.
            plaw_index = self._get_plaw_projected(cosmo, mass,
                                                  a, mass_def) - 1
            # Compute cumulative surface density through fftlog
            status = 0
            result, status = lib.fftlog_transform(k_arr, p_fourier,
                                                  2, 1, plaw_index,
                                                  2 * k_arr.size, status)
            check(status)
            r_t_arr, sig_r_t_arr = result.reshape([2, k_arr.size])
            sig_r_t_arr /= r_t_arr

            # Resample into input r_t values
            sig_r_t = resample_array(np.log(r_t_arr), sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t
        sig_r_t_out = sig_r_t_out.T

        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out


class HaloProfileGaussian(HaloProfile):
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
        prof = np.exp(-(r_use[:, None] / rs[None, :])**2)
        prof = prof * rho0[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfilePowerLaw(HaloProfile):
    name = 'PowerLaw'

    def __init__(self, r_scale, tilt):
        self.r_s = r_scale
        self.tilt = tilt
        super(HaloProfilePowerLaw, self).__init__()

    def _get_plaw_fourier(self, cosmo, M, a, mass_def):
        return self.tilt(cosmo, M, a, mass_def)

    def _get_plaw_projected(self, cosmo, M, a, mass_def):
        return -3 - self.tilt(cosmo, M, a, mass_def)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_s(cosmo, M_use, a, mass_def)
        tilt = self.tilt(cosmo, M_use, a, mass_def)
        # Form factor
        prof = (r_use[:, None] / rs[None, :])**tilt[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileNFW(HaloProfile):
    name = 'NFW'

    def __init__(self, c_M_relation, fourier_analytic=False,
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

        x = r_use[:, None] / R_s[None, :]
        prof = 1./(x * (1 + x)**2)
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        norm = self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
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

        x = r_use[:, None] / R_s[None, :]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
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

        x = r_use[:, None] / R_s[None, :]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = k_use[:, None] * R_s[None, :]
        Si2, Ci2 = sici(x)
        P1 = M / (np.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[None, :]) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c_M[None, :] * x) / ((1 + c_M[None, :]) * x)
            prof = P1[None, :] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[None, :] * P2

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileEinasto(HaloProfile):
    name = 'Einasto'

    def __init__(self, c_M_relation, truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        super(HaloProfileEinasto, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _get_alpha(self, cosmo, M, a, mdef):
        mdef_vir = MassDef('vir', 'matter')
        Mvir = mdef.translate_mass(cosmo, M, a, mdef_vir)
        sM = sigmaM(cosmo, Mvir, a)
        nu = 1.686 / sM
        alpha = 0.155 + 0.0095 * nu * nu
        return alpha

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, M_use, a, mass_def)

        status = 0
        norm, status = lib.einasto_norm(R_s, R_M, alpha, M_use.size, status)
        check(status)
        norm = M_use / norm

        x = r_use[:, None] / R_s[None, :]
        prof = norm[None, :] * np.exp(-2. * (x**alpha[None, :] - 1) /
                                      alpha[None, :])
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileHernquist(HaloProfile):
    name = 'Hernquist'

    def __init__(self, c_M_relation, truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        super(HaloProfileHernquist, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        status = 0
        norm, status = lib.hernquist_norm(R_s, R_M, M_use.size, status)
        check(status)
        norm = M_use / norm

        x = r_use[:, None] / R_s[None, :]
        prof = norm[None, :] / (x * (1 + x)**3)
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
