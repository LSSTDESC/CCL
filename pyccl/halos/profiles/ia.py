import pyccl
import numpy as np
from .hod import HaloProfileHOD
from ... import warnings


__all__ = ("SatelliteShearHOD",)


class SatelliteShearHOD(HaloProfileHOD):
    """ Halo HOD class that calculates the satellite galaxy intrinsic shear
    field in real and fourier space, according to `Fortuna et al. 2021.
    <https://arxiv.org/abs/2003.02700>`_.
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
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        concentration (:obj:`~pyccl.halos.halo_model_base.Concentration`):
            concentration-mass relation to use with this profile.
        a1h (:obj:`float`): Amplitude of the satellite intrinsic shear profile.
        b (:obj:`float`): Power-law index of the satellite intrinsic shear profile.
            If zero, the profile is assumed to be constant inside the halo.
        lmax (:obj:`int`): Maximum multipole to be summed in the plane-wave expansion
            (Eq. (C1) in `Fortuna et al. 2021
            <https://arxiv.org/abs/2003.02700>`_, default=6).
        log10Mmin_0 (:obj:`float`): offset parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        log10Mmin_p (:obj:`float`): tilt parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        siglnM_0 (:obj:`float`): offset parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        siglnM_p (:obj:`float`): tilt parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        log10M0_0 (:obj:`float`): offset parameter for
            :math:`\\log_{10}M_0`.
        log10M0_p (:obj:`float`): tilt parameter for
            :math:`\\log_{10}M_0`.
        log10M1_0 (:obj:`float`): offset parameter for
            :math:`\\log_{10}M_1`.
        log10M1_p (:obj:`float`): tilt parameter for
            :math:`\\log_{10}M_1`.
        alpha_0 (:obj:`float`): offset parameter for
            :math:`\\alpha`.
        alpha_p (:obj:`float`): tilt parameter for
            :math:`\\alpha`.
        bg_0 (:obj:`float`): offset parameter for
            :math:`\\beta_g`.
        bg_p (:obj:`float`): tilt parameter for
            :math:`\\beta_g`.
        bmax_0 (:obj:`float`): offset parameter for
            :math:`\\beta_{\\rm max}`.
        bmax_p (:obj:`float`): tilt parameter for
            :math:`\\beta_{\\rm max}`.
        a_pivot (:obj:`float`): pivot scale factor :math:`a_*`.
        ns_independent (:obj:`bool`): drop requirement to only form
            satellites when centrals are present.
        integration_method (:obj:`str`): Method used to obtain the fourier transform
            of the profile. Can be ``'FFTLog'``, ``'simpson'`` or
            ``'spline'``.
        rmin (:obj:`float`): For ``'simpson'`` or ``'spline'`` integration, minimum value of
            physical radius used to carry out the radial integral (in Mpc).
        N_r (:obj:`int`): For ``'simpson'`` or ``'spline'`` integration, number of points
            to be used when sampling the radial integral (in log space).
        N_jn (:obj:`int`): For ``'simpson'`` or ``'spline'`` integration, number of points
            to be used when sampling the spherical Bessel functions, that are
            later used to interpolate. Interpolating the Bessel functions
            increases the speed of the computations compared to explicitly
            evaluating them, without significant loss of accuracy.
    """ # noqa
    __repr_attrs__ = __eq_attrs__ = (
        "a1h", "b", "lmax", "integration_method",
        "log10Mmin_0", "log10Mmin_p", "siglnM_0", "siglnM_p", "log10M0_0",
        "log10M0_p", "log10M1_0", "log10M1_p", "alpha_0", "alpha_p",
        "bg_0", "bg_p", "bmax_0", "bmax_p", "a_pivot", "ns_independent",
        "rmin", "N_r", "N_jn", "concentration", "integration_method",
        "precision_fftlog",)

    def __init__(self, *, mass_def, concentration, a1h=0.001, b=-2,
                 lmax=6, log10Mmin_0=12., log10Mmin_p=0., siglnM_0=0.4,
                 siglnM_p=0., log10M0_0=7., log10M0_p=0.,
                 log10M1_0=13.3, log10M1_p=0., alpha_0=1.,
                 alpha_p=0., bg_0=1., bg_p=0., bmax_0=1., bmax_p=0.,
                 a_pivot=1., ns_independent=False,
                 integration_method='FFTLog', rmin=0.001, N_r=512,
                 N_jn=10000):
        if lmax >= 13:
            lmax = 12
            warnings.warn(
                'Maximum l provided too high. Using lmax=12.',
                category=pyccl.CCLWarning, importance='high')
        elif lmax < 2:
            lmax = 2
            warnings.warn(
                'Maximum l provided too low. Using lmax=2.',
                category=pyccl.CCLWarning, importance='high')
        self.a1h = a1h
        self.b = b
        if integration_method not in ['FFTLog',
                                      'simpson',
                                      'spline']:
            raise ValueError("Integration method provided not "
                             "supported. Use `FFTLog`, `simpson`, "
                             "or `spline`.")
        self.integration_method = integration_method
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
        self.concentration = concentration
        self.rmin = rmin
        self.N_r = N_r
        self.N_jn = N_jn
        super().__init__(mass_def=mass_def,
                         concentration=concentration,
                         log10Mmin_0=log10Mmin_0,
                         log10Mmin_p=log10Mmin_p,
                         siglnM_0=siglnM_0,
                         siglnM_p=siglnM_p,
                         log10M0_0=log10M0_0,
                         log10M0_p=log10M0_p,
                         log10M1_0=log10M1_0,
                         log10M1_p=log10M1_p,
                         fc_0=0.0, fc_p=0.0,
                         alpha_0=alpha_0,
                         alpha_p=alpha_p,
                         bg_0=bg_0,
                         bg_p=bg_p,
                         bmax_0=bmax_0,
                         bmax_p=bmax_p,
                         a_pivot=a_pivot,
                         ns_independent=ns_independent)
        self.update_precision_fftlog(padding_lo_fftlog=1E-2,
                                     padding_hi_fftlog=1E3,
                                     n_per_decade=350,
                                     plaw_fourier=-3.7)

    def update_parameters(self, *, a1h=None, b=None, lmax=None,
                          log10Mmin_0=None, log10Mmin_p=None,
                          siglnM_0=None, siglnM_p=None,
                          log10M0_0=None, log10M0_p=None,
                          log10M1_0=None, log10M1_p=None,
                          alpha_0=None, alpha_p=None,
                          bg_0=None, bg_p=None,
                          bmax_0=None, bmax_p=None,
                          a_pivot=None, ns_independent=None,
                          rmin=None, N_r=None, N_jn=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            a1h (:obj:`float`): Amplitude of the satellite intrinsic shear profile.
            b (:obj:`float`): Power-law index of the satellite intrinsic shear
                profile. If zero, the profile is assumed to be constant inside
                the halo.
            lmax (:obj:`int`): Maximum multipole to be summed in the plane-wave
                expansion.
            log10Mmin_0 (:obj:`float`): offset parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            log10Mmin_p (:obj:`float`): tilt parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            siglnM_0 (:obj:`float`): offset parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            siglnM_p (:obj:`float`): tilt parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            log10M0_0 (:obj:`float`): offset parameter for
                :math:`\\log_{10}M_0`.
            log10M0_p (:obj:`float`): tilt parameter for
                :math:`\\log_{10}M_0`.
            log10M1_0 (:obj:`float`): offset parameter for
                :math:`\\log_{10}M_1`.
            log10M1_p (:obj:`float`): tilt parameter for
                :math:`\\log_{10}M_1`.
            alpha_0 (:obj:`float`): offset parameter for
                :math:`\\alpha`.
            alpha_p (:obj:`float`): tilt parameter for
                :math:`\\alpha`.
            bg_0 (:obj:`float`): offset parameter for
                :math:`\\beta_g`.
            bg_p (:obj:`float`): tilt parameter for
                :math:`\\beta_g`.
            bmax_0 (:obj:`float`): offset parameter for
                :math:`\\beta_{\\rm max}`.
            bmax_p (:obj:`float`): tilt parameter for
                :math:`\\beta_{\\rm max}`.
            a_pivot (:obj:`float`): pivot scale factor :math:`a_*`.
            ns_independent (:obj:`bool`): drop requirement to only form
                satellites when centrals are present
            rmin (:obj:`float`): For `simpson` or `spline` integration, minimum value of
                physical radius used to carry out the radial integral (in Mpc).
            N_r (:obj:`int`): For `simpson` or `spline` integration, number of points to be
                used when sampling the radial integral (in logarithmic space).
            N_jn (:obj:`int`): For `simpson` or `spline` integration, number of points to
                be used when sampling the spherical Bessel functions, that are
                later used to interpolate. Interpolating the Bessel functions
                increases the speed of the computations compared to explicitly
                evaluating them, without significant loss of accuracy.
        """ # noqa
        if a1h is not None:
            self.a1h = a1h
        if b is not None:
            self.b = b
        if lmax is not None:
            self.lmax = lmax
        if log10Mmin_0 is not None:
            self.log10Mmin_0 = log10Mmin_0
        if log10Mmin_p is not None:
            self.log10Mmin_p = log10Mmin_p
        if log10M0_0 is not None:
            self.log10M0_0 = log10M0_0
        if log10M0_p is not None:
            self.log10M0_p = log10M0_p
        if log10M1_0 is not None:
            self.log10M1_0 = log10M1_0
        if log10M1_p is not None:
            self.log10M1_p = log10M1_p
        if siglnM_0 is not None:
            self.siglnM_0 = siglnM_0
        if siglnM_p is not None:
            self.siglnM_p = siglnM_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
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
        if rmin is not None:
            self.rmin = rmin
        if N_r is not None:
            self.N_r = N_r
        if N_jn is not None:
            self.N_jn = N_jn

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
        Eq. (C8) in `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`_.
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

    def get_normalization(self, cosmo, a, *, hmc):
        """Returns the normalization of this profile, which is the
        mean galaxy number density.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.
            hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
                model calculator object.

        Returns:
            float: normalization factor of this profile.
        """
        def integ(M):
            Nc = self._Nc(M, a)
            Ns = self._Ns(M, a)
            if self.ns_independent:
                return Nc+Ns
            return Nc*(1+Ns)
        return hmc.integrate_over_massfunc(integ, cosmo, a)

    def gamma_I(self, r, r_vir):
        '''
        Returns the intrinsic satellite shear,

        .. math::
            \\gamma^I(r)=a_{1\\mathrm{h}}
            \\left(\\frac{r}{r_\\mathrm{vir}}\\right)^b.

        If :math:`b` is 0, then only the value of the amplitude
        :math:`a_\\mathrm{1h}` is returned. In addition, according to
        `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`_, we
        use a constant value of 0.06 Mpc for
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

    def _real(self, cosmo, r, M, a):
        '''
        Returns the real part of the satellite intrinsic shear field,

        .. math::
            \\gamma^I(r) u(r|M),

        with :math:`u` being the halo density profile divided by its mass.
        For now, it assumes a NFW profile.
        '''
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
        '''
        Returns the fourier transform of the satellite intrinsic shear field.
        The density profile of the halo is assumed to be a truncated NFW
        profile and the radial integral (in the case of 'simpson' or 'spline'
        method) is evaluated up to the virial radius.
        '''
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        l_arr = np.arange(2, self.lmax+1, 2, dtype='int32')
        if not self.integration_method == 'FFTLog':
            # Define the r-integral sampling and the sph. bessel
            # function sampling. The bessel function will be sampled
            # and interpolated to gain speed.
            r_use = np.linspace(self.rmin,
                                self.mass_def.get_radius(cosmo, M_use, a) / a,
                                self.N_r).T
            x_jn = np.geomspace(k_use.min() * r_use.min(),
                                k_use.max() * r_use.max(),
                                self.N_jn)
            jn = np.empty(shape=(len(l_arr), len(x_jn)))
        prof = np.zeros(shape=(len(M_use), len(k_use)))
        # Loop over all multipoles:
        for j, l in enumerate(l_arr):
            if self.integration_method == 'FFTLog':
                prof += (self._fftlog_wrap(cosmo, k_use, M_use, a,
                                           ell=int(l), fourier_out=True)
                         * (1j**l).real * (2*l+1) * self._angular_fl[j]
                         / (4*np.pi))
            else:
                from scipy.special import spherical_jn
                jn[j] = spherical_jn(l_arr[j], x_jn)
                k_dot_r = np.multiply.outer(k_use, r_use)
                jn_interp = np.interp(k_dot_r, x_jn, jn[j])
                integrand = (r_use**2 * jn_interp *
                             self._real(cosmo, r_use, M_use, a))
                if self.integration_method == 'simpson':
                    from scipy.integrate import simpson
                    for i, M_i in enumerate(M_use):
                        prof[i, :] += (simpson(integrand[:, i, :],
                                               x=r_use[i, :]).T
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
