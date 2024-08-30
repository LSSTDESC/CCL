__all__ = ("EFTCalculator",)

import warnings

import numpy as np

from .. import (CCLAutoRepr, CCLError, CCLWarning, Pk2D,
                get_pk_spline_a, unlock_instance)


# TODO: Update docstring for EFT!
class EFTCalculator(CCLAutoRepr):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    effective field theory-based correlations. These calculations
    are currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    In the parametrisation used here, the galaxy overdensity
    is expanded as:

    .. math::
        \\delta_g=b_1\\,\\delta+\\frac{b_2}{2}\\delta^2+
        \\frac{b_s}{2}s^2+\\frac{b_{3nl}}{2}\\psi_{nl}+
        \\frac{b_{k2}}{2}\\nabla^2\\delta.

    In turn, the intrinsic alignment component is expanded as

    .. math::
        s^I_{ij}=c_1\\,s_{ij}+c_2(s_{ik}s_{jk}-s^2\\delta_{ik}/3)
        +c_\\delta\\,\\delta\\,s_{ij}

    (note that the higher-order terms are not divided by 2!).

    .. note:: Only the leading-order non-local term (i.e.
              :math:`\\langle \\delta\\,\\nabla^2\\delta`) is
              taken into account in the expansion. All others are
              set to zero.

    .. note:: Terms of the form
              :math:`\\langle \\delta^2 \\psi_{nl}\\rangle` (and
              likewise for :math:`s^2`) are set to zero.

    .. warning:: The full non-linear model for the cross-correlation
                 between number counts and intrinsic alignments is
                 still work in progress in FastPT. As a workaround
                 CCL assumes a non-linear treatment of IAs, but only
                 linearly biased number counts.

    .. note:: This calculator does not account for any form of
              stochastic bias contribution to the power spectra.
              If necessary, consider adding it in post-processing.

    Args:
        with_NC (:obj:`bool`): set to ``True`` if you'll want to use
            this calculator to compute correlations involving
            number counts.
        with_IA(:obj:`bool`): set to ``True`` if you'll want to use
            this calculator to compute correlations involving
            intrinsic alignments.
        with_matter_1loop(:obj:`bool`): set to ``True`` if you'll want to use
            this calculator to compute the one-loop matter power
            spectrum (automatically on if ``with_NC==True``).
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            If present, internal PT power spectrum templates will
            be initialized. If ``None``, you will need to initialize
            them using the :meth:`update_ingredients` method.
        log10k_min (:obj:`float`): decimal logarithm of the minimum
            Fourier scale (in :math:`{\\rm Mpc}^{-1}`) for which you want to
            calculate perturbation theory quantities. CHECK THIS
        log10k_max (:obj:`float`): decimal logarithm of the maximum
            Fourier scale (in :math:`{\\rm Mpc}^{-1}`) for which you want to
            calculate perturbation theory quantities. CHECK THIS
        nk_per_decade (:obj:`int` or :obj:`float`): number of k values per
            decade.
        a_arr (array): array of values of the scale factor at
            which all power spectra will be evaluated. If ``None``,
            the default sampling used internally by CCL will be
            used. Note that this may be slower than a bespoke sampling
            optimised for your particular application.
        k_cutoff (:obj:`float`): exponential cutoff scale. All power
            spectra will be multiplied by a cutoff factor of the
            form :math:`\\exp(-(k/k_*)^n)`, where :math:`k_*` is
            the cutoff scale. This may be useful when using the
            resulting power spectra to compute correlation
            functions if some of the PT contributions do not
            fall sufficiently fast on small scales. If ``None``
            (default), no cutoff factor will be applied.
        n_exp_cutoff (:obj:`float`): exponent of the cutoff factor (see
            ``k_cutoff``).
        pad_factor (:obj:`float`): fraction of the :math:`\\log_{10}(k)`
             interval you to add as padding for FFTLog calculations.
        low_extrap (:obj:`float`): decimal logaritm of the minimum Fourier
             scale (in :math:`{\\rm Mpc}^{-1}`) for which FAST-PT will
             extrapolate.
        high_extrap (:obj:`float`): decimal logaritm of the maximum Fourier
             scale (in :math:`{\\rm Mpc}^{-1}`) for which FAST-PT will
             extrapolate.
        P_window (array): 2-element array describing the
             tapering window used by FAST-PT. See FAST-PT
             documentation for more details.
        C_window (:obj:`float`):  `C_window` parameter used by FAST-PT to
             smooth the edges and avoid ringing. See FAST-PT
             documentation for more details.
        sub_lowk (:obj:`bool`): if ``True``, the small-scale white noise
             contribution to some of the terms will be subtracted. SHOULD I??
    """
    __repr_attrs__ = __eq_attrs__ = ('with_NC', 'with_IA', 'with_matter_1loop',
                                     'k_s', 'a_s', 'exp_cutoff', 'fastpt_par', )

    def __init__(self, *, with_NC=False, with_IA=False,
                 with_matter_1loop=True, cosmo=None,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 a_arr=None, k_cutoff=None, n_exp_cutoff=4,
                 pad_factor=1.0, low_extrap=-5.0, high_extrap=3.0,
                 P_window=None, C_window=0.75, sub_lowk=False):
        self.with_matter_1loop = with_matter_1loop
        self.with_NC = with_NC
        self.with_IA = with_IA

        # Set FAST-PT parameters
        self.fastpt_par = {'pad_factor': pad_factor,
                           'low_extrap': low_extrap,
                           'high_extrap': high_extrap,
                           'P_window': P_window,
                           'C_window': C_window,
                           'sub_lowk': sub_lowk}

        to_do = ['EFT']
        if self.with_NC: # FIXME: To optimize integrals, but not necessary.
            to_do.append('dd_bias')
        if self.with_IA:
            to_do.append('IA')

        # k sampling
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.k_s = np.logspace(log10k_min, log10k_max, nk_total)

        # a sampling
        if a_arr is None:
            a_arr = get_pk_spline_a()
        self.a_s = a_arr.copy()
        self.z_s = 1/self.a_s-1

        # Cutoff factor
        if k_cutoff is not None:
            self.exp_cutoff = np.exp(-(self.k_s/k_cutoff)**n_exp_cutoff)
            self.exp_cutoff = self.exp_cutoff[None, :]
        else:
            self.exp_cutoff = 1

        # Call FAST-PT
        import fastpt as fpt
        n_pad = int(self.fastpt_par['pad_factor'] * len(self.k_s))
        self.pt = fpt.FASTPT(self.k_s, to_do=to_do,
                             low_extrap=self.fastpt_par['low_extrap'],
                             high_extrap=self.fastpt_par['high_extrap'],
                             n_pad=n_pad)

        # Initialize all expensive arrays to ``None``.
        self._cosmo = None

        # Fill them out if cosmo is present
        if cosmo is not None:
            self.update_ingredients(cosmo)

        # List of Pk2Ds to fill out
        self._pk2d_temp = {}

    def _check_init(self):
        if self.initialised:
            return
        raise CCLError("PT templates have not been initialised "
                       "for this calculator. Please do so using "
                       "`update_ingredients`.")

    @property
    def initialised(self):
        return hasattr(self, "_pklin")

    @unlock_instance
    def update_ingredients(self, cosmo):
        """ Update the internal PT arrays.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        """
        if self.initialised and (cosmo == self._cosmo):
            return

        pklz0 = cosmo.linear_matter_power(self.k_s, 1.0)
        g = cosmo.growth_factor(self.a_s)
        self._g4 = g**4
        self._g4T = self._g4[:, None]

        kw = {'P': pklz0, 'P_window': self.fastpt_par['P_window'],
              'C_window': self.fastpt_par['C_window']}

        (self.I11, self.I12, self.I13, self.I22, self.I23, self.I24,
            self.I33, self.I34, self.I44, self.I55, self.J1, self.J2,
            self.J3) = self.pt.eft_integrals(**kw)
        self.I14 = ((28*self.I12-self.I22+self.I23)/(14*np.sqrt(6)) +
                    5*(self.I34-self.I24)/7)
        self.I66 = (2*self.I22-2*np.sqrt(6)*self.I24+3*self.I44)/18
        self.I67 = (2*self.I22+6*self.I23-5*np.sqrt(6)*self.I24 -
                    3*np.sqrt(6)*self.I34+12*self.I44)/72
        self.I77 = (self.I22+6*self.I23+9*self.I33-4*np.sqrt(6)*self.I24 -
                    12*np.sqrt(6)*self.I34+24*self.I44)/144

        self._pklin = np.array([cosmo.linear_matter_power(self.k_s, a)
                                for a in self.a_s])

        # Galaxy clustering templates
        if self.with_NC:
            pass
        elif self.with_matter_1loop:  # Only 1-loop matter needed
            pass
        # Intrinsic alignment templates
        if self.with_IA:
            pass

    def _get_pgg(self, tr1, tr2):
        """ Get the number counts auto-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            tr1 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): first
                tracer to correlate.
            tr2 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): second
                tracer to correlate.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        Ps1s1 = self._g4T * 2 * (self.I11 + self.J1)
        Ps1s21 = self._g4T * 2 * self.I12
        Ps21s21 = self._g4T * 2 * self.I22
        Ps1s22 = self._g4T * 2 * self.I13
        Ps21s22 = self._g4T * 2 * self.I23
        Ps22s22 = self._g4T * 2 * self.I33
        Ps1s31 = self._g4T * self.J2

        # Get biases
        b1s_1 = tr1.b1s(self.z_s)
        b21s_1 = tr1.b21s(self.z_s)
        b22s_1 = tr1.b22s(self.z_s)
        b31s_1 = tr1.b31s(self.z_s)
        bRs_1 = tr1.bRs(self.z_s)

        b1s_2 = tr2.b1s(self.z_s)
        b21s_2 = tr2.b21s(self.z_s)
        b22s_2 = tr2.b22s(self.z_s)
        b31s_2 = tr2.b31s(self.z_s)
        bRs_2 = tr2.bRs(self.z_s)

        # FIXME: What is the value of R?
        # Make clear bRs is actually b_R^s * R^2.
        R = 1
        pgg = (((b1s_1*b1s_2)[:, None] +
                (b1s_1*bRs_2+b1s_2*bRs_1)[:, None]*R**2*self.k_s[None,:]**2) *
               self._pklin +
               (b1s_1*b1s_2)[:, None]*Ps1s1 +
               (b1s_1*b21s_2+b1s_2*b21s_1)[:, None]*Ps1s21 +
               (b21s_1*b21s_2)[:, None]*Ps21s21 +
               (b1s_1*b22s_2+b22s_1*b1s_2)[:, None]*Ps1s22 +
               (b21s_1*b22s_2+b21s_2*b22s_1)[:, None]*Ps21s22 +
               (b22s_1*b22s_2)[:, None]*Ps22s22 +
               (b1s_1*b31s_2+b1s_2*b31s_1)[:, None]*Ps1s31
               )

        return pgg*self.exp_cutoff

    def _get_pgi(self, trg, tri):
        """ Get the number counts - IA cross-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            trg (:class:`~pyccl.nl_eft.tracers.EFTTracer`): number
                counts tracer.
            tri (:class:`~pyccl.nl_eft.tracers.EFTTracer`): intrinsic
                alignment tracer.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        Ps1g1 = 2 * np.sqrt(2/3) * self._g4T * (self.I11 + self.J1)
        Ps21g1 = 2 * np.sqrt(2/3) * self._g4T * self.I12
        Ps22g1 = 2 * np.sqrt(2/3) * self._g4T * self.I13
        Ps1g21 = 2 * self._g4T * self.I14
        Ps21g21 = 2 * self._g4T * self.I24
        Ps22g21 = 2 * self._g4T * self.I34
        Ps1g22 = np.sqrt(1/6) * self._g4T * (self.I13-self.I12)
        Ps21g22 = np.sqrt(1/6) * self._g4T * (self.I23-self.I22)
        Ps22g22 = np.sqrt(1/6) * self._g4T * (self.I33-self.I23)
        Ps1g31 = np.sqrt(2/3) * self._g4T * self.J2
        Ps1g32 = np.sqrt(2/3) * self._g4T * self.J3

        # Get biases
        b1s = trg.b1s(self.z_s)
        b21s = trg.b21s(self.z_s)
        b22s = trg.b22s(self.z_s)
        b31s = trg.b31s(self.z_s)
        bRs = trg.bRs(self.z_s)

        b1g = tri.b1g(self.z_s)
        b21g = tri.b21g(self.z_s)
        b22g = tri.b22g(self.z_s)
        b23g = tri.b23g(self.z_s)
        b31g = tri.b31g(self.z_s)
        b32g = tri.b32g(self.z_s)
        bRg = tri.bRg(self.z_s)


        # Make clear bRs is actually b_R^s * R^2.
        R = 1
        pgi = (((b1s*b1g)[:, None] +
                (b1s*bRg+b1g*bRs)[:, None]*R**2*self.k_s[None,:]**2) *
               np.sqrt(2/3) * self._pklin +
               (b1s*b1g)[:, None] * Ps1g1 +
               (b21s*b1g)[:, None] * Ps21g1 +
               (b22s*b1g)[:, None] * Ps22g1 +
               (b1s*b21g)[:, None] * Ps1g21 +
               (b21s*b21g)[:, None] * Ps21g21 +
               (b22s*b21g)[:, None] * Ps22g21 +
               (b1s*b22g)[:, None] * Ps1g22 +
               (b21s*b22g)[:, None] * Ps21g22 +
               (b22s*b22g)[:, None] * Ps22g22 +
               (b31s*b1g+b1s*b31g)[:, None] * Ps1g31 +
               (b1s*b32g)[:, None] * Ps1g32
               )

        return (1/2)*np.sqrt(3/2)*pgi*self.exp_cutoff


    def _get_pgm(self, trg):
        """ Get the number counts - matter cross-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            trg (:class:`~pyccl.nl_pt.tracers.PTTracer`): number
                counts tracer.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates
        Pd1d1 = self._pklin

        # Get biases
        b1s = trg.b1s(self.z_s)
        b21s = trg.b21s(self.z_s)
        b22s = trg.b22s(self.z_s)
        b31s = trg.b31s(self.z_s)
        bRs = trg.bRs(self.z_s)

        pgm = self._pklin

        return pgm*self.exp_cutoff

    def _get_pii(self, tr1, tr2, return_bb=False):
        """ Get the intrinsic alignment auto-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            tr1 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): first tracer
                to correlate.
            tr2 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): first tracer
                to correlate.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates
        Pg1g1_0 = 4 / 3 * self._g4T * (self.I11+self.J1)
        Pg1g21_0 = 2 * np.sqrt(2/3) * self._g4T * self.I14
        Pg1g22_0 = 1 / 3 * self._g4T * (self.I13-self.I12)
        Pg22g22_0 = 1 / 12 * self._g4T * (self.I22-2*self.I23+self.I33)
        Pg21g22_0 = 1 / 2 * np.sqrt(2/3) * self._g4T * (self.I34-self.I24)
        Pg21g21_0 = 2 * self._g4T * self.I44
        Pg1g31_0 = 2 / 3 * self._g4T * self.J2
        Pg1g32_0 = 2 / 3 * self._g4T * self.J3
        Pg21g21_1 = 2 * self._g4T * self.I55
        Pg21g21_2 = 2 * self._g4T * self.I66
        Pg21g23_2 = 2 * self._g4T * (self.I67-self.I66)
        Pg23g23_2 = 2 * self._g4T * (self.I66-2*self.I67+self.I77)

        # Get biases
        b1g_1 = tr1.b1g(self.z_s)
        b21g_1 = tr1.b21g(self.z_s)
        b22g_1 = tr1.b22g(self.z_s)
        b23g_1 = tr1.b23g(self.z_s)
        b31g_1 = tr1.b31g(self.z_s)
        b32g_1 = tr1.b32g(self.z_s)
        bRg_1 = tr1.bRg(self.z_s)
        
        b1g_2 = tr2.b1g(self.z_s)
        b21g_2 = tr2.b21g(self.z_s)
        b22g_2 = tr2.b22g(self.z_s)
        b23g_2 = tr2.b23g(self.z_s)
        b31g_2 = tr2.b31g(self.z_s)
        b32g_2 = tr2.b32g(self.z_s)
        bRg_2 = tr2.bRg(self.z_s)

        if return_bb:
            pii = ((b21g_1*b21g_2)[:, None]*Pg21g21_1 +
                   (b21g_1*b21g_2)[:, None]*Pg21g21_2 +
                   (b21g_1*b23g_2+b21g_2*b23g_1)[:, None]*Pg21g23_2 +
                   (b23g_1*b23g_2)[:, None]*Pg23g23_2
                   )/2
        else:
            R = 1
            pii = (((b1g_1*b1g_2)[:, None] +
                    (b1g_1*bRg_2+b1g_2*bRg_1)[:, None] *
                    R**2*self.k_s[None,:]**2)*np.sqrt(2/3) * self._pklin +
                   (b1g_1*b1g_2)[:, None]*Pg1g1_0*3/8 +
                   (b1g_1*b21g_2+b1g_2*b21g_1)[:, None]*Pg1g21_0*3/8 +
                   (b1g_1*b22g_2+b1g_2*b22g_1)[:, None]*Pg1g22_0*3/8 +
                   (b22g_1*b22g_2)[:, None]*Pg22g22_0*3/8 +
                   (b21g_1*b22g_2+b21g_2*b22g_1)[:, None]*Pg21g22_0*3/8 +
                   (b21g_1*b21g_2)[:, None]*Pg21g21_0*3/8 +
                   (b1g_1*b31g_2+b1g_2*b31g_1)[:, None]*Pg1g31_0*3/8 +
                   (b1g_1*b32g_2+b1g_2*b32g_1)[:, None]*Pg1g32_0*3/8 +
                   (b21g_1*b21g_2)[:, None]*Pg21g21_1*1/2 +
                   (b21g_1*b21g_2)[:, None]*Pg21g21_2*1/8 +
                   (b21g_1*b23g_2+b21g_2*b23g_1)[:, None]*Pg21g23_2*1/8 +
                   (b23g_1*b23g_2)[:, None]*Pg23g23_2*1/8
                   )

        return pii*self.exp_cutoff

    def _get_pim(self, tri):
        """ Get the matter - IA cross-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            tri (:class:`~pyccl.nl_pt.tracers.PTTracer`): intrinsic
                alignment tracer.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates


        # Get biases
        b1g = tri.b1g(self.z_s)
        b21g = tri.b21g(self.z_s)
        b22g = tri.b22g(self.z_s)
        b23g = tri.b23g(self.z_s)
        b31g = tri.b31g(self.z_s)
        b32g = tri.b32g(self.z_s)
        bRg = tri.bRg(self.z_s)

        pim = self._pklin
        return pim*self.exp_cutoff

    def _get_pmm(self):
        """ Get the one-loop matter power spectrum.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()

        pk = self._pklin
        return pk*self.exp_cutoff

    def get_biased_pk2d(self, tracer1, *, tracer2=None, return_ia_bb=False,
                        extrap_order_lok=1, extrap_order_hik=2):
        """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
        the EFT power spectrum for two quantities defined by
        two :class:`~pyccl.nl_eft.tracers.EFTTracer` objects.

        Args:
            tracer1 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): the first
                tracer being correlated.
            tracer2 (:class:`~pyccl.nl_eft.tracers.EFTTracer`): the second
                tracer being correlated. If ``None``, the auto-correlation
                of the first tracer will be returned.
            return_ia_bb (:obj:`bool`): if ``True``, the B-mode power spectrum
                for intrinsic alignments will be returned (if both
                input tracers are of type
                :class:`~pyccl.nl_eft.tracers.EFTIntrinsicAlignmentTracer`)
                If ``False`` (default) E-mode power spectrum is returned.
            extrap_order_lok (:obj:`int`): extrapolation order to be used on
                k-values below the minimum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.
            extrap_order_hik (:obj:`int`): extrapolation order to be used on
                k-values above the maximum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.

        Returns:
            :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        """
        if return_ia_bb:
            return_ia_bb = True

        if tracer2 is None:
            tracer2 = tracer1

        t1 = tracer1.type
        t2 = tracer2.type

        if ((t1 == 'NC') or (t2 == 'NC')) and (not self.with_NC):
            raise ValueError("Can't use number counts tracer in "
                             "EulerianPTCalculator with 'with_NC=False'")
        if ((t1 == 'IA') or (t2 == 'IA')) and (not self.with_IA):
            raise ValueError("Can't use intrinsic alignment tracer in "
                             "EulerianPTCalculator with 'with_IA=False'")

        if t1 == 'NC':
            if t2 == 'NC':
                pk = self._get_pgg(tracer1, tracer2)
            elif t2 == 'IA':
                pk = self._get_pgi(tracer1, tracer2)
            else:  # Must be matter
                pk = self._get_pgm(tracer1)
        elif t1 == 'IA':
            if t2 == 'NC':
                pk = self._get_pgi(tracer2, tracer1)
            elif t2 == 'IA':
                pk = self._get_pii(tracer1, tracer2,
                                   return_bb=return_ia_bb)
            else:  # Must be matter
                pk = self._get_pim(tracer1)
        else:  # Must be matter
            if t2 == 'NC':
                pk = self._get_pgm(tracer2)
            elif t2 == 'IA':
                pk = self._get_pim(tracer2)
            else:  # Must be matter
                pk = self._get_pmm()

        pk2d = Pk2D(a_arr=self.a_s,
                    lk_arr=np.log(self.k_s),
                    pk_arr=pk,
                    is_logp=False,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik)
        return pk2d
