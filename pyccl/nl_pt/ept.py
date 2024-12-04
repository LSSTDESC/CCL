__all__ = ("EulerianPTCalculator",)

import numpy as np

from .. import (CCLAutoRepr, CCLError, CCLWarning, Pk2D,
                get_pk_spline_a, unlock_instance, warnings)


# All valid Pk pair labels and their aliases
_PK_ALIAS = {
    'm:m': 'm:m', 'm:b1': 'm:m', 'm:b2': 'm:b2',
    'm:b3nl': 'm:b3nl', 'm:bs': 'm:bs', 'm:bk2': 'm:bk2',
    'm:c1': 'm:m', 'm:c2': 'm:c2', 'm:cdelta': 'm:cdelta',
    'b1:b1': 'm:m', 'b1:b2': 'm:b2', 'b1:b3nl': 'm:b3nl',
    'b1:bs': 'm:bs', 'b1:bk2': 'm:bk2', 'b1:c1': 'm:m',
    'b1:c2': 'm:c2', 'b1:cdelta': 'm:cdelta', 'b2:b2': 'b2:b2',
    'b2:b3nl': 'zero', 'b2:bs': 'b2:bs', 'b2:bk2': 'zero',
    'b2:c1': 'zero', 'b2:c2': 'zero', 'b2:cdelta': 'zero',
    'b3nl:b3nl': 'zero', 'b3nl:bs': 'zero',
    'b3nl:bk2': 'zero', 'b3nl:c1': 'zero', 'b3nl:c2':
    'zero', 'b3nl:cdelta': 'zero', 'bs:bs': 'bs:bs',
    'bs:bk2': 'zero', 'bs:c1': 'zero', 'bs:c2': 'zero',
    'bs:cdelta': 'zero', 'bk2:bk2': 'zero', 'bk2:c1': 'zero',
    'bk2:c2': 'zero', 'bk2:cdelta': 'zero', 'c1:c1': 'm:m',
    'c1:c2': 'm:c2', 'c1:cdelta': 'm:cdelta', 'c2:c2': 'c2:c2',
    'c2:cdelta': 'c2:cdelta', 'cdelta:cdelta': 'cdelta:cdelta'}


class EulerianPTCalculator(CCLAutoRepr):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    Eulerian perturbation theory correlations. These calculations
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
            calculate perturbation theory quantities.
        log10k_max (:obj:`float`): decimal logarithm of the maximum
            Fourier scale (in :math:`{\\rm Mpc}^{-1}`) for which you want to
            calculate perturbation theory quantities.
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
        b1_pk_kind (:obj:`str`): power spectrum to use for the first-order
            bias terms in the expansion. ``'linear'``: use the linear
            matter power spectrum. ``'nonlinear'``: use the non-linear
            matter power spectrum. ``'pt'``: use the 1-loop SPT matter
            power spectrum.
        bk2_pk_kind (:obj:`str`): power spectrum to use for the non-local
            bias terms in the expansion. Same options and default as
            ``b1_pk_kind``.
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
             contribution to some of the terms will be subtracted.
    """
    __repr_attrs__ = __eq_attrs__ = ('with_NC', 'with_IA', 'with_matter_1loop',
                                     'k_s', 'a_s', 'exp_cutoff', 'b1_pk_kind',
                                     'bk2_pk_kind', 'fastpt_par', )

    def __init__(self, *, with_NC=False, with_IA=False,
                 with_matter_1loop=True, cosmo=None,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 a_arr=None, k_cutoff=None, n_exp_cutoff=4,
                 b1_pk_kind='nonlinear', bk2_pk_kind='nonlinear',
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

        to_do = ['one_loop_dd']
        if self.with_NC:
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

        # b1/bk P(k) prescription
        if b1_pk_kind not in ['linear', 'nonlinear', 'pt']:
            raise ValueError(f"Unknown P(k) prescription {b1_pk_kind}")
        if bk2_pk_kind not in ['linear', 'nonlinear', 'pt']:
            raise ValueError(f"Unknown P(k) prescription {bk2_pk_kind}")
        self.b1_pk_kind = b1_pk_kind
        self.bk2_pk_kind = bk2_pk_kind
        if (self.b1_pk_kind == 'pt') or (self.bk2_pk_kind == 'pt'):
            self.with_matter_1loop = True

        # Initialize all expensive arrays to ``None``.
        self._cosmo = None

        # Fill them out if cosmo is present
        if cosmo is not None:
            self.update_ingredients(cosmo)

        # All valid Pk pair labels
        self._pk_valid = list(_PK_ALIAS.keys())
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
        return hasattr(self, "pk_bk")

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

        def reshape_fastpt(tupl):
            for qq in tupl:
                if np.ndim(qq) > 0:
                    qq = qq[None, :]

        # Galaxy clustering templates
        if self.with_NC:
            self.dd_bias = self.pt.one_loop_dd_bias_b3nl(**kw)
            reshape_fastpt(self.dd_bias)
            self.one_loop_dd = self.dd_bias[0:1]
            self.with_matter_1loop = True
        elif self.with_matter_1loop:  # Only 1-loop matter needed
            self.one_loop_dd = self.pt.one_loop_dd(**kw)
            reshape_fastpt(self.one_loop_dd)

        # Intrinsic alignment templates
        if self.with_IA:
            self.ia_ta = self.pt.IA_ta(**kw)
            reshape_fastpt(self.ia_ta)
            self.ia_tt = self.pt.IA_tt(**kw)
            reshape_fastpt(self.ia_tt)
            self.ia_mix = self.pt.IA_mix(**kw)
            reshape_fastpt(self.ia_mix)

        # b1/bk power spectrum
        pks = {}
        if 'nonlinear' in [self.b1_pk_kind, self.bk2_pk_kind]:
            pks['nonlinear'] = np.array([cosmo.nonlin_matter_power(self.k_s, a)
                                         for a in self.a_s])
        if 'linear' in [self.b1_pk_kind, self.bk2_pk_kind]:
            pks['linear'] = np.array([cosmo.linear_matter_power(self.k_s, a)
                                      for a in self.a_s])
        if 'pt' in [self.b1_pk_kind, self.bk2_pk_kind]:
            if 'linear' in pks:
                pk = pks['linear']
            else:
                pk = np.array([cosmo.linear_matter_power(self.k_s, a)
                               for a in self.a_s])
            # Add SPT correction
            pk += self._g4T * self.one_loop_dd[0]
            pks['pt'] = pk
        self.pk_b1 = pks[self.b1_pk_kind]
        self.pk_bk = pks[self.bk2_pk_kind]

        # Reset template power spectra
        self._pk2d_temp = {}
        self._cosmo = cosmo

    def _get_pgg(self, tr1, tr2):
        """ Get the number counts auto-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            tr1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): first
                tracer to correlate.
            tr2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): first
                tracer to correlate.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates
        Pd1d1 = self.pk_b1
        Pd1d2 = self._g4T * self.dd_bias[2]
        Pd2d2 = self._g4T * self.dd_bias[3]
        Pd1s2 = self._g4T * self.dd_bias[4]
        Pd2s2 = self._g4T * self.dd_bias[5]
        Ps2s2 = self._g4T * self.dd_bias[6]
        Pd1p3 = self._g4T * self.dd_bias[8]
        Pd1k2 = self.pk_bk * (self.k_s**2)[None, :]

        # Get biases
        b11 = tr1.b1(self.z_s)
        b21 = tr1.b2(self.z_s)
        bs1 = tr1.bs(self.z_s)
        bk21 = tr1.bk2(self.z_s)
        b3nl1 = tr1.b3nl(self.z_s)
        b12 = tr2.b1(self.z_s)
        b22 = tr2.b2(self.z_s)
        bs2 = tr2.bs(self.z_s)
        bk22 = tr2.bk2(self.z_s)
        b3nl2 = tr2.b3nl(self.z_s)

        s4 = 0.
        if self.fastpt_par['sub_lowk']:
            s4 = self._g4T * self.dd_bias[7]

        pgg = ((b11*b12)[:, None] * Pd1d1 +
               0.5*(b11*b22 + b12*b21)[:, None] * Pd1d2 +
               0.25*(b21*b22)[:, None] * (Pd2d2 - 2.*s4) +
               0.5*(b11*bs2 + b12*bs1)[:, None] * Pd1s2 +
               0.25*(b21*bs2 + b22*bs1)[:, None] * (Pd2s2 - (4./3.)*s4) +
               0.25*(bs1*bs2)[:, None] * (Ps2s2 - (8./9.)*s4) +
               0.5*(b12*b3nl1+b11*b3nl2)[:, None] * Pd1p3 +
               0.5*(b12*bk21+b11*bk22)[:, None] * Pd1k2)

        return pgg*self.exp_cutoff

    def _get_pgi(self, trg, tri):
        """ Get the number counts - IA cross-spectrum at the internal
        set of wavenumbers and scale factors.

        .. note:: The full non-linear model for the cross-correlation
                  between number counts and intrinsic alignments is
                  still work in progress in FastPT. As a workaround
                  CCL assumes a non-linear treatment of IAs, but only
                  linearly biased number counts.

        Args:
            trg (:class:`~pyccl.nl_pt.tracers.PTTracer`): number
                counts tracer.
            tri (:class:`~pyccl.nl_pt.tracers.PTTracer`): intrinsic
                alignment tracer.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates
        Pd1d1 = self.pk_b1
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        # Get biases
        b1 = trg.b1(self.z_s)
        b2 = trg.b2(self.z_s)
        bs = trg.bs(self.z_s)
        bk2 = trg.bk2(self.z_s)
        b3nl = trg.b3nl(self.z_s)
        if any([b.any() for b in [b2, bs, bk2, b3nl]]):
            warnings.warn(
                "EulerianPTCalculators assume linear galaxy bias "
                "when computing galaxy-IA cross-correlations.",
                category=CCLWarning, importance='low')
        c1 = tri.c1(self.z_s)
        c2 = tri.c2(self.z_s)
        cd = tri.cdelta(self.z_s)

        pgi = b1[:, None] * (c1[:, None] * Pd1d1 +
                             (self._g4*cd)[:, None] * (a00e + c00e) +
                             (self._g4*c2)[:, None] * (a0e2 + b0e2))
        return pgi*self.exp_cutoff

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
        Pd1d1 = self.pk_b1
        Pd1d2 = self._g4T * self.dd_bias[2]
        Pd1s2 = self._g4T * self.dd_bias[4]
        Pd1p3 = self._g4T * self.dd_bias[8]
        Pd1k2 = self.pk_bk*(self.k_s**2)[None, :]

        # Get biases
        b1 = trg.b1(self.z_s)
        b2 = trg.b2(self.z_s)
        bs = trg.bs(self.z_s)
        bk2 = trg.bk2(self.z_s)
        b3nl = trg.b3nl(self.z_s)

        pgm = (b1[:, None] * Pd1d1 +
               0.5 * b2[:, None] * Pd1d2 +
               0.5 * bs[:, None] * Pd1s2 +
               0.5 * b3nl[:, None] * Pd1p3 +
               0.5 * bk2[:, None] * Pd1k2)

        return pgm*self.exp_cutoff

    def _get_pii(self, tr1, tr2, return_bb=False):
        """ Get the intrinsic alignment auto-spectrum at the internal
        set of wavenumbers and scale factors.

        Args:
            tr1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): first tracer
                to correlate.
            tr2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): first tracer
                to correlate.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        # Get Pk templates
        Pd1d1 = self.pk_b1
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        ae2e2, ab2b2 = self.ia_tt
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        # Get biases
        c11 = tr1.c1(self.z_s)
        c21 = tr1.c2(self.z_s)
        cd1 = tr1.cdelta(self.z_s)
        c12 = tr2.c1(self.z_s)
        c22 = tr2.c2(self.z_s)
        cd2 = tr2.cdelta(self.z_s)

        if return_bb:
            pii = ((cd1*cd2*self._g4)[:, None]*a0b0b +
                   (c21*c22*self._g4)[:, None]*ab2b2 +
                   ((cd1*c22+c21*cd2)*self._g4)[:, None] * d0bb2)
        else:
            pii = ((c11*c12)[:, None] * Pd1d1 +
                   ((c11*cd2+c12*cd1)*self._g4)[:, None]*(a00e+c00e) +
                   (cd1*cd2*self._g4)[:, None]*a0e0e +
                   (c21*c22*self._g4)[:, None]*ae2e2 +
                   ((c11*c22+c21*c12)*self._g4)[:, None]*(a0e2+b0e2) +
                   ((cd1*c22+cd2*c21)*self._g4)[:, None]*d0ee2)

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
        Pd1d1 = self.pk_b1
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        # Get biases
        c1 = tri.c1(self.z_s)
        c2 = tri.c2(self.z_s)
        cd = tri.cdelta(self.z_s)

        pim = (c1[:, None] * Pd1d1 +
               (self._g4*cd)[:, None] * (a00e + c00e) +
               (self._g4*c2)[:, None] * (a0e2 + b0e2))
        return pim*self.exp_cutoff

    def _get_pmm(self):
        """ Get the one-loop matter power spectrum.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()
        if self.b1_pk_kind == 'linear':
            P1loop = self._g4T * self.one_loop_dd[0]
        else:
            P1loop = 0.
        pk = self.pk_b1 + P1loop
        return pk*self.exp_cutoff

    def get_biased_pk2d(self, tracer1, *, tracer2=None, return_ia_bb=False,
                        extrap_order_lok=1, extrap_order_hik=2):
        """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
        the PT power spectrum for two quantities defined by
        two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

        .. note:: The full non-linear model for the cross-correlation
                  between number counts and intrinsic alignments is
                  still work in progress in FastPT. As a workaround
                  CCL assumes a non-linear treatment of IAs, but only
                  linearly biased number counts.

        Args:
            tracer1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the first
                tracer being correlated.
            tracer2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the second
                tracer being correlated. If ``None``, the auto-correlation
                of the first tracer will be returned.
            return_ia_bb (:obj:`bool`): if ``True``, the B-mode power spectrum
                for intrinsic alignments will be returned (if both
                input tracers are of type
                :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
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

    def get_pk2d_template(self, kind, *, extrap_order_lok=1,
                          extrap_order_hik=2, return_ia_bb=False):
        """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
        the power spectrum template for two of the PT operators. The
        combination returned is determined by ``kind``, which must be
        a string of the form ``'q1:q2'``, where ``q1`` and ``q2`` denote
        the two operators whose power spectrum is sought. Valid
        operator names are: ``'m'`` (matter overdensity), ``'b1'``
        (first-order overdensity), ``'b2'`` (:math:`\\delta^2`
        term in galaxy bias expansion), ``'bs'`` (:math:`s^2` term
        in galaxy bias expansion), ``'b3nl'`` (:math:`\\psi_{nl}`
        term in galaxy bias expansion), ``'bk2'`` (non-local
        :math:`\\nabla^2 \\delta` term in galaxy bias expansion),
        ``'c1'`` (linear IA term), ``'c2'`` (:math:`s^2` term in IA
        expansion), ``'cdelta'`` (:math:`s\\delta` term in IA expansion).

        Args:
            kind (:obj:`str`): string defining the pair of PT operators for
                which we want the power spectrum.
            extrap_order_lok (:obj:`int`): extrapolation order to be used on
                k-values below the minimum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.
            extrap_order_hik (:obj:`int`): extrapolation order to be used on
                k-values above the maximum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.
            return_ia_bb (:obj:`bool`): if ``True``, the B-mode power spectrum
                for intrinsic alignments will be returned (if both
                input tracers are of type
                :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
                If ``False`` (default) E-mode power spectrum is returned.

        Returns:
            :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        """
        if not (kind in _PK_ALIAS):
            # Reverse order and check again
            kind_reverse = ':'.join(kind.split(':')[::-1])
            if not (kind_reverse in _PK_ALIAS):
                raise ValueError(f"Pk template {kind} not valid")
            kind = kind_reverse
        pk_name = _PK_ALIAS[kind]

        if return_ia_bb and (pk_name in ['c2:c2', 'c2:cdelta',
                                         'cdelta:cdelta']):
            pk_name += '_bb'

        # If already built, return
        if pk_name in self._pk2d_temp:
            return self._pk2d_temp[pk_name]

        # Construct power spectrum array
        s4 = 0.
        if pk_name == 'm:m':
            pk = self.pk_b1
        elif pk_name == 'm:b2':
            pk = 0.5*self._g4T*self.dd_bias[2]
        elif pk_name == 'm:b3nl':
            pk = 0.5*self._g4T*self.dd_bias[8]
        elif pk_name == 'm:bs':
            pk = 0.5*self._g4T*self.dd_bias[4]
        elif pk_name == 'm:bk2':
            pk = 0.5*self.pk_bk*(self.k_s**2)
        elif pk_name == 'm:c2':
            pk = self._g4T * (self.ia_mix[0]+self.ia_mix[1])
        elif pk_name == 'm:cdelta':
            pk = self._g4T * (self.ia_ta[0]+self.ia_ta[1])
        elif pk_name == 'b2:b2':
            if self.fastpt_par['sub_lowk']:
                s4 = self.dd_bias[7]
            pk = 0.25*self._g4T*(self.dd_bias[3] - 2*s4)
        elif pk_name == 'b2:bs':
            if self.fastpt_par['sub_lowk']:
                s4 = self.dd_bias[7]
            pk = 0.25*self._g4T*(self.dd_bias[5] - 4*s4/3)
        elif pk_name == 'bs:bs':
            if self.fastpt_par['sub_lowk']:
                s4 = self.dd_bias[7]
            pk = 0.25*self._g4T*(self.dd_bias[6] - 8*s4/9)
        elif pk_name == 'c2:c2':
            pk = self._g4T * self.ia_tt[0]
        elif pk_name == 'c2:c2_bb':
            pk = self._g4T * self.ia_tt[1]
        elif pk_name == 'c2:cdelta':
            pk = self._g4T * self.ia_mix[2]
        elif pk_name == 'c2:cdelta_bb':
            pk = self._g4T * self.ia_mix[3]
        elif pk_name == 'cdelta:cdelta':
            pk = self._g4T * self.ia_ta[2]
        elif pk_name == 'cdelta:cdelta_bb':
            pk = self._g4T * self.ia_ta[3]
        elif pk_name == 'zero':
            # If zero, store None and return
            self._pk2d_temp[pk_name] = None
            return None

        # Build interpolator
        pk2d = Pk2D(a_arr=self.a_s,
                    lk_arr=np.log(self.k_s),
                    pk_arr=pk,
                    is_logp=False,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik)

        # Store and return
        self._pk2d_temp[pk_name] = pk2d
        return pk2d
