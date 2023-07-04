__all__ = ("LagrangianPTCalculator",)

import warnings

import numpy as np

from .. import (CCLAutoRepr, CCLError, CCLWarning, Pk2D,
                get_pk_spline_a, unlock_instance)


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


class LagrangianPTCalculator(CCLAutoRepr):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    Lagrangian perturbation theory correlations. These calculations
    are currently based on velocileptors
    (https://github.com/sfschen/velocileptors).

    In the parametrisation used here, the galaxy overdensity
    is expanded as:

    .. math::
        \\delta_g=b_1\\,\\delta+\\frac{b_2}{2}\\delta^2+
        \\frac{b_s}{2}s^2+\\frac{b_{3nl}}{2}O_{3}+
        \\frac{b_{k2}}{2}\\nabla^2\\delta.

    where

    .. math::
        O_{3}(k) = s_{ij}(k)t_{ij}(k) + \\frac{16}{63}\\langle \\delta_{lin}\\rangle

    .. note:: Only the leading-order non-local term (i.e.
              :math:`\\langle \\delta\\,\\nabla^2\\delta`) is
              taken into account in the expansion. All others are
              set to zero.

    .. note:: Terms of the form
              :math:`\\langle \\delta^2 \\psi_{nl}\\rangle` (and
              likewise for :math:`s^2`) are set to zero.

    .. note:: This calculator does not account for any form of
              stochastic bias contribution to the power spectra.
              If necessary, consider adding it in post-processing.

    Args:
        with_NC (:obj:`bool`): set to ``True`` if you'll want to use
            this calculator to compute correlations involving
            number counts.
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
    """
    __repr_attrs__ = __eq_attrs__ = ('with_NC', 'with_matter_1loop',
                                     'k_s', 'a_s', 'exp_cutoff', 'b1_pk_kind',
                                     'bk2_pk_kind', )

    def __init__(self, *, with_NC=False,
                 with_matter_1loop=True, cosmo=None,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 a_arr=None, k_cutoff=None, n_exp_cutoff=4,
                 b1_pk_kind='nonlinear', bk2_pk_kind='nonlinear'):
        self.with_matter_1loop = with_matter_1loop
        self.with_NC = with_NC

        to_do = ['one_loop_dd']
        if self.with_NC:
            to_do.append('dd_bias')

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

        from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
        h = cosmo['h']
        cleft = RKECLEFT(self.k_s / h, pklz0 * h ** 3)
        lpt_table = []
        for gz in g:
            cleft.make_ptable(D=gz, kmin=self.k_s[0] / h,
                              kmax=self.k_s[-1] / h, nk=self.k_s.size)
            lpt_table.append(cleft.pktable)
        lpt_table = np.array(lpt_table)
        lpt_table[:, :, 1:] /= h ** 3

        # Galaxy clustering templates
        # TODO: What to do here?
        if self.with_NC:
            self.lpt_table = lpt_table
            self.one_loop_dd = self.lpt_table[:, :, 1]
            self.with_matter_1loop = True
        elif self.with_matter_1loop:  # Only 1-loop matter needed
            self.one_loop_dd = lpt_table[:, :, 1]

        # b1/bk power spectrum
        pks = {}
        if 'nonlinear' in [self.b1_pk_kind, self.bk2_pk_kind]:
            pks['nonlinear'] = np.array([cosmo.nonlin_matter_power(self.k_s, a)
                                         for a in self.a_s])
        if 'linear' in [self.b1_pk_kind, self.bk2_pk_kind]:
            pks['linear'] = np.array([cosmo.linear_matter_power(self.k_s, a)
                                      for a in self.a_s])
        if 'pt' in [self.b1_pk_kind, self.bk2_pk_kind]:
            pks['pt'] = None
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

        # Transform from Eulerian to Lagrangian biases
        bL11 = b11 - 1
        bL12 = b12 - 1
        # Get Pk templates
        if self.pk_b1 is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            Pd1d1 = self.lpt_table[:, :, 3]
            pgg = (Pdmdm + (bL11+bL12)[:, None] * Pdmd1 +
                   (bL11*bL12)[:, None] * Pd1d1)
        else:
            pgg = (b11*b12)[:, None]*self.pk_b1
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pd1d2 = 0.5*self.lpt_table[:, :, 5]
        Pd2d2 = self.lpt_table[:, :, 6]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pd1s2 = 0.25*self.lpt_table[:, :, 8]
        Pd2s2 = 0.25*self.lpt_table[:, :, 9]
        Ps2s2 = 0.25*self.lpt_table[:, :, 10]
        Pdmo3 = 0.25 * self.lpt_table[:, :, 11]
        Pd1o3 = 0.25 * self.lpt_table[:, :, 12]
        if self.pk_bk is not None:
            Pd1k2 = 0.5*self.pk_bk * (self.k_s**2)[None, :]
        else:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]

            Pdmk2 = 0.5*Pdmdm * (self.k_s**2)[None, :]
            Pd1k2 = 0.5*Pdmd1 * (self.k_s**2)[None, :]
            Pd2k2 = Pdmd2 * (self.k_s**2)[None, :]
            Ps2k2 = Pdms2 * (self.k_s**2)[None, :]
            Pk2k2 = 0.25*Pdmdm * (self.k_s**4)[None, :]

        pgg += ((b21 + b22)[:, None] * Pdmd2 +
                (bs1 + bs2)[:, None] * Pdms2 +
                (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
                (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
                (b21*b22)[:, None] * Pd2d2 +
                (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
                (bs1*bs2)[:, None] * Ps2s2 +
                (b3nl1 + b3nl2)[:, None] * Pdmo3 +
                (bL11*b3nl2 + bL12*b3nl1)[:, None] * Pd1o3)

        if self.pk_bk is not None:
            pgg += (b12*bk21+b11*bk22)[:, None] * Pd1k2
        else:
            pgg += ((bk21 + bk22)[:, None] * Pdmk2 +
                    (bL12 * bk21 + bL11 * bk22)[:, None] * Pd1k2 +
                    (b22 * bk21 + b21 * bk22)[:, None] * Pd2k2 +
                    (bs2 * bk21 + bs1 * bk22)[:, None] * Ps2k2 +
                    (bk21 * bk22)[:, None] * Pk2k2)

        return pgg*self.exp_cutoff

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
        # Get biases
        b1 = trg.b1(self.z_s)
        b2 = trg.b2(self.z_s)
        bs = trg.bs(self.z_s)
        bk2 = trg.bk2(self.z_s)
        b3nl = trg.b3nl(self.z_s)

        # Compute Lagrangian bias
        bL1 = b1 - 1
        # Get Pk templates
        if self.pk_b1 is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            pgm = Pdmdm + bL1[:, None] * Pdmd1
        else:
            pgm = b1[:, None]*self.self.pk_b1
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pdmo3 = 0.25 * self.lpt_table[:, :, 11]

        if self.pk_bk is not None:
            Pdmk2 = 0.5*self.pk_bk * (self.k_s**2)[None, :]
        else:
            Pdmdm = self.lpt_table[:, :, 1]

            Pdmk2 = 0.5*Pdmdm * (self.k_s**2)[None, :]

        pgm += (b2[:, None] * Pdmd2 +
                bs[:, None] * Pdms2 +
                b3nl[:, None] * Pdmo3 +
                bk2[:, None] * Pdmk2)

        return pgm*self.exp_cutoff

    def _get_pmm(self):
        """ Get the one-loop matter power spectrum.

        Returns:
            array: 2D array of shape `(N_a, N_k)`, where `N_k` \
                is the size of this object's `k_s` attribute, and \
                `N_a` is the size of the object's `a_s` attribute.
        """
        self._check_init()

        pk = self.lpt_table[:, :, 1]

        return pk*self.exp_cutoff

    def get_biased_pk2d(self, tracer1, *, tracer2=None,
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
            extrap_order_lok (:obj:`int`): extrapolation order to be used on
                k-values below the minimum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.
            extrap_order_hik (:obj:`int`): extrapolation order to be used on
                k-values above the maximum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.

        Returns:
            :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        """
        if tracer2 is None:
            tracer2 = tracer1

        t1 = tracer1.type
        t2 = tracer2.type

        if ((t1 == 'NC') or (t2 == 'NC')) and (not self.with_NC):
            raise ValueError("Can't use number counts tracer in "
                             "LagrangianPTCalculator with 'with_NC=False'")

        if t1 == 'NC':
            if t2 == 'NC':
                pk = self._get_pgg(tracer1, tracer2)
            else:  # Must be matter
                pk = self._get_pgm(tracer1)
        else:  # Must be matter
            if t2 == 'NC':
                pk = self._get_pgm(tracer2)
            else:  # Must be matter
                pk = self._get_pmm()

        pk2d = Pk2D(a_arr=self.a_s,
                    lk_arr=np.log(self.k_s),
                    pk_arr=pk,
                    is_logp=False,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik)
        return pk2d

    # TODO: adapt to LPT
    def get_pk2d_template(self, kind, *, extrap_order_lok=1,
                          extrap_order_hik=2):
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
        :math:`\\nabla^2 \\delta` term in galaxy bias expansion)

        Args:
            kind (:obj:`str`): string defining the pair of PT operators for
                which we want the power spectrum.
            extrap_order_lok (:obj:`int`): extrapolation order to be used on
                k-values below the minimum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.
            extrap_order_hik (:obj:`int`): extrapolation order to be used on
                k-values above the maximum of the splines. See
                :class:`~pyccl.pk2d.Pk2D`.

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

        # If already built, return
        if pk_name in self._pk2d_temp:
            return self._pk2d_temp[pk_name]

        # Construct power spectrum array
        s4 = 0.
        if pk_name == 'm:m':
            pk = self._get_pmm()
        elif pk_name == 'm:b2':
            pk = 0.5*self._g4T*self.dd_bias[2]
        elif pk_name == 'm:b3nl':
            pk = 0.5*self._g4T*self.dd_bias[8]
        elif pk_name == 'm:bs':
            pk = 0.5*self._g4T*self.dd_bias[4]
        elif pk_name == 'm:bk2':
            pk = 0.5*self.pk_bk*(self.k_s**2)
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
