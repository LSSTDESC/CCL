__all__ = ("BaccoLbiasCalculator",)

import numpy as np

from .. import (CCLAutoRepr, CCLError, Pk2D,
                get_pk_spline_a, unlock_instance)


# All valid Pk pair labels and their aliases
# Note that bacco lbias has no b3nl, so terms
# with b3nl are automatically set to zero.
# TODO: make this common to all nl_pt models.
_PK_ALIAS = {
    'm:m': 'm:m', 'm:b1': 'm:b1', 'm:b2': 'm:b2',
    'm:b3nl': 'zero', 'm:bs': 'm:bs', 'm:bk2': 'm:bk2',
    'b1:b1': 'b1:b1', 'b1:b2': 'b1:b2', 'b1:b3nl': 'zero',
    'b1:bs': 'b1:bs', 'b1:bk2': 'b1:bk2', 'b2:b2': 'b2:b2',
    'b2:b3nl': 'zero', 'b2:bs': 'b2:bs', 'b2:bk2': 'b2:bk2',
    'b3nl:b3nl': 'zero', 'b3nl:bs': 'zero',
    'b3nl:bk2': 'zero', 'bs:bs': 'bs:bs',
    'bs:bk2': 'bs:bk2', 'bk2:bk2': 'bk2:bk2'}


class BaccoLbiasCalculator(CCLAutoRepr):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    hybrid Lagrangian bias expansion correlations using the
    emulator baccoemu
    (https://bitbucket.org/rangulo/baccoemu/src/master/baccoemu/).

    This is a hybrid model, featuring a second order
    Lagrangian bias expansion coupled with advecting the
    Lagrangian fields to Eulerian observables through
    N-body simulations. It has been tested to reproduce the
    galaxy-galaxy and galaxy-matter power spectra down
    to scales of 0.7 h/Mpc.

    In the parametrisation used here, the bias function
    in Lagrangian coordinates is expanded as (ignoring constant
    terms):

    .. math::
        w_{\\rm g}(\\boldsymbol{q})=1 + b_1\\,\\delta+(b_2/2)\\,\\delta^2+
        (b_s/2)\\,s^2+(b_{k2}/2)\\nabla^2\\delta.

    This translates to Eulerian space

    .. math::
        \\delta(\\boldsymbol{x}) = \\int \\mathrm{d}^3\\boldsymbol{q}
        w_{\\rm g}(\\boldsymbol{q}) \\delta^{\\rm D}(\\boldsymbol{x} -
        \\boldsymbol{q} - \\boldsymbol{\\Psi}),

    where the displacement field :math:`\\Psi` is obtained from
    simulations.

    .. note:: This calculator does not account for any form of
              stochastic bias contribution to the power spectra.
              If necessary, consider adding it in post-processing.

    Args:
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
    """
    __repr_attrs__ = __eq_attrs__ = ('k_s', 'a_s', 'exp_cutoff')

    def __init__(self, *, cosmo=None,
                 log10k_min=-4, log10k_max=-0.47, nk_per_decade=20,
                 a_arr=None, k_cutoff=None, n_exp_cutoff=4):
        # Load emulator
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import baccoemu
            self.emu = baccoemu.Lbias_expansion()

        # k sampling
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.k_s = np.logspace(log10k_min, log10k_max, nk_total)

        # a sampling
        if a_arr is None:
            a_arr = get_pk_spline_a().copy()
        mask = a_arr >= self.emu.emulator['nonlinear']['bounds'][-1][0]
        self.a_s = a_arr[mask]
        self.z_s = 1/self.a_s-1

        # Cutoff factor
        if k_cutoff is not None:
            self.exp_cutoff = np.exp(-(self.k_s/k_cutoff)**n_exp_cutoff)
            self.exp_cutoff = self.exp_cutoff[None, :]
        else:
            self.exp_cutoff = 1

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
        return hasattr(self, "lpt_table")

    @unlock_instance
    def update_ingredients(self, cosmo):
        """ Update the internal PT arrays.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        """
        if self.initialised and (cosmo == self._cosmo):
            return

        # hubble
        h = cosmo['h']

        # get bacco parameters
        emupars = dict(
            omega_cold=cosmo['Omega_c'] + cosmo['Omega_b'],
            omega_baryon=cosmo['Omega_b'],
            ns=cosmo['n_s'],
            hubble=cosmo['h'],
            neutrino_mass=np.sum(cosmo['m_nu']),
            w0=cosmo['w0'],
            wa=cosmo['wa'],
            expfactor=self.a_s
        )

        # if cosmo contains sigma8, we use it for baccoemu, otherwise we pass
        # A_s to the emulator
        if np.isnan(cosmo['A_s']):
            sigma8tot = cosmo['sigma8']
            sigma8cold = self._sigma8tot_2_sigma8cold(emupars, sigma8tot)
            emupars['sigma8_cold'] = sigma8cold
        else:
            emupars['A_s'] = cosmo['A_s']

        # call bacco
        k = self.k_s / h
        lpt_table = self.emu.get_nonlinear_pnn(k=k, **emupars)[1]
        lpt_table /= h ** 3

        # save templates in a table
        self.lpt_table = lpt_table

        # Reset template power spectra
        self._pk2d_temp = {}
        self._cosmo = cosmo

    def _sigma8tot_2_sigma8cold(self, emupars, sigma8tot):
        """Use baccoemu to convert sigma8 total matter to sigma8 cdm+baryons
        """
        if hasattr(emupars['omega_cold'], '__len__'):
            _emupars = {}
            for pname in emupars:
                _emupars[pname] = emupars[pname][0]
        else:
            _emupars = emupars
        A_s_fid = 2.1e-9
        sigma8tot_fid = self.emu.matter_powerspectrum_emulator.get_sigma8(
            cold=False, A_s=A_s_fid, **_emupars)
        A_s = (sigma8tot / sigma8tot_fid)**2 * A_s_fid
        return self.emu.matter_powerspectrum_emulator.get_sigma8(cold=True,
                                                                 A_s=A_s,
                                                                 **_emupars)

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
        b12 = tr2.b1(self.z_s)
        b22 = tr2.b2(self.z_s)
        bs2 = tr2.bs(self.z_s)
        bk22 = tr2.bk2(self.z_s)

        # Transform from Eulerian to Lagrangian biases
        bL11 = b11 - 1
        bL12 = b12 - 1
        # Get Pk templates
        Pdmdm = self.lpt_table[0, :, :]          # 1 1
        Pdmd1 = self.lpt_table[1, :, :]          # 1 d
        Pdmd2 = self.lpt_table[2, :, :] * 0.5    # 1 d2
        Pdms2 = self.lpt_table[3, :, :] * 0.5    # 1 s2
        Pdmk2 = self.lpt_table[4, :, :] * 0.5    # 1 lap
        Pd1d1 = self.lpt_table[5, :, :]          # d d
        Pd1d2 = self.lpt_table[6, :, :] * 0.5    # d d2
        Pd1s2 = self.lpt_table[7, :, :] * 0.5    # d s2
        Pd1k2 = self.lpt_table[8, :, :] * 0.5    # d k2
        Pd2d2 = self.lpt_table[9, :, :] * 0.25   # d2 d2
        Pd2s2 = self.lpt_table[10, :, :] * 0.25  # d2 s2
        Pd2k2 = self.lpt_table[11, :, :] * 0.25  # d2 k2
        Ps2s2 = self.lpt_table[12, :, :] * 0.25  # s2 s2
        Ps2k2 = self.lpt_table[13, :, :] * 0.25  # s2 k2
        Pk2k2 = self.lpt_table[14, :, :] * 0.25  # k2 k2

        pgg = (Pdmdm +
               (bL11+bL12)[:, None] * Pdmd1 +
               (bL11*bL12)[:, None] * Pd1d1 +
               (b21 + b22)[:, None] * Pdmd2 +
               (bs1 + bs2)[:, None] * Pdms2 +
               (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
               (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
               (b21*b22)[:, None] * Pd2d2 +
               (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
               (bs1*bs2)[:, None] * Ps2s2 +
               (bk21 + bk22)[:, None] * Pdmk2 +
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

        # Compute Lagrangian bias
        bL1 = b1 - 1
        # Get Pk templates
        Pdmdm = self.lpt_table[0, :, :]        # 1 1
        Pdmd1 = self.lpt_table[1, :, :]        # 1 d
        Pdmd2 = self.lpt_table[2, :, :] * 0.5  # 1 d2
        Pdms2 = self.lpt_table[3, :, :] * 0.5  # 1 s2
        Pdmk2 = self.lpt_table[4, :, :] * 0.5  # 1 k2

        pgm = (Pdmdm +
               bL1[:, None] * Pdmd1 +
               b2[:, None] * Pdmd2 +
               bs[:, None] * Pdms2 +
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

        pk = self.lpt_table[0, :, :]

        return pk*self.exp_cutoff

    def get_biased_pk2d(self, tracer1, *, tracer2=None,
                        extrap_order_lok=1, extrap_order_hik=2):
        """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
        the PT power spectrum for two quantities defined by
        two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

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

        if t1 == 'IA' or t2 == 'IA':
            raise ValueError("Intrinsic alignments not implemented in "
                             "BaccoLbiasCalculator.")

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
        in galaxy bias expansion), ``'bk2'`` (non-local
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

        self._check_init()

        if pk_name == 'm:m':
            pk = self._get_pmm()
        elif pk_name == 'm:b1':
            pk = self.lpt_table[1, :, :]
        elif pk_name == 'm:b2':
            pk = self.lpt_table[2, :, :] * 0.5
        elif pk_name == 'm:bs':
            pk = self.lpt_table[3, :, :] * 0.5
        elif pk_name == 'm:bk2':
            pk = self.lpt_table[4, :, :] * 0.5
        elif pk_name == 'b1:b1':
            pk = self.lpt_table[5, :, :]
        elif pk_name == 'b1:b2':
            pk = self.lpt_table[6, :, :] * 0.5
        elif pk_name == 'b1:bs':
            pk = self.lpt_table[7, :, :] * 0.5
        elif pk_name == 'b1:bk2':
            pk = self.lpt_table[8, :, :] * 0.5
        elif pk_name == 'b2:b2':
            pk = self.lpt_table[9, :, :] * 0.25
        elif pk_name == 'b2:bs':
            pk = self.lpt_table[10, :, :] * 0.25
        elif pk_name == 'b2:bk2':
            pk = self.lpt_table[11, :, :] * 0.25
        elif pk_name == 'bs:bs':
            pk = self.lpt_table[12, :, :] * 0.25
        elif pk_name == 'bs:bk2':
            pk = self.lpt_table[13, :, :] * 0.25
        elif pk_name == 'bk2:bk2':
            pk = self.lpt_table[14, :, :] * 0.25
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
