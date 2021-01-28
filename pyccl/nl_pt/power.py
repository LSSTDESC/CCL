import numpy as np
from .. import ccllib as lib
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
from ..background import growth_factor
from .tracers import PTTracer

try:
    import fastpt as fpt
    HAVE_FASTPT = True
except ImportError:
    HAVE_FASTPT = False


class PTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        with_NC (bool): set to True if you'll want to use
            this calculator to compute correlations involving
            number counts.
        with_IA(bool): set to True if you'll want to use
            this calculator to compute correlations involving
            intrinsic alignments.
        with_dd(bool): set to True if you'll want to use
            this calculator to compute the one-loop matter power
            spectrum.
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        pad_factor (float): fraction of the log(k) interval
            you want to add as padding for FFTLog calculations
            within FAST-PT.
        low_extrap (float): decimal logaritm of the minimum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        high_extrap (float): decimal logaritm of the maximum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        P_window (array_like or None): 2-element array describing
            the tapering window used by FAST-PT. See FAST-PT
            documentation for more details.
        C_window (float): `C_window` parameter used by FAST-PT
            to smooth the edges and avoid ringing. See FAST-PT
            documentation for more details.
    """
    def __init__(self, with_NC=False, with_IA=False, with_dd=True,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 pad_factor=1, low_extrap=-5, high_extrap=3,
                 P_window=None, C_window=.75):
        assert HAVE_FASTPT, (
            "You must have the `FAST-PT` python package "
            "installed to use CCL to get PT observables! "
            "You can install it with pip install fast-pt.")

        self.with_dd = with_dd
        self.with_NC = with_NC
        self.with_IA = with_IA
        self.P_window = P_window
        self.C_window = C_window

        to_do = ['one_loop_dd']
        if self.with_NC:
            to_do.append('dd_bias')
        if self.with_IA:
            to_do.append('IA')

        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        n_pad = int(pad_factor * len(self.ks))

        self.pt = fpt.FASTPT(self.ks, to_do=to_do,
                             low_extrap=low_extrap,
                             high_extrap=high_extrap,
                             n_pad=n_pad)
        self.one_loop_dd = None
        self.dd_bias = None
        self.ia_ta = None
        self.ia_tt = None
        self.ia_mix = None

    def update_pk(self, pk):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if self.with_NC:
            self._get_dd_bias(pk)
            self.with_dd = True
        elif self.with_dd:
            self._get_one_loop_dd(pk)
        if self.with_IA:
            self._get_ia_bias(pk)

    def _get_one_loop_dd(self, pk):
        # Precompute quantities needed for one-loop dd
        # power spectra. Only needed if dd_bias is not called.
        self.one_loop_dd = self.pt.one_loop_dd(pk,
                                               P_window=self.P_window,
                                               C_window=self.C_window)

    def _get_dd_bias(self, pk):
        # Precompute quantities needed for number counts
        # power spectra.
        self.dd_bias = self.pt.one_loop_dd_bias(pk,
                                                P_window=self.P_window,
                                                C_window=self.C_window)
        self.one_loop_dd = self.dd_bias[0:1]

    def _get_ia_bias(self, pk):
        # Precompute quantities needed for intrinsic alignment
        # power spectra.
        self.ia_ta = self.pt.IA_ta(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_tt = self.pt.IA_tt(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_mix = self.pt.IA_mix(pk,
                                     P_window=self.P_window,
                                     C_window=self.C_window)

    def get_pgg(self, Pd1d1, g4,
                b11, b21, bs1, b12, b22, bs2,
                sub_lowk):
        """ Get the number counts auto-spectrum at the internal
        set of wavenumbers (given by this object's `ks` attribute)
        and a number of redshift values.

        Args:
            Pd1d1 (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.
            b11 (array_like): 1-st order bias for the first tracer
                being correlated at the same set of input redshifts.
            b21 (array_like): 2-nd order bias for the first tracer
                being correlated at the same set of input redshifts.
            bs1 (array_like): tidal bias for the first tracer
                being correlated at the same set of input redshifts.
            b12 (array_like): 1-st order bias for the second tracer
                being correlated at the same set of input redshifts.
            b22 (array_like): 2-nd order bias for the second tracer
                being correlated at the same set of input redshifts.
            bs2 (array_like): tidal bias for the second tracer
                being correlated at the same set of input redshifts.
            sub_lowk (bool): if True, the small-scale white noise
                contribution will be subtracted.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        Pd1d2 = g4[None, :] * self.dd_bias[2][:, None]
        Pd2d2 = g4[None, :] * self.dd_bias[3][:, None]
        Pd1s2 = g4[None, :] * self.dd_bias[4][:, None]
        Pd2s2 = g4[None, :] * self.dd_bias[5][:, None]
        Ps2s2 = g4[None, :] * self.dd_bias[6][:, None]

        s4 = 0.
        if sub_lowk:
            s4 = g4 * self.dd_bias[7]
            s4 = s4[None, :]

        pgg = ((b11*b12)[None, :] * Pd1d1 +
               0.5*(b11*b22 + b12*b21)[None, :] * Pd1d2 +
               0.25*(b21*b22)[None, :] * (Pd2d2 - 2.*s4) +
               0.5*(b11*bs2 + b12*bs1)[None, :] * Pd1s2 +
               0.25*(b21*bs2 + b22*bs1)[None, :] * (Pd2s2 - (4./3.)*s4) +
               0.25*(bs1*bs2)[None, :] * (Ps2s2 - (8./9.)*s4))
        return pgg

    def get_pgi(self, Pd1d1, g4, b1, b2, bs, c1, c2, cd):
        """ Get the number counts - IA cross-spectrum at the
        internal set of wavenumbers (given by this object's
        `ks` attribute) and a number of redshift values.

        .. note:: The full non-linear model for the cross-correlation
                  between number counts and intrinsic alignments is
                  still work in progress in FastPT. As a workaround
                  CCL assumes a non-linear treatment of IAs, but only
                  linearly biased number counts.

        Args:
            Pd1d1 (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.
            b1 (array_like): 1-st order bias for the number counts
                being correlated at the same set of input redshifts.
            b2 (array_like): 2-nd order bias for the number counts
                being correlated at the same set of input redshifts.
            bs (array_like): tidal bias for the number counts
                being correlated at the same set of input redshifts.
            c1 (array_like): 1-st order bias for the IA tracer
                being correlated at the same set of input redshifts.
            c2 (array_like): 2-nd order bias for the IA tracer
                being correlated at the same set of input redshifts.
            cd (array_like): overdensity bias for the IA tracer
                being correlated at the same set of input redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        pgi = b1[None, :] * (c1[None, :] * Pd1d1 +
                             (g4*cd)[None, :] * (a00e + c00e)[:, None] +
                             (g4*c2)[None, :] * (a0e2 + b0e2)[:, None])
        return pgi

    def get_pgm(self, Pd1d1, g4, b1, b2, bs):
        """ Get the number counts - matter cross-spectrum at the
        internal set of wavenumbers (given by this object's `ks`
        attribute) and a number of redshift values.

        Args:
            Pd1d1 (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.
            b1 (array_like): 1-st order bias for the number counts
                tracer being correlated at the same set of input
                redshifts.
            b2 (array_like): 2-nd order bias for the number counts
                tracer being correlated at the same set of input
                redshifts.
            bs (array_like): tidal bias for the number counts
                tracer being correlated at the same set of input
                redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        Pd1d2 = g4[None, :] * self.dd_bias[2][:, None]
        Pd1s2 = g4[None, :] * self.dd_bias[4][:, None]

        pgm = (b1[None, :] * Pd1d1 +
               0.5 * b2[None, :] * Pd1d2 +
               0.5 * bs[None, :] * Pd1s2)
        return pgm

    def get_pii(self, Pd1d1, g4, c11, c21, cd1,
                c12, c22, cd2, return_bb=False,
                return_both=False):
        """ Get the intrinsic alignment auto-spectrum at the internal
        set of wavenumbers (given by this object's `ks` attribute)
        and a number of redshift values.

        Args:
            Pd1d1 (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.
            c11 (array_like): 1-st order bias for the first tracer
                being correlated at the same set of input redshifts.
            c21 (array_like): 2-nd order bias for the first tracer
                being correlated at the same set of input redshifts.
            cd1 (array_like): overdensity bias for the first tracer
                being correlated at the same set of input redshifts.
            c12 (array_like): 1-st order bias for the second tracer
                being correlated at the same set of input redshifts.
            c22 (array_like): 2-nd order bias for the second tracer
                being correlated at the same set of input redshifts.
            cd2 (array_like): overdensity bias for the second tracer
                being correlated at the same set of input redshifts.
            return_bb (bool): if `True`, the B-mode power spectrum
                will be returned.
            return_both (bool): if `True`, both the E- and B-mode
                power spectra will be returned. Supersedes `return_bb`.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        ae2e2, ab2b2 = self.ia_tt
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        if return_both:
            return_bb = True

        if return_bb:
            pii_bb = ((cd1*cd2*g4)[None, :] * a0b0b[:, None] +
                      (c21*c22*g4)[None, :] * ab2b2[:, None] +
                      ((cd1*c22 + c21*cd2)*g4)[None, :] * d0bb2[:, None])
            if not return_both:
                pii = pii_bb

        if (not return_bb) or return_both:
            pii = ((c11*c12)[None, :] * Pd1d1 +
                   ((c11*cd2 + c12*cd1)*g4)[None, :] * (a00e + c00e)[:, None] +
                   (cd1*cd2*g4)[None, :] * a0e0e[:, None] +
                   (c21*c22*g4)[None, :] * ae2e2[:, None] +
                   ((c11*c22 + c21*c12)*g4)[None, :] * (a0e2 + b0e2)[:, None] +
                   ((cd1*c22 + cd2*c21)*g4)[None, :] * d0ee2[:, None])

        if return_both:
            return pii, pii_bb
        else:
            return pii

    def get_pim(self, Pd1d1, g4, c1, c2, cd):
        """ Get the intrinsic alignment - matter cross-spectrum at
        the internal set of wavenumbers (given by this object's `ks`
        attribute) and a number of redshift values.

        Args:
            Pd1d1 (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.
            c1 (array_like): 1-st order bias for the IA
                tracer being correlated at the same set of input
                redshifts.
            c2 (array_like): 2-nd order bias for the IA
                tracer being correlated at the same set of input
                redshifts.
            cd (array_like): overdensity bias for the IA
                tracer being correlated at the same set of input
                redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix

        pim = (c1[None, :] * Pd1d1 +
               (g4*cd)[None, :] * (a00e + c00e)[:, None] +
               (g4*c2)[None, :] * (a0e2 + b0e2)[:, None])
        return pim

    def get_pmm(self, Pd1d1_lin, g4):
        """ Get the one-loop matter power spectrum.

        Args:
            Pd1d1_lin (array_like): 1-loop linear matter power spectrum
                at the wavenumber values given by this object's
                `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        P1loop = g4[None, :] * self.one_loop_dd[0][:, None]
        pmm = (Pd1d1_lin + P1loop)
        return pmm


def get_pt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                sub_lowk=False, nonlin_pk_type='nonlinear',
                a_arr=None, extrap_order_lok=1, extrap_order_hik=2,
                return_ia_bb=False, return_ia_ee_and_bb=False,
                return_ptc=False):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

    .. note:: The full non-linear model for the cross-correlation
              between number counts and intrinsic alignments is
              still work in progress in FastPT. As a workaround
              CCL assumes a non-linear treatment of IAs, but only
              linearly biased number counts.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        tracer1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the first
            tracer being correlated.
        ptc (:class:`PTCalculator`): a perturbation theory
            calculator.
        tracer2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the second
            tracer being correlated. If `None`, the auto-correlation
            of the first tracer will be returned.
        sub_lowk (bool): if True, the small-scale white noise
            contribution will be subtracted for number counts
            auto-correlations.
        nonlin_pk_type (str): type of 1-loop matter power spectrum
            to use. 'linear' for linear P(k), 'nonlinear' for the internal
            non-linear power spectrum, 'spt' for standard perturbation
            theory power spectrum. Default: 'nonlinear'.
        a_arr (array): an array holding values of the scale factor
            at which the power spectrum should be calculated for
            interpolation. If `None`, the internal values used by
            `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        return_ia_bb (bool): if `True`, the B-mode power spectrum
            for intrinsic alignments will be returned (if both
            input tracers are of type
            :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
            If `False` (default) E-mode power spectrum is returned.
        return_ia_ee_and_bb (bool): if `True`, the E-mode power spectrum
            for intrinsic alignments will be returned in addition to
            the B-mode one (if both input tracers are of type
            :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
            If `False` (default) E-mode power spectrum is returned.
            Supersedes `return_ia_bb`.
        return_ptc (bool): if `True`, the fastpt object used as the PT
            calculator (ptc) will also be returned. This feature may
            be useful if an input ptc is not specified and one is
            initialized when this function is called. If `False` (default)
            the ptc is not output, whether or not it is initialized as
            part of the function call.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        :class:`~pyccl.nl_pt.power.PTCalculator`: PT Calc [optional]
    """
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status)

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, PTTracer):
        raise TypeError("tracer1 must be of type `PTTracer`")
    if not isinstance(tracer2, PTTracer):
        raise TypeError("tracer2 must be of type `PTTracer`")

    if ptc is None:
        with_NC = ((tracer1.type == 'NC')
                   or (tracer2.type == 'NC'))
        with_IA = ((tracer1.type == 'IA')
                   or (tracer2.type == 'IA'))
        with_dd = nonlin_pk_type == 'spt'
        ptc = PTCalculator(with_dd=with_dd,
                           with_NC=with_NC,
                           with_IA=with_IA)
    if not isinstance(ptc, PTCalculator):
        raise TypeError("ptc should be of type `PTCalculator`")

    if (tracer1.type == 'NC') or (tracer2.type == 'NC'):
        if not ptc.with_NC:
            raise ValueError("Need number counts bias, "
                             "but calculator didn't compute it")
    if (tracer1.type == 'IA') or (tracer2.type == 'IA'):
        if not ptc.with_IA:
            raise ValueError("Need intrinsic alignment bias, "
                             "but calculator didn't compute it")
    if nonlin_pk_type == 'spt':
        if not ptc.with_dd:
            raise ValueError("Need 1-loop matter power spectrum, "
                             "but calculator didn't compute it")

    if return_ia_ee_and_bb:
        return_ia_bb = True

    # z
    z_arr = 1. / a_arr - 1
    # P_lin(k) at z=0
    pk_lin_z0 = linear_matter_power(cosmo, ptc.ks, 1.)

    # Linear growth factor
    ga = growth_factor(cosmo, a_arr)
    ga4 = ga**4

    # update the PTC to have the require Pk components
    ptc.update_pk(pk_lin_z0)

    if nonlin_pk_type == 'nonlinear':
        Pd1d1 = np.array([nonlin_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    elif nonlin_pk_type == 'linear':
        Pd1d1 = np.array([linear_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    elif nonlin_pk_type == 'spt':
        pklin = np.array([linear_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
        Pd1d1 = ptc.get_pmm(pklin, ga4)
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            p_pt = ptc.get_pgg(Pd1d1, ga4,
                               b11, b21, bs1, b12, b22, bs2,
                               sub_lowk)
        elif (tracer2.type == 'IA'):
            c12 = tracer2.c1(z_arr)
            c22 = tracer2.c2(z_arr)
            cd2 = tracer2.cdelta(z_arr)
            p_pt = ptc.get_pgi(Pd1d1, ga4,
                               b11, b21, bs1, c12, c22, cd2)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(Pd1d1, ga4,
                               b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'IA'):
        c11 = tracer1.c1(z_arr)
        c21 = tracer1.c2(z_arr)
        cd1 = tracer1.cdelta(z_arr)
        if (tracer2.type == 'IA'):
            c12 = tracer2.c1(z_arr)
            c22 = tracer2.c2(z_arr)
            cd2 = tracer2.cdelta(z_arr)
            p_pt = ptc.get_pii(Pd1d1, ga4,
                               c11, c21, cd1, c12, c22, cd2,
                               return_bb=return_ia_bb,
                               return_both=return_ia_ee_and_bb)
        elif (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgi(Pd1d1, ga4,
                               b12, b22, bs2, c11, c21, cd1)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pim(Pd1d1, ga4,
                               c11, c21, cd1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgm(Pd1d1, ga4,
                               b12, b22, bs2)
        elif (tracer2.type == 'IA'):
            c12 = tracer2.c1(z_arr)
            c22 = tracer2.c2(z_arr)
            cd2 = tracer2.cdelta(z_arr)
            p_pt = ptc.get_pim(Pd1d1, ga4,
                               c12, c22, cd2)
        elif (tracer2.type == 'M'):
            p_pt = Pd1d1
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    if return_ia_ee_and_bb:
        pt_pk_ee = Pk2D(a_arr=a_arr,
                        lk_arr=np.log(ptc.ks),
                        pk_arr=p_pt[0].T,
                        is_logp=False)
        pt_pk_bb = Pk2D(a_arr=a_arr,
                        lk_arr=np.log(ptc.ks),
                        pk_arr=p_pt[1].T,
                        is_logp=False)
        if return_ptc:
            return pt_pk_ee, pt_pk_bb, ptc
        else:
            return pt_pk_ee, pt_pk_bb
    else:
        pt_pk = Pk2D(a_arr=a_arr,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt.T,
                     is_logp=False)
        if return_ptc:
            return pt_pk, ptc
        else:
            return pt_pk
