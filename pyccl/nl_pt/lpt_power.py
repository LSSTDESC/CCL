import numpy as np
from .. import ccllib as lib
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
from ..background import growth_factor
from .tracers import PTTracer

try:
    from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
    HAVE_CLEFT = True
except ImportError:
    HAVE_CLEFT = False


class LPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    Lagrangian perturbation theory bias expansions. These calculations
    are currently based on velocileptors
    (https://github.com/sfschen/velocileptors).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        nk_per_decade (int): number of wavenumbers per decade
            to sample.
        a_arr (array): an array holding values of the scale factor
            at which the power spectrum should be calculated for
            interpolation. If `None`, the internal values used by
            `cosmo` will be used.
    """
    def __init__(self, cosmo, log10k_min=-4, log10k_max=2,
                 nk_per_decade=20, a_arr=None):
        assert HAVE_CLEFT, (
            "You must have the `velocileptors` python package "
            "installed to use CCL to get LPT observables!")

        # k sampling
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)

        # a sampling
        if a_arr is None:
            status = 0
            na = lib.get_pk_spline_na(cosmo.cosmo)
            a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
            check(status)
        self.a_s = a_arr.copy()

        # Pk and growth arrays
        pk = linear_matter_power(cosmo, self.ks, 1.)
        Dz = growth_factor(cosmo, self.a_s)

        # Create LPT arrays. This is the slow part.
        h = cosmo['h']
        cleft = RKECLEFT(self.ks/h, pk*h**3)
        self.lpt_table = []
        for D in Dz:
            cleft.make_ptable(D=D, kmin=self.ks[0]/h,
                              kmax=self.ks[-1]/h, nk=self.ks.size)
            self.lpt_table.append(cleft.pktable)
        self.lpt_table = np.array(self.lpt_table)
        self.lpt_table /= h**3

    def get_pgg(self, Pnl, b11, b21, bs1, b12, b22, bs2):
        """ Get the number counts auto-spectrum at the internal
        set of wavenumbers (given by this object's `ks` attribute)
        and a number of redshift values.

        Args:
            Pnl (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
                if `None`, the "1-1", "1-b1" and "b1-b1" terms will
                be computed separately using LPT. Otherwise, those
                terms will be combined together into a single term
                of the form `b11*b12*Pnl`.
            b11 (array_like): 1-st order bias for the first tracer
                being correlated at the same set of input redshifts.
                Note that this is meant to be the first-order
                **Eulerian** bias (i.e. the Lagrangian bias plus 1).
            b21 (array_like): 2-nd order bias for the first tracer
                being correlated at the same set of input redshifts.
            bs1 (array_like): tidal bias for the first tracer
                being correlated at the same set of input redshifts.
            b12 (array_like): 1-st order bias for the second tracer
                being correlated at the same set of input redshifts.
                Note that this is meant to be the first-order
                **Eulerian** bias (i.e. the Lagrangian bias plus 1).
            b22 (array_like): 2-nd order bias for the second tracer
                being correlated at the same set of input redshifts.
            bs2 (array_like): tidal bias for the second tracer
                being correlated at the same set of input redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the array of scale factors used \
                to initialize this object.
        """
        # Clarification:
        # CLEFT uses the following expansion for the galaxy overdensity:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2
        # (see Eq. 4.4 of https://arxiv.org/pdf/2005.00523.pdf).
        # To add to the confusion, this is different from the prescription
        # used by EPT, where s^2 is divided by 2 :-|
        #
        # The LPT table below contains the following power spectra
        # in order:
        #  <1,1>
        #  2*<1,d>
        #  <d,d>
        #  2*<1,d^2/2>
        #  2*<d,d^2/2>
        #  <d^2/2,d^2/2> (!)
        #  2*<1,s^2>
        #  2*<d,s^2>
        #  2*<d^2/2,s^2> (!)
        #  <s^2,s^2> (!)
        #
        # So:
        #   a) The cross-correlations need to be divided by 2.
        #   b) The spectra involving b2 are for d^2/2, NOT d^2!!
        #   c) The spectra invoving bs are for s^2, NOT s^2/2!!
        #
        # Importantly, we have corrected the spectra involving s2 to
        # make the definition of bs equivalent in the EPT and LPT
        # expansions.
        bL11 = b11-1
        bL12 = b12-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            Pd1d1 = self.lpt_table[:, :, 3]
            pgg = (Pdmdm + (bL11+bL12)[:, None] * Pdmd1 +
                   (bL11*bL12)[:, None] * Pd1d1)
        else:
            pgg = (b11*b12)[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pd1d2 = 0.5*self.lpt_table[:, :, 5]
        Pd2d2 = self.lpt_table[:, :, 6]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pd1s2 = 0.25*self.lpt_table[:, :, 8]
        Pd2s2 = 0.25*self.lpt_table[:, :, 9]
        Ps2s2 = 0.25*self.lpt_table[:, :, 10]

        pgg += ((b21 + b22)[:, None] * Pdmd2 +
                (bs1 + bs2)[:, None] * Pdms2 +
                (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
                (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
                (b21*b22)[:, None] * Pd2d2 +
                (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
                (bs1*bs2)[:, None] * Ps2s2)
        return pgg

    def get_pgm(self, Pnl, b1, b2, bs):
        """ Get the number counts - matter cross-spectrum at the
        internal set of wavenumbers (given by this object's `ks`
        attribute) and a number of redshift values.

        Args:
            Pnl (array_like): 1-loop matter power spectrum at the
                wavenumber values given by this object's `ks` list.
                if `None`, the "1-1" and "1-b1" terms will
                be computed separately using LPT. Otherwise, those
                terms will be combined together into a single term
                of the form `b1*Pnl`.
            b1 (array_like): 1-st order bias for the number counts
                tracer being correlated at the same set of input
                redshifts. Note that this is meant to be the
                first-order **Eulerian** bias (i.e. the Lagrangian
                bias plus 1).
            b2 (array_like): 2-nd order bias for the number counts
                tracer being correlated at the same set of input
                redshifts.
            bs (array_like): tidal bias for the number counts
                tracer being correlated at the same set of input
                redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the array of scale factors used \
                to initialize this object.
        """
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        bL1 = b1-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            pgm = Pdmdm + bL1[:, None] * Pdmd1
        else:
            pgm = b1[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]

        pgm += (b2[:, None] * Pdmd2 +
                bs[:, None] * Pdms2)
        return pgm


def get_lpt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                 nonlin_pk_type='nonlinear', a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2,
                 return_ptc=False):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the LPT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

    .. note:: The current implementation only allows for
              correlations between number-counts and matter
              tracers.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        tracer1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the first
            tracer being correlated.
        tracer2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the second
            tracer being correlated. If `None`, the auto-correlation
            of the first tracer will be returned.
        ptc (:class:`LPTCalculator`): a perturbation theory
            calculator.
        nonlin_pk_type (str): type of 1-loop matter power spectrum
            to use. 'linear' for linear P(k), 'nonlinear' for the internal
            non-linear power spectrum, 'lpt' for Lagrangian perturbation
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
        return_ptc (bool): if `True`, the LPT calculator object used
            (`ptc`) will also be returned. This feature may
            be useful if an input `ptc` is not specified and one is
            initialized when this function is called. If `False` (default)
            the `ptc` is not output, whether or not it is initialized as
            part of the function call.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        :class:`~pyccl.nl_pt.power.PTCalculator`: PT Calc [optional]
    """

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, ccl.nl_pt.PTTracer):
        raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    if not isinstance(tracer2, ccl.nl_pt.PTTracer):
        raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    if not isinstance(ptc, LPTCalculator):
        raise TypeError("ptc should be of type `LPTCalculator`")

    if ptc is None:
        ptc = LPTCalculator(cosmo, a_arr=a_arr)
    if not isinstance(ptc, LPTCalculator):
        raise TypeError("ptc should be of type `LPTCalculator`")

    if nonlin_pk_type == 'nonlinear':
        Pnl = np.array([ccl.nonlin_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'linear':
        Pnl = np.array([ccl.linear_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'lpt':
        Pnl = None
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))

    z_arr = 1. / ptc.a_s - 1
    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            p_pt = ptc.get_pgg(Pnl,
                               b11, b21, bs1,
                               b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(Pnl, b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgm(Pnl, b12, b22, bs2)
        elif (tracer2.type == 'M'):
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = ccl.Pk2D(a_arr=ptc.a_s,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt,
                     is_logp=False)
    if return_ptc:
        return pt_pk, ptcx
    else:
        return pt_pk
