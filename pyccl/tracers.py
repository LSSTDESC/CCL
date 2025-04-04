r"""
Tracers represent projected quantities that can be cross-correlated to get
angular power spectra (see :func:`~pyccl.cells.angular_cl`). In the most
general case, the angular power spectrum between two tracers is given by

.. math::
    C^{\alpha\beta}_\ell=\frac{2}{\pi}\int d\chi_1\,d\chi_2\,dk\,k^2
    P_{\alpha\beta}(k,\chi_1,\chi_2)\,\Delta^\alpha_\ell(k,\chi_1)\,
    \Delta^\beta_\ell(k,\chi_2),

where :math:`P_{\alpha\beta}` is a generalized power spectrum (see
:class:`~pyccl.pk2d.Pk2D`),
and :math:`\Delta^\alpha_\ell(k,\chi)` is a sum over different contributions
associated to tracer :math:`\alpha`, where every contribution takes the form:

.. math::
    \Delta^\alpha_\ell(k,\chi)=f^\alpha_\ell\,W_\alpha(\chi)\,
    T_\alpha(k,\chi)\,j^{(n_\alpha)}_\ell(k\chi).

Here, :math:`f^\alpha_\ell` is an :math:`\ell`-dependent **prefactor**,
:math:`W_\alpha(\chi)` is the **radial kernel**, :math:`T_\alpha(k,\chi)`
is the **transfer function**, and :math:`j^{(n)}_\ell(x)` is a
generalized version of the **spherical Bessel functions**. Descriptions
of each of these ingredients, and how to implement generalised Tracers
as sub-classes of the :class:`Tracer` base class can be found below. The
documentation of the base :class:`Tracer` class is a good place to start.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from . import ccllib as lib
from .pyutils import check
from .errors import CCLWarning, warnings
from ._core.parameters import physical_constants
from ._core import CCLObject, UnlockInstance, unlock_instance
from .pyutils import (_check_array_params, NoneArr, _vectorize_fn6,
                      _get_spline1d_arrays, _get_spline2d_arrays)

__all__ = ("get_density_kernel", "get_lensing_kernel", "get_kappa_kernel",
           "Tracer", "NzTracer", "NumberCountsTracer", "WeakLensingTracer",
           "CMBLensingTracer", "tSZTracer", "CIBTracer", "ISWTracer",)


def _Sig_MG(cosmo, a, k):
    """Redshift-dependent modification to Poisson equation for massless
    particles under modified gravity.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
        k (:obj:`float` or `array`): Wavenumber for scale

    Returns:
        (:obj:`float` or `array`): Modification to Poisson equation under
            modified gravity at scale factor a.
            Sig_MG is assumed to be proportional to Omega_Lambda(z),
            see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.
    """
    cosmo.compute_distances()
    return _vectorize_fn6(lib.Sig_MG, lib.Sig_MG_vec, cosmo, a, k)


def _check_background_spline_compatibility(cosmo, z):
    """Check that a redshift array lies within the support of the
    CCL background splines.
    """
    cosmo.compute_distances()
    a_bg, _ = _get_spline1d_arrays(cosmo.cosmo.data.chi)
    a = 1/(1+z)

    if a.min() < a_bg.min() or a.max() > a_bg.max():
        raise ValueError(
            "Tracer has wider redshift support than internal CCL splines. "
            f"Tracer: z=[{1/a.max()-1}, {1/a.min()-1}]. "
            f"Background splines: z=[{1/a_bg.max()-1}, {1/a_bg.min()-1}].")


def get_density_kernel(cosmo, *, dndz):
    """This convenience function returns the radial kernel for
    galaxy-clustering-like tracers. Given an unnormalized
    redshift distribution, it returns two arrays: :math:`\\chi`,
    :math:`W(\\chi)`, where :math:`\\chi` is an array of radial
    distances in units of Mpc and :math:`W(\\chi) = p(z)\\,H(z)`, where
    :math:`H(z)` is the expansion rate in units of :math:`{\\rm Mpc}^{-1}`,
    and :math:`p(z)` is the normalized redshift distribution.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object
            used to transform redshifts into distances.
        dndz (:obj:`tuple`): A tuple of arrays ``(z, N(z))``
            giving the redshift distribution of the objects.
            The units are arbitrary; ``N(z)`` will be normalized
            to unity.
    """
    z_n, n = _check_array_params(dndz, 'dndz')
    _check_background_spline_compatibility(cosmo, dndz[0])
    # this call inits the distance splines neded by the kernel functions
    chi = cosmo.comoving_radial_distance(1./(1.+z_n))
    status = 0
    wchi, status = lib.get_number_counts_kernel_wrapper(cosmo.cosmo,
                                                        z_n, n,
                                                        len(z_n),
                                                        status)
    check(status, cosmo=cosmo)
    return chi, wchi


def get_lensing_kernel(cosmo, *, dndz, mag_bias=None, n_chi=None):
    r"""This convenience function returns the radial kernel for
    weak-lensing-like. Given an unnormalized redshift distribution
    and an optional magnification bias function, it returns
    two arrays: :math:`\chi`, :math:`W(\chi)`, where :math:`\chi` is
    an array of radial distances in units of Mpc and :math:`W(\chi)` is
    the lensing kernel:

    .. math::
        W(\chi)=\frac{3H_0^2\Omega_m}{2a}\chi\,
        \int_{z(\chi)}^\infty dz\,\left(1-\frac{5s(z)}{2}\right)\,
        \frac{\chi(z)-\chi}{\chi(z)}

    .. note:: If using this function to compute the magnification bias
              kernel, the result must be multiplied by -2.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object used to
            transform redshifts into distances.
        dndz (:obj:`tuple`): A tuple of arrays ``(z, N(z))``
            giving the redshift distribution of the objects.
            The units are arbitrary; ``N(z)`` will be normalized to unity.
        mag_bias (:obj:`tuple`): A tuple of arrays ``(z, s(z))``
            giving the magnification bias as a function of redshift. If
            ``None``, ``s=0`` will be assumed.
    """
    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias, 'mag_bias')
    _check_background_spline_compatibility(cosmo, dndz[0])

    if n_chi is None:
        # Calculate number of samples in chi
        n_chi = lib.get_nchi_lensing_kernel_wrapper(z_n)

    if (n_chi > len(z_n)
            and cosmo.cosmo.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION):
        warnings.warn(
            f"The number of samples in the n(z) ({len(z_n)}) is smaller than "
            f"the number of samples in the lensing kernel ({n_chi}). Consider "
            "disabling spline integration for the lensing kernel by setting "
            "pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False "
            "before instantiating the Cosmology passed.",
            category=CCLWarning, importance='low')

    # Compute array of chis
    status = 0
    chi, status = lib.get_chis_lensing_kernel_wrapper(cosmo.cosmo, z_n[-1],
                                                      n_chi, status)
    # Compute kernel
    wchi, status = lib.get_lensing_kernel_wrapper(cosmo.cosmo,
                                                  z_n, n, z_n[-1],
                                                  int(has_magbias), z_s, s,
                                                  chi, n_chi, status)
    check(status, cosmo=cosmo)
    return chi, wchi


def get_kappa_kernel(cosmo, *, z_source, n_samples=100):
    """This convenience function returns the radial kernel for
    CMB-lensing-like tracers.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_source (:obj:`float`): Redshift of source plane for CMB lensing.
        n_samples (:obj:`int`): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    _check_background_spline_compatibility(cosmo, np.array([z_source]))
    # this call inits the distance splines neded by the kernel functions
    chi_source = cosmo.comoving_radial_distance(1./(1.+z_source))
    chi = np.linspace(0, chi_source, n_samples)

    status = 0
    wchi, status = lib.get_kappa_kernel_wrapper(cosmo.cosmo, chi_source,
                                                chi, n_samples, status)
    check(status, cosmo=cosmo)
    return chi, wchi


class Tracer(CCLObject):
    """Tracers contain the information necessary to describe the
    contribution of a given sky observable to its cross-power spectrum
    with any other tracer. Tracers are composed of 4 main ingredients:

    * A radial kernel: this expresses the support in redshift/distance
      over which this tracer extends.

    * A transfer function: this is a function of wavenumber and
      scale factor that describes the connection between the tracer
      and the power spectrum on different scales and at different
      cosmic times.

    * An ell-dependent prefactor: normally associated with angular
      derivatives of a given fundamental quantity.

    * The order of the derivative of the Bessel functions with which
      they enter the computation of the angular power spectrum.

    A ``Tracer`` object will in reality be a list of different such
    tracers that get combined linearly when computing power spectra.
    """
    from ._core.repr_ import build_string_Tracer as __repr__

    def __init__(self):
        """By default this `Tracer` object will contain no actual
        tracers
        """
        # Do nothing, just initialize list of tracers
        self._trc = []
        self.chi_fft_dict = {}
        self.avg_weighted_a = []

    def __eq__(self, other):
        # Check object id.
        if self is other:
            return True

        # Check the object class.
        if type(self) is not type(other):
            return False

        # If the tracer collections are empty, return early.
        if not (self or other):
            return True

        # If the tracer collections are not the same length, return early.
        if len(self._trc) != len(other._trc):
            return False

        # Check `der_angles` & `der_bessel` for each tracer in the collection.
        bessel = self.get_bessel_derivative(), other.get_bessel_derivative()
        angles = self.get_angles_derivative(), other.get_angles_derivative()
        if not (np.array_equal(*bessel) and np.array_equal(*angles)):
            return False

        # Check the kernels.
        for t1, t2 in zip(self._trc, other._trc):
            if bool(t1.kernel) ^ bool(t2.kernel):
                # only one of them has a kernel
                return False
            if t1.kernel is None:
                # none of them has a kernel
                continue
            if not np.array_equal(_get_spline1d_arrays(t1.kernel.spline),
                                  _get_spline1d_arrays(t2.kernel.spline)):
                # both have kernels, but they are unequal
                return False

        # Check the transfer functions.
        for t1, t2 in zip(self._trc, other._trc):
            if bool(t1.transfer) ^ bool(t2.transfer):
                # only one of them has a transfer
                return False
            if t1.transfer is None:
                # none of them has a transfer
                continue
            # Check the characteristics of the transfer function.
            for arg in ("extrap_order_lok", "extrap_order_hik",
                        "is_factorizable", "is_log"):
                if getattr(t1.transfer, arg) != getattr(t2.transfer, arg):
                    return False

            c2py = {"fa": _get_spline1d_arrays,
                    "fk": _get_spline1d_arrays,
                    "fka": _get_spline2d_arrays}
            for attr in c2py.keys():
                spl1 = getattr(t1.transfer, attr, None)
                spl2 = getattr(t2.transfer, attr, None)
                if bool(spl1) ^ bool(spl2):
                    # only one of them has this transfer type
                    return False
                if spl1 is None:
                    # none of them has this transfer type
                    continue
                # `pts` contain the the grid points and the transfer functions
                pts1, pts2 = c2py[attr](spl1), c2py[attr](spl2)
                for pt1, pt2 in zip(pts1, pts2):
                    # loop through output points of `_get_splinend_arrays`
                    if not np.array_equal(pt1, pt2):
                        # both have this transfer type, but they are unequal
                        # or are defined at different grid points
                        return False
        return True

    def __hash__(self):
        return hash(repr(self))

    def __bool__(self):
        return bool(self._trc)

    @property
    def chi_min(self):
        """Returns the minimum comoving distance over which this tracer's
        radial kernel is defined, if it exists. For tracers with more than
        one contribution with an associated ``chi_min``, the
        lowest value is returned.
        """
        chis = [tr.chi_min for tr in self._trc]
        return min(chis) if chis else None

    @property
    def chi_max(self):
        """Returns the maximum comoving distance over which this tracer's
        radial kernel is defined, if it exists. For tracers with more than
        one contribution with an associated ``chi_max``, the
        highest value is returned.
        """
        chis = [tr.chi_max for tr in self._trc]
        return max(chis) if chis else None

    def get_kernel(self, chi=None):
        """Get the radial kernels for all tracers contained
        in this ``Tracer``.

        Args:
            chi (:obj:`float` or `array`): values of the comoving
                radial distance in increasing order and in Mpc. If ``None``,
                returns the kernel at the internal spline nodes.

        Returns:
            `array`: list of radial kernels for each tracer. The shape
            will be ``(n_tracer, chi.size)``, where ``n_tracer`` is the
            number of tracers. The last dimension will be squeezed if the
            input is a scalar.
        """
        if chi is None:
            chis = []
        else:
            chi_use = np.atleast_1d(chi)

        kernels = []
        for t in self._trc:
            if t.kernel is None:
                continue
            else:
                if chi is None:
                    chi_use, w = _get_spline1d_arrays(t.kernel.spline)
                    chis.append(chi_use)
                else:
                    status = 0
                    w, status = lib.cl_tracer_get_kernel(
                        t, chi_use, chi_use.size, status)
                    check(status)
                kernels.append(w)

        if chi is None:
            return kernels, chis
        kernels = np.array(kernels)
        if np.ndim(chi) == 0 and kernels.shape != (0,):
            kernels = np.squeeze(kernels, axis=-1)
        return kernels

    def get_f_ell(self, ell):
        """Get the :math:`\\ell`-dependent prefactors for all tracers
        contained in this `Tracer`.

        Args:
            ell (:obj:`float` or `array`): angular multipole values.

        Returns:
            `array`: list of prefactors for each tracer.
            The shape will be ``(n_tracer, ell.size)``, where
            ``n_tracer`` is the number of tracers. The last
            dimension will be squeezed if the input ``ell`` is a
            scalar.
        """
        ell_use = np.atleast_1d(ell)
        f_ells = []
        for t in self._trc:
            status = 0
            f, status = lib.cl_tracer_get_f_ell(t, ell_use,
                                                ell_use.size,
                                                status)
            check(status)
            f_ells.append(f)
        f_ells = np.array(f_ells)
        if np.ndim(ell) == 0:
            if f_ells.shape != (0,):
                f_ells = np.squeeze(f_ells, axis=-1)
        return f_ells

    def get_transfer(self, lk, a):
        """Get the transfer functions for all tracers contained
        in this ``Tracer``.

        Args:
            lk (:obj:`float` or `array`): values of the natural logarithm of
                the wave number (in units of inverse Mpc) in increasing
                order.
            a (:obj:`float` or `array`): values of the scale factor.

        Returns:
            `array`: list of transfer functions for each tracer.
            The shape will be ``(n_tracer, lk.size, a.size)``, where
            ``n_tracer`` is the number of tracers. The other dimensions
            will be squeezed if the inputs are scalars.
        """
        lk_use = np.atleast_1d(lk)
        a_use = np.atleast_1d(a)
        transfers = []
        for t in self._trc:
            status = 0
            t, status = lib.cl_tracer_get_transfer(t, lk_use, a_use,
                                                   lk_use.size * a_use.size,
                                                   status)
            check(status)
            transfers.append(t.reshape([lk_use.size, a_use.size]))
        transfers = np.array(transfers)
        if transfers.shape != (0,):
            if np.ndim(a) == 0:
                transfers = np.squeeze(transfers, axis=-1)
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-1)
            else:
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-2)
        return transfers

    def get_bessel_derivative(self):
        """Get list of Bessel function derivative orders for all tracers
        contained in this ``Tracer``.

        Returns:
            `array`: list of Bessel derivative orders for each tracer.
        """
        return np.array([t.der_bessel for t in self._trc])

    def get_angles_derivative(self):
        r"""Get list of the :math:`\ell`-dependent prefactor order for all
        tracers contained in this ``Tracer``.

        Returns:
            `array`: list of angular derivative orders for each tracer.
        """
        return np.array([t.der_angles for t in self._trc])

    def _get_fkem_fft(self, tracer, Nchi, chimin, chimax, ell):
        """Get list fft integral over chi for FKEM non-limber calculation
        contained in this ``Tracer``.

        Returns:
            `tuple`: k values and fft integral values at each k
        """
        temp = self.chi_fft_dict.get((tracer, Nchi, chimin, chimax, ell))
        if temp is None:
            return None, None
        return temp[0], temp[1]

    def _set_fkem_fft(self, tracer, Nchi, chimin, chimax, ell, ks, fft):
        """Set list fft integral over chi for FKEM non-limber calculation
        contained in this ``Tracer``.

        Returns:
            `tuple`: k values and fft integral values at each k
        """
        self.chi_fft_dict[(tracer, Nchi, chimin, chimax, ell)] = (ks, fft)
        return ks, fft

    def get_avg_weighted_a(self):
        """Get list of kernel-averaged scale factors for all tracers
        contained in this ``Tracer``.

        Returns:
            `array`: list of kernel-averaged scale factors for each tracer.
        """

        return np.array(self.avg_weighted_a)

    def _MG_add_tracer(self, cosmo, kernel, z_b, der_bessel=0, der_angles=0,
                       bias_transfer_a=None, bias_transfer_k=None):
        """ function to set mg_transfer in the right format and add MG tracers
            for different cases including different cases and biases like
            intrinsic alignements (IA) when present
        """
        # Getting MG transfer function and building a k-array
        mg_transfer = self._get_MG_transfer_function(cosmo, z_b)

        # case with no astro biases
        if ((bias_transfer_a is None) and (bias_transfer_k is None)):
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a and  k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * (bias_transfer_k[1] *
                                mg_transfer[2]).T).T)
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a but not k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * mg_transfer[2].T).T)
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on k but not a
        elif ((bias_transfer_a is None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_k[1] * mg_transfer[2]))
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

    def _get_MG_transfer_function(self, cosmo, z):
        """ This function allows to obtain the function Sigma(z,k) (1 or 2D
            arrays) for an array of redshifts coming from a redshift
            distribution (defined by the user) and a single value or
            an array of k specified by the user. We obtain then Sigma(z,k) as a
            1D array for those z and k arrays and then convert it to a 2D array
            taking into consideration the given sizes of the arrays for z and k
            The MG parameter array goes then as a multiplicative factor within
            the MG transfer function. If k is not specified then only a 1D
            array for Sigma(a,k=0) is used.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object used
                to transform redshifts into distances.
            z (float or tuple of arrays): a single z value (e.g. for CMB)
                or a tuple of arrays (z, N(z)) giving the redshift distribution
                of the objects. The units are arbitrary; N(z) will be
                normalized to unity.
            k (float or array): a single k value or an array of k for which we
                calculate the MG parameter Sigma(a,k). For now, the k range
                should be limited to linear scales.
        """
        # Sampling scale factor from a very small (at CMB for example)
        # all the way to 1 here and today for the transfer function.
        # For a < a_single it is GR (no early MG)
        if isinstance(z, (int, float)):
            a_single = 1/(1+z)
            a = np.linspace(a_single, 1, 100)
            # a_single is for example like for the CMB surface
        else:
            if z[0] != 0.0:
                stepsize = z[1]-z[0]
                samplesize = int(z[0]/stepsize)
                z_0_to_zmin = np.linspace(0.0, z[0] - stepsize, samplesize)
                z = np.concatenate((z_0_to_zmin, z))
            a = 1./(1.+z)
        a.sort()
        # Scale-dependant MG case with an array of k
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        status = 0
        lk, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
        k = np.exp(lk)
        # computing MG factor array
        mgfac_1d = 1
        mgfac_1d += _Sig_MG(cosmo, a, k)
        # converting 1D MG factor to a 2D array, so it is compatible
        # with the transfer_ka input structure in MG_add.tracer and
        # add.tracer
        mgfac_2d = mgfac_1d.reshape(len(a), -1, order='F')
        # setting transfer_ka for this case
        mg_transfer = (a, lk, mgfac_2d)

        return mg_transfer

    @unlock_instance
    def add_tracer(self, cosmo, *, kernel=None,
                   transfer_ka=None, transfer_k=None, transfer_a=None,
                   der_bessel=0, der_angles=0,
                   is_logt=False, extrap_order_lok=0, extrap_order_hik=2):
        """Adds one more tracer to the list contained in this ``Tracer``.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object.
            kernel (:obj:`tuple`): A tuple of arrays
                ``(chi, w_chi)`` describing the radial kernel of this
                tracer. ``chi`` should contain values of the comoving
                radial distance in increasing order, and ``w_chi`` should
                contain the values of the kernel at those values of the
                radial distance. The kernel will be assumed to be zero
                outside the range of distances covered by ``chi``. If
                ``kernel`` is ``None`` a constant kernel
                :math:`W(\\chi)=1` will be assumed everywhere.
            transfer_ka (:obj:`tuple`): a tuple of arrays
                ``(a, lk,t_ka)`` describing the most general transfer
                function for a tracer. ``a`` should be an array of scale
                factor values in increasing order. ``lk`` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                ``t_ka`` should be an array of shape ``(na, nk)``, where
                ``na`` and ``nk`` are the sizes of ``a`` and ``lk``
                respectively.``t_ka`` should hold the values of the transfer
                function at the corresponding values of ``a`` and ``lk``. If
                your transfer function is factorizable (i.e.
                :math:`T(a,k) = A(a)\\, K(k))`,
                it is more efficient to set this to ``None`` and use
                ``transfer_k`` and ``transfer_a`` to describe :math:`K` and
                :math:`A` respectively. The transfer function will be assumed
                continuous and constant outside the range of scale factors
                covered by ``a``. It will be extrapolated using polynomials
                of order ``extrap_order_lok`` and ``extrap_order_hik`` below
                and above the range of wavenumbers covered by ``lk``
                respectively. If this argument is not ``None``, the values of
                ``transfer_k`` and ``transfer_a`` will be ignored.
            transfer_k (:obj:`tuple`): a tuple of arrays
                ``(lk,t_k)`` describing the scale-dependent part of a
                factorizable transfer function. ``lk`` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                ``t_k`` should be an array of the same size holding the
                values of the k-dependent part of the transfer function
                at those wavenumbers. It will be extrapolated using
                polynomials of order ``extrap_order_lok`` and
                ``extrap_order_hik`` below and above the range of wavenumbers
                covered by ``lk`` respectively. If ``None``, the k-dependent
                part of the transfer function will be set to 1 everywhere.
            transfer_a (:obj:`tuple`): a tuple of arrays
                `(a, t_a)`` describing the time-dependent part of a
                factorizable transfer function. ``a`` should be an array of
                scale factor values in increasing order. ``t_a`` should
                contain the time-dependent part of the transfer function
                at those values of the scale factor. The time dependence
                will be assumed continuous and constant outside the range
                covered by ``a``. If ``None``, the time-dependent part of the
                transfer function will be set to 1 everywhere.
            der_bessel (:obj:`int`): order of the derivative of the Bessel
                functions with which this tracer enters the calculation
                of the power spectrum. Allowed values are -1, 0, 1 and 2.
                0, 1 and 2 correspond to the raw functions, their first
                derivatives or their second derivatives. -1 corresponds to
                the raw functions divided by the square of their argument.
                We enable this special value because this type of dependence
                is ubiquitous for many common tracers (lensing, IAs), and
                makes the corresponding transfer functions more stable
                for small :math:`k` or :math:`\\chi`.
            der_angles (:obj:`int`): integer describing the ell-dependent prefactor
                associated with this tracer. Allowed values are 0, 1 and 2.
                0 means no prefactor. 1 means a prefactor
                :math:`\\ell(\\ell+1)`, associated with the angular
                Laplacian and used e.g. for lensing convergence and
                magnification. 2 means a prefactor
                :math:`\\sqrt{(\\ell+2)!/(\\ell-2)!}`, associated with the
                angular derivatives of spin-2 fields (e.g. cosmic shear, IAs).
            is_logt (:obj:`bool`): if ``True``, ``transfer_ka``, ``transfer_k`` and
                ``transfer_a`` will contain the natural logarithm of the
                transfer function (or their factorizable parts).
            extrap_order_lok (:obj:`int`): polynomial order used to extrapolate the
                transfer functions for low wavenumbers not covered by the
                input arrays.
            extrap_order_hik (:obj:`int`): polynomial order used to extrapolate the
                transfer functions for high wavenumbers not covered by the
                input arrays.
        """ # noqa
        is_factorizable = transfer_ka is None
        is_k_constant = (transfer_ka is None) and (transfer_k is None)
        is_a_constant = (transfer_ka is None) and (transfer_a is None)
        is_kernel_constant = kernel is None

        chi_s, wchi_s = _check_array_params(kernel, 'kernel')
        if is_factorizable:
            a_s, ta_s = _check_array_params(transfer_a, 'transfer_a')
            lk_s, tk_s = _check_array_params(transfer_k, 'transfer_k')
            tka_s = NoneArr
            if (not is_a_constant) and (a_s.shape != ta_s.shape):
                raise ValueError("Time-dependent transfer arrays "
                                 "should have the same shape")
            if (not is_k_constant) and (lk_s.shape != tk_s.shape):
                raise ValueError("Scale-dependent transfer arrays "
                                 "should have the same shape")
        else:
            a_s, lk_s, tka_s = _check_array_params(transfer_ka, 'transer_ka',
                                                   arr3=True)
            if tka_s.shape != (len(a_s), len(lk_s)):
                raise ValueError("2D transfer array has inconsistent "
                                 "shape. Should be (na,nk)")
            tka_s = tka_s.flatten()
            ta_s = NoneArr
            tk_s = NoneArr

        if not (np.diff(a_s) > 0).all():
            raise ValueError("Scale factor must be monotonically "
                             "increasing")

        status = 0
        ret = lib.cl_tracer_t_new_wrapper(cosmo.cosmo,
                                          int(der_bessel),
                                          int(der_angles),
                                          chi_s, wchi_s,
                                          a_s, lk_s,
                                          tka_s, tk_s, ta_s,
                                          int(is_logt),
                                          int(is_factorizable),
                                          int(is_k_constant),
                                          int(is_a_constant),
                                          int(is_kernel_constant),
                                          int(extrap_order_lok),
                                          int(extrap_order_hik),
                                          status)
        self._trc.append(_check_returned_tracer(ret))
        a = cosmo.scale_factor_of_chi(chi_s)
        if len(wchi_s) == 0:
            avg_a = 1.0
        else:
            wint = simpson(wchi_s, x=a)
            if wint != 0:  # Avoid division by zero
                avg_a = simpson(a*wchi_s, x=a)/wint
            else:  # If kernel integral is zero, just set to z=0
                avg_a = 1.0
        self.avg_weighted_a.append(avg_a)

    @classmethod
    def from_z_power(cls, cosmo, *, A, alpha, z_min=0., z_max=6., n_chi=1024):
        """Constructor for tracers associated with a radial kernel of the form

        .. math::
            W(\\chi) = \\frac{A}{(1+z)^\\alpha},

        where :math:`A` is an amplitude and :math:`\\alpha` is a power
        law index. The kernel only has support in the redshift range
        ``[z_min, z_max]``.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
            A (:obj:`float`): amplitude parameter.
            alpha (:obj:`float`): power law index.
            z_min (:obj:`float`): minimum redshift from to which we define the kernel.
            z_max (:obj:`float`): maximum redshift up to which we define the kernel.
            n_chi (:obj:`float`): number of intervals in the radial comoving
                distance on which we sample the kernel.
        """ # noqa
        if z_min >= z_max:
            raise ValueError("z_min should be smaller than z_max.")

        tracer = cls()

        chi_min = cosmo.comoving_radial_distance(1./(1+z_min))
        chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
        chi_arr = np.linspace(chi_min, chi_max, n_chi)
        a_arr = cosmo.scale_factor_of_chi(chi_arr)
        w_arr = A * a_arr**alpha

        tracer.add_tracer(cosmo, kernel=(chi_arr, w_arr))
        return tracer

    def __del__(self):
        # Sometimes lib is freed before some Tracers, in which case, this
        # doesn't work.
        # So just check that lib.cl_tracer_t_free is still a real function.
        if hasattr(self, '_trc') and lib.cl_tracer_t_free is not None:
            for t in self._trc:
                lib.cl_tracer_t_free(t)


class NzTracer(Tracer):
    """Specific base class for tracers with an internal ``_dndz``
    redshift distribution interpolator. These include
    :func:`NumberCountsTracer` and :func:`WeakLensingTracer`.
    """

    def get_dndz(self, z):
        """Get the redshift distribution for this tracer.

        Args:
            z (:obj:`float` or `array`): redshift values.

        Returns:
            `array`: redshift distribution evaluated at the
            input values of ``z``.
        """
        return self._dndz(z)


def NumberCountsTracer(cosmo, *, dndz, bias=None, mag_bias=None,
                       has_rsd, n_samples=256):
    """Specific `Tracer` associated to galaxy clustering with linear
    scale-independent bias, including redshift-space distortions and
    magnification. The associated contributions are described in
    detail in Section 2.4.1 of the `CCL paper
    <https://arxiv.org/abs/1812.05995>`_.

    .. warning:: When including redshift-space distortions, the
        current implementation assumes linear, scale-independent growth
        Although this should be valid in :math:`\\Lambda` CDM and on the
        large scales (especially when considering a broad :math:`N(z)`),
        this approximation should be borne in mind.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        dndz (:obj:`tuple`): A tuple of arrays ``(z, N(z))``
            giving the redshift distribution of the objects. The units are
            arbitrary; ``N(z)`` will be normalized to unity.
        bias (:obj:`tuple`): A tuple of arrays ``(z, b(z))``
            giving the galaxy bias. If ``None``, this tracer won't include
            a term proportional to the matter density contrast.
        mag_bias (:obj:`tuple`): A tuple of arrays ``(z, s(z))``
            giving the magnification bias as a function of redshift. If
            ``None``, the tracer is assumed to not have magnification bias
            terms.
        has_rsd (:obj:`bool`): If ``True``, this tracer will include a
            redshift-space distortion term.
        n_samples (:obj:`int`): number of samples over which the
            magnification lensing kernel is desired. These will be equi-spaced
            in radial distance. The kernel is quite smooth, so usually O(100)
            samples is enough.
    """
    tracer = NzTracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    with UnlockInstance(tracer, mutate=False):
        tracer._dndz = interp1d(z_n, n, bounds_error=False, fill_value=0)

    if (bias is None) and (not has_rsd) and (mag_bias is None):
        raise ValueError("Number counts tracers must have a non-zero bias, "
                         "RSDs, or a magnification bias contribution.")

    kernel_d = None
    if bias is not None:  # Has density term
        # Kernel
        if kernel_d is None:
            kernel_d = get_density_kernel(cosmo, dndz=dndz)
        # Transfer
        z_b, b = _check_array_params(bias, 'bias')
        # Reverse order for increasing a
        t_a = (1./(1+z_b[::-1]), b[::-1])
        tracer.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a)

    if has_rsd:  # Has RSDs
        # Kernel
        if kernel_d is None:
            kernel_d = get_density_kernel(cosmo, dndz=dndz)
        # Transfer (growth rate)
        z_b, _ = _check_array_params(dndz, 'dndz')
        a_s = 1./(1+z_b[::-1])
        t_a = (a_s, -cosmo.growth_rate(a_s))
        tracer.add_tracer(cosmo, kernel=kernel_d,
                          transfer_a=t_a, der_bessel=2)
    if mag_bias is not None:  # Has magnification bias
        # Kernel
        chi, w = get_lensing_kernel(cosmo, dndz=dndz, mag_bias=mag_bias,
                                    n_chi=n_samples)
        # Multiply by -2 for magnification
        kernel_m = (chi, -2 * w)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(cosmo, kernel=kernel_m,
                              der_bessel=-1, der_angles=1)
        else:
            # MG case
            z_b, _ = _check_array_params(dndz, 'dndz')
            tracer._MG_add_tracer(cosmo, kernel_m, z_b,
                                  der_bessel=-1, der_angles=1)
    return tracer


def WeakLensingTracer(cosmo, *, dndz, has_shear=True, ia_bias=None,
                      use_A_ia=True, n_samples=256):
    """Specific `Tracer` associated to galaxy shape distortions including
    lensing shear and intrinsic alignments within the L-NLA model.
    The associated contributions are described in detail in Section 2.4.1
    of the `CCL paper <https://arxiv.org/abs/1812.05995>`_.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        dndz (:obj:`tuple`): A tuple of arrays ``(z, N(z))``
            giving the redshift distribution of the objects. The units are
            arbitrary; ``N(z)`` will be normalized to unity.
        has_shear (:obj:`bool`): set to ``False`` if you want to omit the
            lensing shear contribution from this tracer.
        ia_bias (:obj:`tuple`): A tuple of arrays
            ``(z, A_IA(z))`` giving the intrinsic alignment amplitude
            ``A_IA(z)``. If ``None``, the tracer is assumed to not have
            intrinsic alignments.
        use_A_ia (:obj:`bool`): set to ``True`` to use the conventional IA
            normalization. Set to ``False`` to use the raw input amplitude,
            which will usually be 1 for use with perturbation theory IA
            modeling.
        n_samples (:obj:`int`): number of samples over which the lensing
            kernel is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    tracer = NzTracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    with UnlockInstance(tracer, mutate=False):
        tracer._dndz = interp1d(z_n, n, bounds_error=False, fill_value=0)

    if has_shear:
        kernel_l = get_lensing_kernel(cosmo, dndz=dndz, n_chi=n_samples)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(cosmo, kernel=kernel_l,
                              der_bessel=-1, der_angles=2)
        else:
            # MG case
            tracer._MG_add_tracer(cosmo, kernel_l, z_n,
                                  der_bessel=-1, der_angles=2)
    else:
        if ia_bias is None:
            raise ValueError("Weak lensing tracers with no shear must "
                             "have a non-zero intrinsic alignment amplitude.")

    if ia_bias is not None:  # Has intrinsic alignments
        z_a, tmp_a = _check_array_params(ia_bias, 'ia_bias')
        # Kernel
        kernel_i = get_density_kernel(cosmo, dndz=dndz)
        if use_A_ia:
            # Normalize so that A_IA=1
            D = cosmo.growth_factor(1./(1+z_a))
            # Transfer
            # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
            # and note that we use C_1= 5e-14 from arXiv:0705.0166
            rho_m = physical_constants.RHO_CRITICAL * cosmo['Omega_m']
            a = - tmp_a * 5e-14 * rho_m / D
        else:
            # use the raw input normalization. Normally, this will be 1
            # to allow nonlinear PT IA models, where normalization is
            # already applied to the power spectrum.
            a = tmp_a
        # Reverse order for increasing a
        t_a = (1./(1+z_a[::-1]), a[::-1])
        tracer.add_tracer(cosmo, kernel=kernel_i, transfer_a=t_a,
                          der_bessel=-1, der_angles=2)
    return tracer


def CMBLensingTracer(cosmo, *, z_source, n_samples=100):
    r"""A Tracer for CMB lensing convergence :math:`\kappa`.
    The associated kernel and transfer function are described
    in Eq. 31 of the `CCL paper <https://arxiv.org/abs/1812.05995>`_.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_source (:obj:`float`): Redshift of source plane for CMB lensing.
        n_samples (:obj:`int`): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    tracer = Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()
    kernel = get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
    if (cosmo['sigma_0'] == 0):
        tracer.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
    else:
        tracer._MG_add_tracer(cosmo, kernel, z_source,
                              der_bessel=-1, der_angles=1)
    return tracer


def tSZTracer(cosmo, *, z_max=6., n_chi=1024):
    """Specific :class:`Tracer` associated with the thermal Sunyaev Zel'dovich
    Compton-y parameter. The radial kernel for this tracer is simply given by

    .. math::
       W(\\chi) = \\frac{\\sigma_T}{m_ec^2} \\frac{1}{1+z},

    where :math:`\\sigma_T` is the Thomson scattering cross section and
    :math:`m_e` is the electron mass.

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the electron pressure
    in physical (non-comoving) units of :math:`eV\\,{\\rm cm}^{-3}`.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_max (:obj:`float`): maximum redshift up to which we define the
            kernel.
        n_chi (:obj:`float`): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """
    # This is \sigma_T / (m_e * c^2)
    prefac = 4.01710079e-06
    return Tracer.from_z_power(cosmo, A=prefac, alpha=1, z_min=0.,
                               z_max=z_max, n_chi=n_chi)


def CIBTracer(cosmo, *, z_min=0., z_max=6., n_chi=1024):
    """Specific :class:`Tracer` associated with the cosmic infrared
    background (CIB). The radial kernel for this tracer is simply

    .. math::
       W(\\chi) = \\frac{1}{1+z}.

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the CIB emissivity
    density in units of
    :math:`{\\rm Jy}\\,{\\rm Mpc}^{-1}\\,{\\rm srad}^{-1}` (or
    multiples thereof -- see e.g.
    :class:`~pyccl.halos.profiles.cib_shang12.HaloProfileCIBShang12`).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_min (:obj:`float`): minimum redshift down to which we define the
            kernel.
        z_max (:obj:`float`): maximum redshift up to which we define the
            kernel.
        n_chi (:obj:`float`): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """
    return Tracer.from_z_power(cosmo, A=1.0, alpha=1, z_min=z_min,
                               z_max=z_max, n_chi=n_chi)


def ISWTracer(cosmo, *, z_max=6., n_chi=1024):
    """Specific :class:`Tracer` associated with the integrated Sachs-Wolfe
    effect (ISW). Useful when cross-correlating any low-redshift probe with
    the primary CMB anisotropies. The ISW contribution to the temperature
    fluctuations is:

    .. math::
        \\Delta T_{\\rm CMB} =
        2T_{\\rm CMB} \\int_0^{\\chi_{LSS}}d\\chi a\\,\\dot{\\phi}

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the matter power spectrum.

    .. warning:: The current implementation of this tracer assumes a
        standard Poisson equation relating :math:`\\phi` and
        :math:`\\delta`, and linear, scale-independent structure growth.
        Although this should be valid in :math:`\\Lambda` CDM and on the
        large scales the ISW is sensitive to, these approximations must
        be borne in mind.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_max (:obj:`float`): maximum redshift up to which we define the
            kernel.
        n_chi (:obj:`float`): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """
    tracer = Tracer()

    chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
    chi = np.linspace(0, chi_max, n_chi)
    a_arr = cosmo.scale_factor_of_chi(chi)
    H0 = cosmo['h'] / physical_constants.CLIGHT_HMPC
    OM = cosmo['Omega_c']+cosmo['Omega_b']
    Ez = cosmo.h_over_h0(a_arr)
    fz = cosmo.growth_rate(a_arr)
    w_arr = 3*cosmo['T_CMB']*H0**3*OM*Ez*chi**2*(1-fz)

    tracer.add_tracer(cosmo, kernel=(chi, w_arr), der_bessel=-1)
    return tracer


def _check_returned_tracer(return_val):
    """Wrapper to catch exceptions when tracers are spawned from C."""
    if (isinstance(return_val, int)):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr
