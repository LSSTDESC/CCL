from . import ccllib as lib
from .core import check
from .background import comoving_radial_distance, growth_rate, growth_factor
import numpy as np
import collections

NoneArr = np.array([])


def get_density_kernel(cosmo, dndz):
    """This convenience function returns the radial kernel for
    galaxy-clustering-like tracers. Given an unnormalized
    redshift distribution, it returns two arrays: chi, w(chi),
    where chi is an array of radial distances in units of
    Mpc and w(chi) = p(z) * H(z), where H(z) is the expansion
    rate in units of Mpc^-1 and p(z) is the normalized
    redshift distribution.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
            transform redshifts into distances.
        dndz (tulple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects.
            The units are arbitrary; N(z) will be normalized
            to unity.
    """
    if ((not isinstance(dndz, collections.Iterable))
        or (len(dndz) != 2)
        or (not (isinstance(dndz[0], collections.Iterable)
                 and isinstance(dndz[1], collections.Iterable)))):
        raise ValueError("dndz needs to be a tuple of two arrays.")
    z_n, n = _check_array_params(dndz)
    # this call inits the distance splines neded by the kernel functions
    chi = comoving_radial_distance(cosmo, 1./(1.+z_n))
    status = 0
    wchi, status = lib.get_number_counts_kernel_wrapper(cosmo.cosmo,
                                                        z_n, n,
                                                        len(z_n),
                                                        status)
    check(status)
    return chi, wchi


def get_lensing_kernel(cosmo, dndz, mag_bias=None):
    """This convenience function returns the radial kernel for
    weak-lensing-like. Given an unnormalized redshift distribution
    and an optional magnification bias function, it returns
    two arrays: chi, w(chi), where chi is an array of radial
    distances in units of Mpc and w(chi) is the lensing shear
    kernel (or the magnification one if `mag_bias` is not `None`).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
            transform redshifts into distances.
        dndz (tulple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects.
            The units are arbitrary; N(z) will be normalized
            to unity.
        mag_bias (tuple of arrays, optional): A tuple of arrays (z, s(z))
            giving the magnification bias as a function of redshift. If
            `None`, s=0 will be assumed
    """
    if ((not isinstance(dndz, collections.Iterable))
        or (len(dndz) != 2)
        or (not (isinstance(dndz[0], collections.Iterable)
                 and isinstance(dndz[1], collections.Iterable)))):
        raise ValueError("dndz needs to be a tuple of two arrays.")

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz)
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias)

    # Calculate number of samples in chi
    nchi = lib.get_nchi_lensing_kernel_wrapper(z_n)
    # Compute array of chis
    status = 0
    chi, status = lib.get_chis_lensing_kernel_wrapper(cosmo.cosmo, z_n[-1],
                                                      nchi, status)
    # Compute kernel
    wchi, status = lib.get_lensing_kernel_wrapper(cosmo.cosmo,
                                                  z_n, n, z_n[-1],
                                                  int(has_magbias), z_s, s,
                                                  chi, nchi, status)
    check(status)
    return chi, wchi


def get_kappa_kernel(cosmo, z_source, nsamples):
    """This convenience function returns the radial kernel for
    CMB-lensing-like tracers.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        z_source (float): Redshift of source plane for CMB lensing.
        nsamples (int): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    # this call inits the distance splines neded by the kernel functions
    chi_source = comoving_radial_distance(cosmo, 1./(1.+z_source))
    chi = np.linspace(0, chi_source, nsamples)

    status = 0
    wchi, status = lib.get_kappa_kernel_wrapper(cosmo.cosmo, chi_source,
                                                chi, nsamples, status)
    check(status)
    return chi, wchi


class Tracer(object):
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

    A `Tracer` object will in reality be a list of different such
    tracers that get combined linearly when computing power spectra.
    Further details can be found in Section 4.9 of the CCL note.
    """
    def __init__(self):
        """By default this `Tracer` object will contain no actual
        tracers
        """
        # Do nothing, just initialize list of tracers
        self._trc = []

    def _dndz(self, z):
        raise NotImplementedError("`get_dndz` not implemented for "
                                  "this `Tracer` type.")

    def get_dndz(self, z):
        """Get the redshift distribution for this tracer.
        Only available for some tracers (:class:`NumberCountsTracer` and
        :class:`WeakLensingTracer`).

        Args:
            z (float or array_like): redshift values.

        Returns:
            array_like: redshift distribution evaluated at the \
                input values of `z`.
        """
        return self._dndz(z)

    def get_kernel(self, chi):
        """Get the radial kernels for all tracers contained
        in this `Tracer`.

        Args:
            chi (float or array_like): values of the comoving
                radial distance in increasing order and in Mpc.

        Returns:
            array_like: list of radial kernels for each tracer. \
                The shape will be `(n_tracer, chi.size)`, where \
                `n_tracer` is the number of tracers. The last \
                dimension will be squeezed if the input is a \
                scalar.
        """
        if not hasattr(self, '_trc'):
            return []

        chi_use = np.atleast_1d(chi)
        kernels = []
        for t in self._trc:
            status = 0
            w, status = lib.cl_tracer_get_kernel(t, chi_use,
                                                 chi_use.size,
                                                 status)
            check(status)
            kernels.append(w)
        kernels = np.array(kernels)
        if np.ndim(chi) == 0:
            if kernels.shape != (0,):
                kernels = np.squeeze(kernels, axis=-1)
        return kernels

    def get_f_ell(self, ell):
        """Get the ell-dependent prefactors for all tracers
        contained in this `Tracer`.

        Args:
            ell (float or array_like): angular multipole values.

        Returns:
            array_like: list of prefactors for each tracer. \
                The shape will be `(n_tracer, ell.size)`, where \
                `n_tracer` is the number of tracers. The last \
                dimension will be squeezed if the input is a \
                scalar.
        """
        if not hasattr(self, '_trc'):
            return []

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
        in this `Tracer`.

        Args:
            lk (float or array_like): values of the natural logarithm of
                the wave number (in units of inverse Mpc) in increasing
                order.
            a (float or array_like): values of the scale factor.

        Returns:
            array_like: list of transfer functions for each tracer. \
                The shape will be `(n_tracer, lk.size, a.size)`, where \
                `n_tracer` is the number of tracers. The other \
                dimensions will be squeezed if the inputs are scalars.
        """
        if not hasattr(self, '_trc'):
            return []

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
        """Get Bessel function derivative orders for all tracers contained
        in this `Tracer`.

        Returns:
            array_like: list of Bessel derivative orders for each tracer.
        """
        if not hasattr(self, '_trc'):
            return []

        return np.array([t.der_bessel for t in self._trc])

    def add_tracer(self, cosmo, kernel=None,
                   transfer_ka=None, transfer_k=None, transfer_a=None,
                   der_bessel=0, der_angles=0,
                   is_logt=False, extrap_order_lok=0, extrap_order_hik=2):
        """Adds one more tracer to the list contained in this `Tracer`.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): cosmology object.
            kernel (tulple of arrays, optional): A tuple of arrays
                (`chi`, `w_chi`) describing the radial kernel of this
                tracer. `chi` should contain values of the comoving
                radial distance in increasing order, and `w_chi` should
                contain the values of the kernel at those values of the
                radial distance. The kernel will be assumed to be zero
                outside the range of distances covered by `chi`. If
                `kernel` is `None` a constant kernel w(chi)=1 will be
                assumed everywhere.
            transfer_ka (tuple of arrays, optional): a tuple of arrays
                (`a`,`lk`,`t_ka`) describing the most general transfer
                function for a tracer. `a` should be an array of scale
                factor values in increasing order. `lk` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                `t_ka` should be an array of shape `(na,nk)`, where
                `na` and `nk` are the sizes of `a` and `lk` respectively.
                `t_ka` should hold the values of the transfer function at
                the corresponding values of `a` and `lk`. If your transfer
                function is factorizable (i.e. T(a,k) = A(a) * K(k)), it is
                more efficient to set this to `None` and use `transfer_k`
                and `transfer_a` to describe K and A respectively. The
                transfer function will be assumed continuous and constant
                outside the range of scale factors covered by `a`. It will
                be extrapolated using polynomials of order `extrap_order_lok`
                and `extrap_order_hik` below and above the range of
                wavenumbers covered by `lk` respectively. If this argument
                is not `None`, the values of `transfer_k` and `transfer_a`
                will be ignored.
            transfer_k (tuple of arrays, optional): a tuple of arrays
                (`lk`,`t_k`) describing the scale-dependent part of a
                factorizable transfer function. `lk` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                `t_k ` should be an array of the same size holding the
                values of the k-dependent part of the transfer function
                at those wavenumbers. It will be extrapolated using
                polynomials of order `extrap_order_lok` and `extrap_order_hik`
                below and above the range of wavenumbers covered by `lk`
                respectively. If `None`, the k-dependent part of the transfer
                function will be set to 1 everywhere.
            transfer_a (tuple of arrays, optional): a tuple of arrays
                (`a`,`t_a`) describing the time-dependent part of a
                factorizable transfer function. `a` should be an array of
                scale factor values in increasing order. `t_a` should
                contain the time-dependent part of the transfer function
                at those values of the scale factor. The time dependence
                will be assumed continuous and constant outside the range
                covered by `a`. If `None`, the time-dependent part of the
                transfer function will be set to 1 everywhere.
            der_bessel (int): order of the derivative of the Bessel
                functions with which this tracer enters the calculation
                of the power spectrum. Allowed values are -1, 0, 1 and 2.
                0, 1 and 2 correspond to the raw functions, their first
                derivatives or their second derivatives. -1 corresponds to
                the raw functions divided by the square of their argument.
                We enable this special value because this type of dependence
                is ubiquitous for many common tracers (lensing, IAs), and
                makes the corresponding transfer functions more stables
                for small k or chi.
            der_angles (int): integer describing the ell-dependent prefactor
                associated with this tracer. Allowed values are 0, 1 and 2.
                0 means no prefactor. 1 means a prefactor ell*(ell+1),
                associated with the angular laplacian and used e.g. for
                lensing convergence and magnification. 2 means a prefactor
                sqrt((ell+2)!/(ell-2)!), associated with the angular
                derivatives of spin-2 fields (e.g. cosmic shear, IAs).
            is_logt (bool): if `True`, `transfer_ka`, `transfer_k` and
                `transfer_a` will contain the natural logarithm of the
                transfer function (or their factorizable parts). Default is
                `False`.
            extrap_order_lok (int): polynomial order used to extrapolate the
                transfer functions for low wavenumbers not covered by the
                input arrays.
            extrap_order_hik (int): polynomial order used to extrapolate the
                transfer functions for high wavenumbers not covered by the
                input arrays.
        """
        is_factorizable = transfer_ka is None
        is_k_constant = (transfer_ka is None) and (transfer_k is None)
        is_a_constant = (transfer_ka is None) and (transfer_a is None)
        is_kernel_constant = kernel is None

        chi_s, wchi_s = _check_array_params(kernel)
        if is_factorizable:
            a_s, ta_s = _check_array_params(transfer_a)
            lk_s, tk_s = _check_array_params(transfer_k)
            tka_s = NoneArr
            if (not is_a_constant) and (a_s.shape != ta_s.shape):
                raise ValueError("Time-dependent transfer arrays "
                                 "should have the same shape")
            if (not is_k_constant) and (lk_s.shape != tk_s.shape):
                raise ValueError("Scale-dependent transfer arrays "
                                 "should have the same shape")
        else:
            a_s, lk_s, tka_s = _check_array_params(transfer_ka, arr3=True)
            if tka_s.shape != (len(a_s), len(lk_s)):
                raise ValueError("2D transfer array has inconsistent "
                                 "shape. Should be (na,nk)")
            tka_s = tka_s.flatten()
            ta_s = NoneArr
            tk_s = NoneArr

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

    def __del__(self):
        if hasattr(self, '_trc'):
            for t in self._trc:
                lib.cl_tracer_t_free(t)


class NumberCountsTracer(Tracer):
    """Specific `Tracer` associated to galaxy clustering with linear
    scale-independent bias, including redshift-space distortions and
    magnification.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        has_rsd (bool): Flag for whether the tracer has a
            redshift-space distortion term.
        dndz (tuple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects. The units are
            arbitrary; N(z) will be normalized to unity.
        bias (tuple of arrays): A tuple of arrays (z, b(z))
            giving the galaxy bias. If `None`, this tracer won't include
            a term proportional to the matter density contrast.
        mag_bias (tuple of arrays, optional): A tuple of arrays (z, s(z))
            giving the magnification bias as a function of redshift. If
            `None`, the tracer is assumed to not have magnification bias
            terms. Defaults to None.
    """
    def __init__(self, cosmo, has_rsd, dndz, bias, mag_bias=None):
        self._trc = []

        # we need the distance functions at the C layer
        cosmo.compute_distances()

        from scipy.interpolate import interp1d
        z_n, n = _check_array_params(dndz)
        self._dndz = interp1d(z_n, n, bounds_error=False,
                              fill_value=0)

        kernel_d = None
        if bias is not None:  # Has density term
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer
            z_b, b = _check_array_params(bias)
            # Reverse order for increasing a
            t_a = (1./(1+z_b[::-1]), b[::-1])
            self.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a)
        if has_rsd:  # Has RSDs
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer (growth rate)
            z_b, _ = _check_array_params(dndz)
            a_s = 1./(1+z_b[::-1])
            t_a = (a_s, -growth_rate(cosmo, a_s))
            self.add_tracer(cosmo, kernel=kernel_d,
                            transfer_a=t_a, der_bessel=2)
        if mag_bias is not None:  # Has magnification bias
            # Kernel
            chi, w = get_lensing_kernel(cosmo, dndz, mag_bias=mag_bias)
            # Multiply by -2 for magnification
            kernel_m = (chi, -2 * w)
            self.add_tracer(cosmo, kernel=kernel_m,
                            der_bessel=-1, der_angles=1)


class WeakLensingTracer(Tracer):
    """Specific `Tracer` associated to galaxy shape distortions including
    lensing shear and intrinsic alignments within the L-NLA model.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        dndz (tuple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects. The units are
            arbitrary; N(z) will be normalized to unity.
        has_shear (bool): set to `False` if you want to omit the lensing shear
            contribution from this tracer.
        ia_bias (tuple of arrays, optional): A tuple of arrays
            (z, A_IA(z)) giving the intrinsic alignment amplitude A_IA(z).
            If `None`, the tracer is assumped to not have intrinsic
            alignments. Defaults to None.
    """
    def __init__(self, cosmo, dndz, has_shear=True, ia_bias=None):
        self._trc = []

        # we need the distance functions at the C layer
        cosmo.compute_distances()

        from scipy.interpolate import interp1d
        z_n, n = _check_array_params(dndz)
        self._dndz = interp1d(z_n, n, bounds_error=False,
                              fill_value=0)

        if has_shear:
            # Kernel
            kernel_l = get_lensing_kernel(cosmo, dndz)
            self.add_tracer(cosmo, kernel=kernel_l,
                            der_bessel=-1, der_angles=2)
        if ia_bias is not None:  # Has intrinsic alignments
            z_a, tmp_a = _check_array_params(ia_bias)
            # Kernel
            kernel_i = get_density_kernel(cosmo, dndz)
            # Normalize so that A_IA=1
            D = growth_factor(cosmo, 1./(1+z_a))
            # Transfer
            # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
            # and note that we use C_1= 5e-14 from arXiv:0705.0166
            rho_m = lib.cvar.constants.RHO_CRITICAL * cosmo['Omega_m']
            a = - tmp_a * 5e-14 * rho_m / D
            # Reverse order for increasing a
            t_a = (1./(1+z_a[::-1]), a[::-1])
            self.add_tracer(cosmo, kernel=kernel_i, transfer_a=t_a,
                            der_bessel=-1, der_angles=2)


class CMBLensingTracer(Tracer):
    """A Tracer for CMB lensing.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        z_source (float): Redshift of source plane for CMB lensing.
        nsamples (int, optional): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    def __init__(self, cosmo, z_source, n_samples=100):
        self._trc = []

        # we need the distance functions at the C layer
        cosmo.compute_distances()

        kernel = get_kappa_kernel(cosmo, z_source, n_samples)
        self.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)


def _check_returned_tracer(return_val):
    """Wrapper to catch exceptions when tracers are spawned from C.
    """
    if (isinstance(return_val, int)):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr


def _check_array_params(f_arg, arr3=False):
    """Check whether an argument `f_arg` passed into the constructor of
    Tracer() is valid.

    If the argument is set to `None`, it will be replaced with a special array
    that signals to the CCL wrapper that this argument is NULL.
    """
    if f_arg is None:
        # Return empty array if argument is None
        f1 = NoneArr
        f2 = NoneArr
        f3 = NoneArr
    else:
        f1 = np.atleast_1d(np.array(f_arg[0], dtype=float))
        f2 = np.atleast_1d(np.array(f_arg[1], dtype=float))
        if arr3:
            f3 = np.atleast_1d(np.array(f_arg[2], dtype=float))
    if arr3:
        return f1, f2, f3
    else:
        return f1, f2
