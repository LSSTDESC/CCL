from . import ccllib as lib
from .core import check
from .background import comoving_radial_distance, growth_rate, \
    growth_factor, scale_factor_of_chi
from .pyutils import _check_array_params, NoneArr, _vectorize_fn6
import numpy as np


def _Sig_MG(cosmo, a, k=None):
    """Redshift-dependent modification to Poisson equation for massless
    particles under modified gravity.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        k (float or array_like): Wavenumber for scale

    Returns:
        float or array_like: Modification to Poisson equation under \
            modified gravity at scale factor a. \
            Sig_MG is assumed to be proportional to Omega_Lambda(z), \
            see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.
    """
    return _vectorize_fn6(lib.Sig_MG, lib.Sig_MG_vec, cosmo, a, k)


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
    z_n, n = _check_array_params(dndz, 'dndz')
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
    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias, 'mag_bias')

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
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a and  k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * (bias_transfer_k[1] *
                                mg_transfer[2]).T).T)
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a but not k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * mg_transfer[2].T).T)
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on k but not a
        elif ((bias_transfer_a is None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_k[1] * mg_transfer[2]))
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
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
            cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
                transform redshifts into distances.
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
        if isinstance(z, float):
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
        check(status)
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
        # Sometimes lib is freed before some Tracers, in which case, this
        # doesn't work.
        # So just check that lib.cl_tracer_t_free is still a real function.
        if hasattr(self, '_trc') and lib.cl_tracer_t_free is not None:
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
        z_n, n = _check_array_params(dndz, 'dndz')
        self._dndz = interp1d(z_n, n, bounds_error=False,
                              fill_value=0)

        kernel_d = None
        if bias is not None:  # Has density term
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer
            z_b, b = _check_array_params(bias, 'bias')
            # Reverse order for increasing a
            t_a = (1./(1+z_b[::-1]), b[::-1])
            self.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a)

        if has_rsd:  # Has RSDs
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer (growth rate)
            z_b, _ = _check_array_params(dndz, 'dndz')
            a_s = 1./(1+z_b[::-1])
            t_a = (a_s, -growth_rate(cosmo, a_s))
            self.add_tracer(cosmo, kernel=kernel_d,
                            transfer_a=t_a, der_bessel=2)
        if mag_bias is not None:  # Has magnification bias
            # Kernel
            chi, w = get_lensing_kernel(cosmo, dndz, mag_bias=mag_bias)
            # Multiply by -2 for magnification
            kernel_m = (chi, -2 * w)
            if (cosmo['sigma_0'] == 0):
                # GR case
                self.add_tracer(cosmo, kernel=kernel_m,
                                der_bessel=-1, der_angles=1)
            else:
                # MG case
                z_b, _ = _check_array_params(dndz, 'dndz')
                self._MG_add_tracer(cosmo, kernel_m, z_b,
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
        use_A_ia (bool): set to True to use the conventional IA
            normalization. Set to False to use the raw input amplitude,
            which will usually be 1 for use with PT IA modeling.
            Defaults to True.
    """
    def __init__(self, cosmo, dndz, has_shear=True, ia_bias=None,
                 use_A_ia=True):
        self._trc = []

        # we need the distance functions at the C layer
        cosmo.compute_distances()

        from scipy.interpolate import interp1d
        z_n, n = _check_array_params(dndz, 'dndz')
        self._dndz = interp1d(z_n, n, bounds_error=False,
                              fill_value=0)

        if has_shear:
            kernel_l = get_lensing_kernel(cosmo, dndz)
            if (cosmo['sigma_0'] == 0):
                # GR case
                self.add_tracer(cosmo, kernel=kernel_l,
                                der_bessel=-1, der_angles=2)
            else:
                # MG case
                self._MG_add_tracer(cosmo, kernel_l, z_n,
                                    der_bessel=-1, der_angles=2)
        if ia_bias is not None:  # Has intrinsic alignments
            z_a, tmp_a = _check_array_params(ia_bias, 'ia_bias')
            # Kernel
            kernel_i = get_density_kernel(cosmo, dndz)
            if use_A_ia:
                # Normalize so that A_IA=1
                D = growth_factor(cosmo, 1./(1+z_a))
                # Transfer
                # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
                # and note that we use C_1= 5e-14 from arXiv:0705.0166
                rho_m = lib.cvar.constants.RHO_CRITICAL * cosmo['Omega_m']
                a = - tmp_a * 5e-14 * rho_m / D
            else:
                # use the raw input normalization. Normally, this will be 1
                # to allow nonlinear PT IA models, where normalization is
                # already applied to the power spectrum.
                a = tmp_a
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
        if (cosmo['sigma_0'] == 0):
            self.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
        else:
            self._MG_add_tracer(cosmo, kernel, z_source,
                                der_bessel=-1, der_angles=1)


class tSZTracer(Tracer):
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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        zmax (float): maximum redshift up to which we define the
            kernel.
        n_chi (float): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """
    def __init__(self, cosmo, z_max=6., n_chi=1024):
        self.chi_max = comoving_radial_distance(cosmo, 1./(1+z_max))
        chi_arr = np.linspace(0, self.chi_max, n_chi)
        a_arr = scale_factor_of_chi(cosmo, chi_arr)
        # This is \sigma_T / (m_e * c^2)
        prefac = 4.01710079e-06
        w_arr = prefac * a_arr

        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))


def _check_returned_tracer(return_val):
    """Wrapper to catch exceptions when tracers are spawned from C.
    """
    if (isinstance(return_val, int)):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr
