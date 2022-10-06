from . import ccllib as lib

from .pyutils import check, _get_spline2d_arrays, _get_spline3d_arrays
import numpy as np

from . import core
import warnings
from .errors import CCLWarning


class Tk3D(object):
    """A container for \"isotropized\" connected trispectra relevant for
    covariance matrix calculations. I.e. functions of 3 variables of the
    form :math:`T(k_1,k_2,a)`, where :math:`k_i` are wave vector moduli
    and :math:`a` is the scale factor. This function can be provided as
    a 3D array (one dimension per variable), or as two 2D arrays
    corresponding to functions :math:`f_i(k,a)` such that

    .. math::
        T(k_1,k_2,a) = f_1(k_1,a)\\,f_2(k_2,a)

    Typical uses for these objects will be:

    * To store perturbation theory or halo model \"isotropized\"
      connected trispectra of the form:

      .. math::
          \\bar{T}_{abcd}(k_1, k_2, a) = \\int \\frac{d\\varphi_1}{2\\pi}
          \\int \\frac{d\\varphi_2}{2\\pi}
          T_{abcd}({\\bf k_1},-{\\bf k_1},{\\bf k_2},-{\\bf k_2}),

      where :math:`{\bf k}_i\\equiv k_i(\\cos\\varphi_i,\\sin\\varphi_i,0)`,
      and :math:`T_{abcd}({\\bf k}_a,{\\bf k}_b,{\\bf k}_c,{\\bf k}_d)` is
      the connected trispectrum of fields :math:`\\{a,b,c,d\\}`.

    * To store the kernel for super-sample covariance calculations as a
      product of the responses of power spectra to long-wavelength
      overdensity modes :math:`\\delta_L`:

    .. math::
        \\bar{T}_{abcd}(k_1,k_2,a)=
        \\frac{\\partial P_{ab}(k_1,a)}{\\partial\\delta_L}\\,
        \\frac{\\partial P_{cd}(k_2,a)}{\\partial\\delta_L}.

    These objects can then be used, analogously to
    :class:`~pyccl.pk2d.Pk2D` objects, to construct the non-Gaussian
    covariance of angular power spectra via Limber integration.

    Args:
        a_arr (array): an array holding values of the scale factor. Note
            that the trispectrum will be extrapolated as constant on
            values of the scale factor outside those held by this array.
        lk_arr (array): an array holding values of the natural logarithm
            of the wavenumber (in units of Mpc^-1).
        tkk_arr (array): a 3D array with shape `[na,nk,nk]`, where `na`
            and `nk` are the sizes of `a_arr` and `lk_arr` respectively.
            This array should contain the values of the trispectrum
            at the values of scale factor and wavenumber held by `a_arr`
            and `lk_arr`. The array can hold the values of the natural
            logarithm of the trispectrum, depending on the value of
            `is_logt`. If `tkk_arr` is `None`, then it is assumed that
            the trispectrum can be factorized as described above, and
            the two functions :math:`f_i(k_i,a)` are described by
            `pk1_arr` and `pk2_arr`. You are responsible of making sure
            all these arrays are sufficiently well sampled (i.e. the
            resolution of `a_arr` and `lk_arr` is high enough to sample
            the main features in the trispectrum). For reference, CCL
            will use bicubic interpolation to evaluate the trispectrum
            in the 2D space of wavenumbers :math:`(k_1,k_2)` at a fixed
            scale factor, and will use linear interpolation in the
            scale factor dimension.
        pk1_arr (array): a 2D array with shape `[na,nk]` describing the
            first function :math:`f_1(k,a)` that makes up a factorizable
            trispectrum :math:`T(k_1,k_2,a)=f_1(k_1,a)f_2(k_2,a)`.
            `pk1_arr` and `pk2_arr` are ignored if `tkk_arr` is not
            `None`.
        pk2_arr (array): a 2D array with shape `[na,nk]` describing the
            second factor :math:`f_2(k,a)` for a factorizable trispectrum.
        extrap_order_lok (int): extrapolation order to be used on k-values
            below the minimum of the splines (use 0 or 1). Note that
            the extrapolation will be done in either
            :math:`\\log(T(k_1,k_2,a)` or :math:`T(k_1,k_2,a)`,
            depending on the value of `is_logt`.
        extrap_order_hik (int): same as `extrap_order_lok` for
            k-values above the maximum of the splines.
        is_logt (boolean): if True, `tkk_arr`/`pk1_arr`/`pk2_arr` hold the
            natural logarithm of the trispectrum (or its factors).
            Otherwise, the true values of the corresponding quantities are
            expected. Note that arrays will be interpolated in log space
            if `is_logt` is set to `True`.
    """
    def __init__(self, a_arr, lk_arr, tkk_arr=None,
                 pk1_arr=None, pk2_arr=None, extrap_order_lok=1,
                 extrap_order_hik=1, is_logt=True):
        na = len(a_arr)
        nk = len(lk_arr)

        if not np.all(a_arr[1:]-a_arr[:-1] > 0):
            raise ValueError("`a_arr` must be strictly increasing")

        if not np.all(lk_arr[1:]-lk_arr[:-1] > 0):
            raise ValueError("`lk_arr` must be strictly increasing")

        if ((extrap_order_hik not in (0, 1)) or
                (extrap_order_lok not in (0, 1))):
            raise ValueError("Only constant or linear extrapolation in "
                             "log(k) is possible (`extrap_order_hik` or "
                             "`extrap_order_lok` must be 0 or 1).")
        status = 0

        if tkk_arr is None:
            if pk2_arr is None:
                pk2_arr = pk1_arr
            if (pk1_arr is None) or (pk2_arr is None):
                raise ValueError("If trispectrum is factorizable "
                                 "you must provide the two factors")
            if (pk1_arr.shape != (na, nk)) or (pk2_arr.shape != (na, nk)):
                raise ValueError("Input trispectrum factor "
                                 "shapes are wrong")

            self.tsp, status = lib.tk3d_new_factorizable(lk_arr, a_arr,
                                                         pk1_arr.flatten(),
                                                         pk2_arr.flatten(),
                                                         int(extrap_order_lok),
                                                         int(extrap_order_lok),
                                                         int(is_logt), status)
        else:
            if tkk_arr.shape != (na, nk, nk):
                raise ValueError("Input trispectrum shape is wrong")

            self.tsp, status = lib.tk3d_new_from_arrays(lk_arr, a_arr,
                                                        tkk_arr.flatten(),
                                                        int(extrap_order_lok),
                                                        int(extrap_order_lok),
                                                        int(is_logt), status)
        check(status)

    @property
    def has_tsp(self):
        return 'tsp' in vars(self)

    def eval(self, k, a):
        """Evaluate trispectrum. If `k` is a 1D array with size `nk`, the
        output `out` will be a 2D array with shape `[nk,nk]` holding
        `out[i,j] = T(k[j],k[i],a)`, where `T` is the trispectrum function
        held by this `Tk3D` object.

        Args:
            k (float or array_like): wavenumber value(s) in units of Mpc^-1.
            a (float): value of the scale factor

        Returns:
            float or array_like: value(s) of the trispectrum.
        """
        status = 0

        if np.ndim(a) != 0:
            raise TypeError("a must be a floating point number")

        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = lib.tk3d_eval_single(self.tsp, np.log(k), a, status)
        else:
            k_use = np.atleast_1d(k)
            nk = k_use.size
            f, status = lib.tk3d_eval_multi(self.tsp, np.log(k_use),
                                            a, nk*nk, status)
            f = f.reshape([nk, nk])
        check(status)
        return f

    def __call__(self, k, a):
        """Callable vectorized instance."""
        out = np.array([self.eval(k, aa)
                        for aa in np.atleast_1d(a).astype(float)])
        return out.squeeze()[()]

    def __del__(self):
        if hasattr(self, 'has_tsp'):
            if self.has_tsp and hasattr(self, 'tsp'):
                lib.f3d_t_free(self.tsp)

    def __bool__(self):
        return self.has_tsp

    def get_spline_arrays(self):
        """Get the spline data arrays.

        Returns:
            a_arr (1D ``numpy.ndarray``):
                Array of scale factors.
            lk_arr1, lk_arr2 (1D ``numpy.ndarray``):
                Arrays of :math:``ln(k)``.
            out (list of ``numpy.ndarray``):
                The trispectrum T(k1, k2, z) or its factors f(k1, z), f(k2, z).
        """
        if not self.has_tsp:
            raise ValueError("Tk3D object does not have data.")

        out = []
        if self.tsp.is_product:
            a_arr, lk_arr1, pk_arr1 = _get_spline2d_arrays(self.tsp.fka_1.fka)
            _, lk_arr2, pk_arr2 = _get_spline2d_arrays(self.tsp.fka_2.fka)
            out.append(pk_arr1)
            out.append(pk_arr2)
        else:
            status = 0
            a_arr, status = lib.get_array(self.tsp.a_arr, self.tsp.na, status)
            check(status)
            lk_arr1, lk_arr2, tkka_arr = _get_spline3d_arrays(self.tsp.tkka,
                                                              self.tsp.na)
            out.append(tkka_arr)

        if self.tsp.is_log:
            # exponentiate in-place
            [np.exp(tk, out=tk) for tk in out]

        return a_arr, lk_arr1, lk_arr2, out


def Tk3D_SSC_Terasawa22(cosmo, deltah=0.02,
                        lk_arr=None, a_arr=None,
                        extrap_order_lok=1, extrap_order_hik=1,
                        use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Currently this
    function only applicable to matter power spectrum in flat
    cosmology. Each response is calculated using the method
    developed in Terasawa et al. 2022 (arXiv:2205.10339v2) as:

    .. math::
        \\frac{\\partial P_{mm}(k)}{\\partial\\delta_L} =
        \\left(1 + \\frac{26}{21}T_{h}(k)
        -\\frac{1}{3}\\frac{d\\log P_{mm}(k)}{d\\log k}\\right)
        P_{mm}(k),

    where the :math:`T_{h}(k)` is the normalized growth response to
    the Hubble parameter defined as
    :math:`T_{h}(k) = \\frac{d\\log P_{mm}(k)}{dh}/(2\\frac{d\\log D}{dh})`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        deltah (float): the variation of h to compute T_{h}(k) by
            the two-sided numerical derivative method.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """

    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    extra_parameters = {"camb": {"halofit_version": "original", }}

# set h-modified cosmology to take finite differencing
    hp = h + deltah
    Omega_c_p = np.power((h/hp), 2) * Omega_c  # \Omega_c h^2 is fixed
    Omega_b_p = np.power((h/hp), 2) * Omega_b  # \Omega_b h^2 is fixed

    hm = h - deltah
    Omega_c_m = np.power((h/hm), 2) * Omega_c  # \Omega_c h^2 is fixed
    Omega_b_m = np.power((h/hm), 2) * Omega_b  # \Omega_b h^2 is fixed

    cosmo_hp = core.Cosmology(Omega_c=Omega_c_p, Omega_b=Omega_b_p,
                              h=hp, n_s=n_s, A_s=A_s,
                              transfer_function="boltzmann_camb",
                              matter_power_spectrum="camb",
                              extra_parameters=extra_parameters)

    cosmo_hm = core.Cosmology(Omega_c=Omega_c_m, Omega_b=Omega_b_m,
                              h=hm, n_s=n_s, A_s=A_s,
                              transfer_function="boltzmann_camb",
                              matter_power_spectrum="camb",
                              extra_parameters=extra_parameters)

    # Growth factor
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    # Power spectrum
    cosmo.compute_linear_power()
    cosmo_hp.compute_linear_power()
    cosmo_hm.compute_linear_power()

    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')

    pk2d = cosmo.get_nonlin_power('delta_matter:delta_matter')
    pk2d_hp = cosmo_hp.get_nonlin_power('delta_matter:delta_matter')
    pk2d_hm = cosmo_hm.get_nonlin_power('delta_matter:delta_matter')

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk = np.zeros(nk)
    T_h = np.zeros(nk)

    kmin = 1e-2
    for ia, aa in enumerate(a_arr):

        pk = pk2d.eval(k_use, aa, cosmo)
        pk_hp = pk2d_hp.eval(k_use, aa, cosmo_hp)
        pk_hm = pk2d_hm.eval(k_use, aa, cosmo_hm)

        dpknl = pk2d.eval_dlogpk_dlogk(k_use, aa, cosmo)
        dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo)

        # use linear theory below kmin
        T_h[k_use <= kmin] = 1

        T_h[k_use > kmin] = (np.log(pk_hp[k_use > kmin])
                             - np.log(pk_hm[k_use > kmin]))  \
/ (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))
        # (hp-hm) term is cancelled out

        dpk[k_use <= kmin] = dpklin[k_use <= kmin]
        dpk[k_use > kmin] = dpknl[k_use > kmin]

        dpk12[ia, :] = pk * (1. + (26. / 21.) * T_h - dpk / 3.)

    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d
