from . import ccllib as lib

from .pyutils import check
import numpy as np


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
        self.has_tsp = True

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

    def __del__(self):
        if hasattr(self, 'has_tsp'):
            if self.has_tsp and hasattr(self, 'tsp'):
                lib.f3d_t_free(self.tsp)
