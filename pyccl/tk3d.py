"""
=========================================
Trispectrum container (:mod:`pyccl.tk3d`)
=========================================

Trispectrum container class.
"""

__all__ = ("Tk3D",)

import warnings
from numbers import Real
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import CCLObject, lib
from . import CCLDeprecationWarning, warn_api
from .pyutils import check, _get_spline2d_arrays, _get_spline3d_arrays


class Tk3D(CCLObject):
    r"""Container of splines of arbitrary functions of 2 wavenumber dimensions
    and scale factor.

    Comparison supported: ``==``, ``!=``.


    **Typical Use**

    Store isotropized connected trispectra, i.e. functions of 3 variables of
    the form :math:`T(k_1, k_2, a)`, where  :math:`k_i` are wave vector moduli
    and :math:`a` is the scale factor.

    The function may be provided as a 3-D array (one dimension per variable)
    or as two 2-D arrays corresponding to functions :math:`f_i(k,a)` such that

    .. math::

        T(k_1, k_2, a) = f_1(k_1, a) \, f_2(k_2, a).


    Typical usage includes:

    * Storing perturbation theory or halo model isotropized connected
      trispectra of the form:

      .. math::

          \bar{T}_{abcd}(k_1, k_2, a) = \int \frac{{\rm d}\varphi_1}{2\pi}
          \int \frac{{\rm d}\varphi_2}{2\pi}
          T_{abcd}({\bf k_1}, -{\bf k_1}, {\bf k_2}, -{\bf k_2}),

      where :math:`{\bf k}_i \equiv k_i(\cos \varphi_i, \sin \varphi_i,0)`,
      and :math:`T_{abcd}({\bf k}_a, {\bf k}_b, {\bf k}_c, {\bf k}_d)` is the
      connected trispectrum of fields :math:`\{ a,b,c,d \}`.

    * Storing the kernel for super-sample covariance calculations as a product
      of the responses of power spectra to long-wavelength overdensity modes
      :math:`\delta_L`:

    .. math::

        \bar{T}_{abcd}(k_1, k_2,a)=
        \frac{\partial P_{ab}(k_1, a)}{\partial \delta_L} \,
        \frac{\partial P_{cd}(k_2,a)}{\partial \delta_L}.

    Similarly to :class:`~pyccl.pk2d.Pk2D` objects, :class:`~Tk3D` objects may
    be used to construct the non-Gaussian covariance of angular power spectra
    via Limber integration.


    Parameters
    ----------
    a_arr : ndarray (na,)
        Monotonically increasing array of scale factor.
    lk_arr : ndarray (nk,)
        Natural logarithm of the wavenumber (in :math:`\rm Mpc^{-1}`).
    tkk_arr : ndarray (na, nk, nk)
        Trispectrum. The array is interpolated with a bicubic spline in the 2-D
        space defined by vectors :math:`(k_1, k_2)`, and linearly along the
        dimension of :math:`a`. Make sure to sufficiently sample any difficult
        regions.
    pk1_arr, pk2_arr : ndarray (na, nk)
        Array of the function which makes up a factorizable trispectrum, i.e.
        :math:`:math:`T(k_1, k_2, a) = f_1(k_1, a) \, f_2(k_2, a)`. Ignored if
        `tkk_arr` is provided.
    is_logt
        Whether the arrays hold the trispectrum in linear- or log-scale.
    extrap_order_lok, extrap_order_hik :  {0, 1, 2}
        Extrapolation order when calling the trispectrum beyond the
        interpolation boundaries in :math:`k`. Extrapolated in linear- or
        log-scale, depending on `is_logt`.
    """
    from .base.repr_ import build_string_Tk3D as __repr__
    tsp: lib.f3d_t
    """The associated C-level f3d struct."""

    @warn_api(reorder=['extrap_order_lok', 'extrap_order_hik', 'is_logt'])
    def __init__(
            self,
            *,
            a_arr: NDArray[Real],
            lk_arr: NDArray[Real],
            tkk_arr: Optional[NDArray[Real]] = None,
            pk1_arr: Optional[NDArray[Real]] = None,
            pk2_arr: Optional[NDArray[Real]] = None,
            is_logt: bool = True,
            extrap_order_lok: Literal[0, 1] = 1,
            extrap_order_hik: Literal[0, 1] = 1
    ):
        if not (np.diff(a_arr) > 0).all():
            raise ValueError("a_arr must be monotonically increasing")
        if not np.all(lk_arr[1:]-lk_arr[:-1] > 0):
            raise ValueError("lk_arr must be monotonically increasing")

        if extrap_order_hik not in (0, 1) or extrap_order_lok not in (0, 1):
            raise ValueError("extrap_order must be either 0 or 1.")

        na, nk = len(a_arr), len(lk_arr)
        status = 0
        if tkk_arr is None:
            if pk2_arr is None:
                pk2_arr = pk1_arr
            if not pk1_arr.shape == pk2_arr.shape == (na, nk):
                raise ValueError("Shape mismatch of input arrays.")

            self.tsp, status = lib.tk3d_new_factorizable(lk_arr, a_arr,
                                                         pk1_arr.flatten(),
                                                         pk2_arr.flatten(),
                                                         int(extrap_order_lok),
                                                         int(extrap_order_lok),
                                                         int(is_logt), status)
        else:
            if tkk_arr.shape != (na, nk, nk):
                raise ValueError("Shape mismatch of input arrays.")

            self.tsp, status = lib.tk3d_new_from_arrays(lk_arr, a_arr,
                                                        tkk_arr.flatten(),
                                                        int(extrap_order_lok),
                                                        int(extrap_order_lok),
                                                        int(is_logt), status)
        check(status)

    @property
    def has_tsp(self) -> bool:  # TODO: Remove in CCLv3 and also from repr.
        """
        .. deprecated:: 2.8.0
        """
        return 'tsp' in vars(self)

    @property
    def extrap_order_lok(self) -> int:
        return self.tsp.extrap_order_lok if self else None  # TODO: No if (v3).

    @property
    def extrap_order_hik(self) -> int:
        return self.tsp.extrap_order_hik if self else None  # TODO: No if (v3)

    def __eq__(self, other):
        # Check object id.
        if self is other:
            return True
        # Check the object class.
        if type(self) is not type(other):
            return False
        # If the objects contain no data, return early.
        if not (self or other):
            return True
        # If one is factorizable and the other one is not, return early.
        if self.tsp.is_product ^ other.tsp.is_product:
            return False
        # Check extrapolation orders.
        if not (self.extrap_order_lok == other.extrap_order_lok
                and self.extrap_order_hik == other.extrap_order_hik):
            return False
        # Check the individual splines.
        a1, lk11, lk12, tk1 = self.get_spline_arrays()
        a2, lk21, lk22, tk2 = other.get_spline_arrays()
        return ((a1 == a2).all()
                and (lk11 == lk21).all() and (lk21 == lk22).all()
                and np.array_equal(tk1, tk2))

    def __hash__(self):
        return hash(repr(self))

    def eval(self, k, a):
        """
        .. deprecated:: 2.8.0

            Use :meth:`Pk2D.__call__`.
        """
        warnings.warn("Tk3D.eval is deprecated. Simply call the object "
                      "itself.", category=CCLDeprecationWarning)
        return self(k, a)

    def __call__(
            self,
            k: Union[Real, NDArray[Real]],
            a: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Evaluate the trispectrum.

        Arguments
        ---------
        k : array_like (nk,)
            Wavenumber (in :math:`\rm Mpc^{-1}`).
        a : array_like (na,)
            Scale factor. If it is out of the interpolated range, constant
            extrapolation is performed.

        Returns
        -------
        array_like (na, nk, nk)
            Evaluated trispectrum.
        """
        a_use = np.atleast_1d(a).astype(float)
        k_use = np.atleast_1d(k).astype(float)
        lk_use = np.log(k_use)

        nk = k_use.size
        out = np.zeros([len(a_use), nk, nk])
        status = 0
        for ia, aa in enumerate(a_use):
            f, status = lib.tk3d_eval_multi(self.tsp, lk_use,
                                            aa, nk*nk, status)
            check(status)
            out[ia] = f.reshape([nk, nk])

        if np.ndim(k) == 0:
            out = np.squeeze(np.squeeze(out, axis=-1), axis=-1)
        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        return out

    def __del__(self):
        if hasattr(self, 'has_tsp'):  # TODO: Remove ifs in CCLv3.
            if self.has_tsp and hasattr(self, 'tsp'):
                lib.f3d_t_free(self.tsp)

    def __bool__(self):  # TODO: Remove in CCLv3.
        return self.has_tsp

    def get_spline_arrays(
            self
    ) -> Tuple[
            NDArray[float],
            NDArray[float],
            Union[
                NDArray[float],
                List[NDArray[float]]]]:
        r"""Get the arrays used to construct the internal splines.

        Returns
        -------
        a_arr : ndarray (na,)
            Scale factor.
        lk_arr : ndarray (nk,)
        out : ndarray (na, nk, nk) | [ndarray (na, nk), ndarray (na, nk)]
            The trispectrum :math:`T(k_1, k_2, a)`
            or its two factors :math:`f_i(k_1, a)`.
        """
        if not self:
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
