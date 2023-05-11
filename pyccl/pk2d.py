"""
============================================
Power spectrum container (:mod:`pyccl.pk2d`)
============================================

Power spectrum container class and parser.
"""
from __future__ import annotations

__all__ = ("Pk2D", "parse_pk2d", "parse_pk",)

import functools
import warnings
from numbers import Real
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import (
    CCLObject, DEFAULT_POWER_SPECTRUM, get_pk_spline_a,
    get_pk_spline_lk, lib, unlock_instance)
from . import CCLWarning, CCLError, CCLDeprecationWarning, warn_api, deprecated
from .pyutils import check, _get_spline1d_arrays, _get_spline2d_arrays

if TYPE_CHECKING:
    from . import Cosmology, SplineParams


# TODO: Remove for CCLv3.
class _Pk2D_descriptor:
    """Descriptor to deprecate usage of `Pk2D` methods as class methods."""
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, base):
        if instance is None:
            warnings.warn("Use of the power spectrum as an argument "
                          f"is deprecated in {self.func.__name__}. "
                          "Use the instance method instead.",
                          CCLDeprecationWarning)
            this = base
        else:
            this = instance

        @functools.wraps(self.func)
        def new_func(*args, **kwargs):
            return self.func(this, *args, **kwargs)

        return new_func


class Pk2D(CCLObject):
    r"""Container of splines of arbitrary functions of wavenumber and scale
    factor.

    Supported binary operations: ``+``, ``-``, ``*``, ``/``, ``**``
    (and in-place) between :class:`~Pk2D` objects and numbers. Exponentiation
    is only supported for numbers. For mismatch of interpolated ranges, the one
    with the narrowest support is output. Comparison supported: ``==``, ``!=``.
    Membership supported: ``in`` (checks if range of one is contained in the
    range of the other).


    Parameters
    ----------
    a_arr : ndarray (na,)
        Monotonically increasing array of scale factor.
        Ignored if `pk_func` is provided.
    lk_arr : ndarray (nk,)
        Natural logarithm of the wavenumber (in :math:`\rm Mpc^{-1}`).
        Ignored if `pk_func` is provided.
    pk_arr : ndarray (na, nk) or (na*nk)
        Power spectrum (in :math:`\rm Mpc^3`). If flattened, it is reshaped to
        `(na, nk)`. The array is interpolated with a bicubic spline. Make sure
        to sufficiently sample any difficult regions.
        Ignored if `pk_func` is provided.
    pkfunc
        Function that returns the power spectrum.

        .. deprecated:: 2.8.0

            Use :meth:`~Pk2D.from_function`.

    cosmo
        Used to determine the sampling rate of `a_arr` and `lk_arr`.

        .. deprecated:: 2.8.0

            Use :meth:`~Pk2D.from_function`.

    is_logp
        Whether `pk_arr` holds the power spectrum in linear- or log-scale.
    extrap_order_lok, extrap_order_hik
        Extrapolation order when calling the power spectrum beyond the
        interpolation boundaries in :math:`k`. Extrapolated in linear- or
        log-scale, depending on `is_logp`.
    empty
        Initialize an empty object.

        .. deprecated:: 2.8.0
    """
    from ._core.repr_ import build_string_Pk2D as __repr__
    psp: lib.f2d_t
    """The associated C-level `f2d` struct."""

    @warn_api(reorder=["pkfunc", "a_arr", "lk_arr", "pk_arr", "is_logp",
                       "extrap_order_lok", "extrap_order_hik", "cosmo"])
    def __init__(
            self,
            *,
            a_arr: Optional[NDArray[Real]] = None,  # TODO: Required in CCLv3.
            lk_arr: Optional[NDArray[Real]] = None,  # TODO: Required in CCLv3.
            pk_arr: Optional[NDArray[Real]] = None,  # TODO: Required in CCLv3.
            pkfunc: Optional[
                Callable[
                    [NDArray[Real], NDArray[Real]],
                    NDArray[Real]]
            ] = None,  # TODO: deprecate in CCLv3
            cosmo: Optional[Cosmology] = None,  # TODO: deprecate in CCLv3.
            is_logp: bool = True,
            extrap_order_lok: Literal[0, 1, 2] = 1,
            extrap_order_hik: Literal[0, 1, 2] = 2,
            empty: bool = False  # TODO: deprecate in CCLv3.
    ):
        if empty:
            warnings.warn("The creation of empty Pk2D objects is now "
                          "deprecated. If you want an empty object, use "
                          "`Pk2D.__new__(Pk2D)`.",
                          category=CCLDeprecationWarning)
            return

        if pkfunc is None:  # Initialize power spectrum from 2D array
            # Make sure input makes sense
            if (a_arr is None) or (lk_arr is None) or (pk_arr is None):
                raise ValueError("If you do not provide a function, "
                                 "you must provide arrays")

            # Check that `a` is a monotonically increasing array.
            if not (np.diff(a_arr) > 0).all():
                raise ValueError("a_arr must be monotonically increasing")

            pkflat = pk_arr.flatten()
            # Check dimensions make sense
            if len(pkflat) != len(a_arr)*len(lk_arr):
                raise ValueError("Shape mismatch of input arrays.")
        else:  # Initialize power spectrum from function
            warnings.warn("The use of a function when initialising a ``Pk2D`` "
                          "object is deprecated. Use `Pk2D.from_function`.",
                          CCLDeprecationWarning)
            # Set k and a sampling from CCL parameters
            a_arr = get_pk_spline_a(cosmo=cosmo)
            lk_arr = get_pk_spline_lk(cosmo=cosmo)

            # Compute power spectrum on 2D grid
            pkflat = np.array([pkfunc(k=np.exp(lk_arr), a=a)
                               for a in a_arr]).flatten()

        status = 0
        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status)

    @property
    def has_psp(self):  # TODO: Remove for CCLv3.
        """
        .. deprecated:: 2.8.0
        """
        return 'psp' in vars(self)

    @property
    def extrap_order_lok(self):
        return self.psp.extrap_order_lok if self else None  # TODO: No if (v3).

    @property
    def extrap_order_hik(self):
        return self.psp.extrap_order_hik if self else None  # TODO: No if (v3).

    @classmethod
    def from_function(
            cls,
            pkfunc: Callable[[NDArray[Real], NDArray[Real]], NDArray[Real]],
            *,
            is_logp: bool = True,
            spline_params: Optional[
                Union[SplineParams,
                      lib.spline_params
                      ]] = None,
            extrap_order_lok: Literal[0, 1, 2] = 1,
            extrap_order_hik: Literal[0, 1, 2] = 2
    ) -> Pk2D:
        """Create a :class:`~Pk2D` object from a function.

        Arguments
        ---------
        pkfunc
            Function that returns the power spectrum. Sampling rates determined
            by `spline_params`.
        is_logp
            Whether `pkfunc` outputs the power spectrum in linear- or
            log-scale.
        spline_params
            Spline parameters. Used to determine sampling rates in scale factor
            and wavenumber. For the spline parameters stored in
            :class:`~Cosmology` objects, use ``cosmo._spline_params``
            (although not recommended). The default uses the global spline
            parameters state at the moment of initialization.
        extrap_order_lok, extrap_order_hik :  {0, 1, 2}
            Extrapolation order when calling the power spectrum beyond the
            interpolation boundaries. Extrapolated in linear- or log-scale,
            depending on `is_logp`.

        Returns
        -------

            Power spectrum.
        """
        # Set k and a sampling from CCL parameters
        if spline_params is None:
            from . import spline_params
        a_arr = get_pk_spline_a(spline_params=spline_params)
        lk_arr = get_pk_spline_lk(spline_params=spline_params)

        # Compute power spectrum on 2D grid
        pk_arr = np.array([pkfunc(k=np.exp(lk_arr), a=a) for a in a_arr])

        return cls(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                   is_logp=is_logp, extrap_order_lok=extrap_order_lok,
                   extrap_order_hik=extrap_order_hik)

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
        # Check extrapolation orders.
        if not (self.extrap_order_lok == other.extrap_order_lok
                and self.extrap_order_hik == other.extrap_order_hik):
            return False
        # Check the individual splines.
        a1, lk1, pk1 = self.get_spline_arrays()
        a2, lk2, pk2 = other.get_spline_arrays()
        return ((a1 == a2).all() and (lk1 == lk2).all()
                and np.array_equal(pk1, pk2))

    def __hash__(self):
        return hash(repr(self))

    @classmethod
    def from_model(cls, cosmo: Cosmology, model: str) -> Pk2D:
        """Create a :class:`~Pk2D` object from a model.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        model
            Model name. Of the models listed in
            :class:`~pyccl.cosmology.TransferFunctions` and
            :class:`~pyccl.cosmology.MatterPowerSpectra`, these are available:

            * ``'bbks'``,
            * ``'eisenstein_hu'``,
            * ``'eisenstein_hu_nowiggles'``,
            * ``'emu'``.

        Returns
        -------

            Power spectrum.
        """  # TODO: Update docstring after `pspec` PR is merged.

        pk2d = Pk2D.__new__(cls)
        status = 0
        if model == 'bbks':
            cosmo.compute_growth()
            ret = lib.compute_linpower_bbks(cosmo.cosmo, status)
        elif model == 'eisenstein_hu':
            cosmo.compute_growth()
            ret = lib.compute_linpower_eh(cosmo.cosmo, 1, status)
        elif model == 'eisenstein_hu_nowiggles':
            cosmo.compute_growth()
            ret = lib.compute_linpower_eh(cosmo.cosmo, 0, status)
        elif model == 'emu':
            ret = lib.compute_power_emu(cosmo.cosmo, status)
        else:
            raise ValueError(f"Invalid model {model}.")

        if np.ndim(ret) == 0:
            status = ret
        else:
            with pk2d.unlock():
                pk2d.psp, status = ret

        cosmo.check(status)
        return pk2d

    @classmethod
    @deprecated(new_api=from_model.__func__)
    def pk_from_model(cls, cosmo: Cosmology, model: str) -> Pk2D:
        """.. deprecated:: 2.8.0 : Use :meth:`~Pk2D.from_model`.
        """
        return cls.from_model(cosmo, model)

    @_Pk2D_descriptor
    @warn_api
    def apply_halofit(
            self,
            cosmo: Cosmology,
            *,
            pk_linear: Optional[Pk2D] = None
    ) -> Pk2D:
        """Apply the "HALOFIT" transformation on power spectrum.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        pk_linear
            Power spectrum to transform.

            .. deprecated:: 2.5.0

                Use the instance method of the object.
        """
        if pk_linear is None:
            pk_linear = self

        if cosmo["wa"] != 0:
            # HALOFIT translates (w0, wa) to a w0_eff. This requires computing
            # the comoving distance to the CMB, which requires the background
            # splines being sampled to sufficiently high redshifts.
            cosmo.compute_distances()
            _, a = _get_spline1d_arrays(cosmo.cosmo.data.achi)
            if min(a) > 1/(1 + 3000):
                raise CCLError("Comoving distance spline does not cover "
                               "sufficiently high redshifts for HALOFIT. "
                               "HALOFIT translates (w0, wa) to a w0_eff. This "
                               "requires computing the comoving distance to "
                               "the CMB, which requires the background "
                               "splines being sampled to sufficiently high "
                               "redshifts. If using the calculator mode, "
                               "check the support of the background data.")

        pk2d = Pk2D.__new__(Pk2D)
        status = 0
        ret = lib.apply_halofit(cosmo.cosmo, pk_linear.psp, status)
        if np.ndim(ret) == 0:
            status = ret
        else:
            with pk2d.unlock():
                pk2d.psp, status = ret
        cosmo.check(status)
        return pk2d

    def eval(self, k, a, cosmo=None, *, derivative=False):
        """.. deprecated:: 2.5.0 Use :meth:`~Pk2D.__call__`."""
        warnings.warn("Pk2D.eval is deprecated. Simply call the object "
                      "itself.", category=CCLDeprecationWarning)
        return self(k, a, cosmo=cosmo, derivative=derivative)

    def eval_dlogpk_dlogk(self, k, a, cosmo):
        """
        .. deprecated:: 2.5.0

            Use :meth:`~Pk2D.__call__` with ``derivative=True``.
        """
        warnings.warn("Pk2D.eval_dlogpk_dlogk is deprecated. Simply call "
                      "the object itself with `derivative=True`.",
                      category=CCLDeprecationWarning)
        return self(k, a, cosmo=cosmo, derivative=True)

    def __call__(
            self,
            k: Union[Real, NDArray[Real]],
            a: Union[Real, NDArray[Real]],
            cosmo: Optional[Cosmology] = None,
            *,
            derivative: bool = False
    ) -> Union[float, NDArray[float]]:
        r"""Evaluate the power spectrum or its logarithmic derivative.

        Arguments
        ---------
        k : array_like (nk,)
            Wavenumber (in :math:`\rm Mpc^{-1}`).
        a : array_like (na,)
            Scale factor.
        cosmo
            Cosmological parameters. Used to evaluate the power spectrum
            outside of the interpolation range in `a` (thorugh the linear
            growth). If None, out-of-bounds queries raise an exception.
        derivative
            Whether to evaluare the logarithmic derivative,
            :math:`\frac{{\rm d}\log P(k)}{{\rm d}\log k}`.

        Returns
        -------
        array_like (na, nk)
            Evaluated power spectrum.
        """
        # determine if logarithmic derivative is needed
        if not derivative:
            eval_func = lib.pk2d_eval_multi
        else:
            eval_func = lib.pk2d_der_eval_multi

        # handle scale factor extrapolation
        if cosmo is None:
            cosmo = self.__call__._cosmo
            self.psp.extrap_linear_growth = 404  # flag no extrapolation
        else:
            cosmo.compute_growth()  # growth factors for extrapolation
            self.psp.extrap_linear_growth = 401  # flag extrapolation

        a_use = np.atleast_1d(a).astype(float)
        k_use = np.atleast_1d(k).astype(float)
        lk_use = np.log(k_use)

        status = 0
        out = np.zeros([len(a_use), len(k_use)])
        for ia, aa in enumerate(a_use):
            f, status = eval_func(self.psp, lk_use, aa,
                                  cosmo.cosmo, k_use.size, status)

            # Catch scale factor extrapolation bounds error.
            if status == lib.CCL_ERROR_SPLINE_EV:
                raise ValueError(
                    "Pk2D evaluation scale factor is outside of the "
                    "interpolation range. To extrapolate, pass a Cosmology.")
            check(status, cosmo)
            out[ia] = f

        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        return out

    # Save a dummy cosmology as an attribute of the `__call__` method so we
    # don't have to initialize one every time no `cosmo` is passed. This is
    # gentle with memory too, as `free` does not work for an empty cosmology.
    __call__._cosmo = type("Dummy", (object,), {"cosmo": lib.cosmology()})()

    def copy(self) -> Pk2D:
        """Return a copy of this Pk2D object."""
        if not self:
            return Pk2D.__new__(Pk2D)
        return self + 0

    def get_spline_arrays(
            self
    ) -> Tuple(NDArray[float], NDArray[float], NDArray[float]):
        r"""Get the arrays used to construct the internal splines.

        Returns
        -------
        a_arr : ndarray (na,)
            Scale factor.
        lk_arr : ndarray (nk,)
            Natural logarithm of wavenumber (in :math:`\rm Mpc^{-1}`).
        pk_arr : ndarray (na, nk)
            Power spectrum.
        """
        if not self:
            raise ValueError("Pk2D object does not have data.")

        a_arr, lk_arr, pk_arr = _get_spline2d_arrays(self.psp.fka)
        if self.psp.is_log:
            pk_arr = np.exp(pk_arr)
        return a_arr, lk_arr, pk_arr

    def __del__(self):
        """Free memory associated with this Pk2D structure."""
        if self:  # TODO: Remove if in CCLv3.
            lib.f2d_t_free(self.psp)

    def __bool__(self):  # TODO: Remove for CCLv3.
        return self.has_psp

    def __contains__(self, other):
        if (self.psp.lkmin > other.psp.lkmin
                or self.psp.lkmax < other.psp.lkmax
                or self.psp.amin > other.psp.amin
                or self.psp.amax < other.psp.amax):
            return False
        return True

    def _get_binary_operator_arrays(self, other):
        if not (self and other):  # TODO: Remove for CCLv3.
            raise ValueError("Pk2D object does not have data.")
        if self not in other:
            raise ValueError("To avoid extrapolation, the 2nd operand must "
                             "have larger support. Try swapping the operands.")

        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        a_arr_b, lk_arr_b, pk_arr_b = other.get_spline_arrays()
        if not (a_arr_a.size == a_arr_b.size
                and lk_arr_a.size == lk_arr_b.size
                and np.allclose(a_arr_a, a_arr_b)
                and np.allclose(lk_arr_a, lk_arr_b)):
            warnings.warn("Operands defined over different ranges. "
                          "Clipping the result to the narrowest support.",
                          CCLWarning)
            pk_arr_b = other(np.exp(lk_arr_a), a_arr_a)

        return a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b

    def __add__(self, other):
        if isinstance(other, (float, int)):
            a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
            pk_arr_new = pk_arr_a + other
        elif isinstance(other, Pk2D):
            a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b = \
                self._get_binary_operator_arrays(other)
            pk_arr_new = pk_arr_a + pk_arr_b
        else:
            raise TypeError(
                "Addition only defined between Pk2D objects and numbers.")

        logp = (pk_arr_new > 0).all()
        if logp:
            np.log(pk_arr_new, out=pk_arr_new)

        return Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                    is_logp=logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
            pk_arr_new = other * pk_arr_a
        elif isinstance(other, Pk2D):
            a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b = \
                self._get_binary_operator_arrays(other)
            pk_arr_new = pk_arr_a * pk_arr_b
        else:
            raise TypeError("Multiplication only defined between "
                            "Pk2D objects and numbers.")

        logp = (pk_arr_new > 0).all()
        if logp:
            np.log(pk_arr_new, out=pk_arr_new)

        return Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                    is_logp=logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __pow__(self, exponent):
        if not isinstance(exponent, (float, int)):
            raise TypeError(
                "Exponentiation of Pk2D only defined for numbers.")
        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        if (pk_arr_a < 0).any() and exponent % 1 != 0:
            warnings.warn(
                "Taking a non-positive Pk2D object to a non-integer "
                "power may lead to unexpected results.", CCLWarning)

        pk_arr_new = pk_arr_a**exponent

        logp = (pk_arr_new > 0).all()
        if logp:
            np.log(pk_arr_new, out=pk_arr_new)

        return Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                    is_logp=logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __sub__(self, other):
        return self + (-1)*other

    def __truediv__(self, other):
        return self * other**(-1)

    __radd__ = __add__

    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + (-1)*self

    def __rtruediv__(self, other):
        return other * self**(-1)

    @unlock_instance
    def __iadd__(self, other):
        self = self + other
        return self

    @unlock_instance
    def __imul__(self, other):
        self = self * other
        return self

    @unlock_instance
    def __isub__(self, other):
        self = self - other
        return self

    @unlock_instance
    def __itruediv__(self, other):
        self = self / other
        return self

    @unlock_instance
    def __ipow__(self, other):
        self = self**other
        return self


@warn_api
def parse_pk2d(
        cosmo: Cosmology,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM,
        *,
        is_linear: bool = False
) -> lib.f2d_t:
    """Get the C-level `f2d` spline associated with a :class:`~Pk2D` object.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    p_of_k_a
        Power spectrum. If a string, retrieve the power spectrum from `cosmo`
        (linear or non-linear, depending on `is_linear`).
    is_linear
        Whether to retrieve the linear power spectrum from `cosmo`. If False,
        retrieve the non-linear power spectrum.

    Returns
    -------

        C-level `f2d` spline.

    Raises
    ------
    ValueError
        If `p_of_k_a` cannot be parsed.
    """
    if isinstance(p_of_k_a, Pk2D):
        psp = p_of_k_a.psp
    else:
        if p_of_k_a is None:
            warnings.warn("The default power spectrum can is now designated "
                          "via ``ccl.DEFAULT_POWER_SPECTRUM``. The use of "
                          "``None`` will be deprecated in future versions.",
                          CCLDeprecationWarning)
            name = DEFAULT_POWER_SPECTRUM
        elif isinstance(p_of_k_a, str):
            name = p_of_k_a
        else:
            raise ValueError("p_of_k_a must be a pyccl.Pk2D object, "
                             "a string, or None")

        if is_linear:
            cosmo.compute_linear_power()
            pk = cosmo.get_linear_power(name)
        else:
            cosmo.compute_nonlin_power()
            pk = cosmo.get_nonlin_power(name)
        psp = pk.psp
    return psp


def parse_pk(
        cosmo: Cosmology,
        p_of_k_a: Union[Literal["linear", "nonlinear"], Pk2D] = "linear"
) -> Pk2D:
    """Helper to retrieve the power spectrum in the halo model.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    p_of_k_a
        Power spectrum. If a string, retrieve the power spectrum from `cosmo`.

    Returns
    -------

        Power spectrum.

    Raises
    ------
    ValueError
        If  `p_of_k_a` cannot be parsed.
    """
    if p_of_k_a is None:
        warnings.warn(
            "None is deprecated as a p_of_k_a value. The default is 'linear'.",
            CCLDeprecationWarning)
        p_of_k_a = "linear"
    if not (isinstance(p_of_k_a, Pk2D) or p_of_k_a in ["linear", "nonlinear"]):
        raise TypeError("p_of_k_a must be 'linear', 'nonlinear', Pk2D.")

    if p_of_k_a == "linear":
        return cosmo.get_linear_power()
    if p_of_k_a == "nonlinear":
        return cosmo.get_nonlin_power()
    if isinstance(p_of_k_a, Pk2D):
        return p_of_k_a
