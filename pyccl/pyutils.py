"""
============================
Utils (:mod:`pyccl.pyutils`)
============================

Utility and helper functions used throughout the source code.
"""

from __future__ import annotations

__all__ = (
    "CLevelErrors", "IntegrationMethods", "debug_mode", "get_pk_spline_lk",
    "get_pk_spline_a", "loglin_spacing", "resample_array")

from enum import Enum
from numbers import Number, Real
from typing import (TYPE_CHECKING, Callable, Iterable, Optional, Sequence,
                    Tuple, Union)

import numpy as np
from numpy.typing import NDArray

from . import CCLError, SplineParams, deprecated, lib

if TYPE_CHECKING:
    from . import Cosmology

NoneArr = np.array([])


class IntegrationMethods(Enum):
    """General integration methods."""
    QAG_QUAD = "qag_quad"
    """Adaptive quadrature integration."""

    SPLINE = "spline"
    """Integration using the spline coefficients of the integrand."""


integ_types = {
    'qag_quad': lib.integration_qag_quad,
    'spline': lib.integration_spline}

extrap_types = {
    'none': lib.f1d_extrap_0,
    'constant': lib.f1d_extrap_const,
    'linx_liny': lib.f1d_extrap_linx_liny,
    'linx_logy': lib.f1d_extrap_linx_logy,
    'logx_liny': lib.f1d_extrap_logx_liny,
    'logx_logy': lib.f1d_extrap_logx_logy}

# This is defined here instead of in `errors.py` because SWIG needs `CCLError`
# from `.errors`, resulting in a cyclic import.
CLevelErrors = {
    lib.CCL_ERROR_CLASS: 'CCL_ERROR_CLASS',
    lib.CCL_ERROR_INCONSISTENT: 'CCL_ERROR_INCONSISTENT',
    lib.CCL_ERROR_INTEG: 'CCL_ERROR_INTEG',
    lib.CCL_ERROR_LINSPACE: 'CCL_ERROR_LINSPACE',
    lib.CCL_ERROR_MEMORY: 'CCL_ERROR_MEMORY',
    lib.CCL_ERROR_ROOT: 'CCL_ERROR_ROOT',
    lib.CCL_ERROR_SPLINE: 'CCL_ERROR_SPLINE',
    lib.CCL_ERROR_SPLINE_EV: 'CCL_ERROR_SPLINE_EV',
    lib.CCL_ERROR_COMPUTECHI: 'CCL_ERROR_COMPUTECHI',
    lib.CCL_ERROR_MF: 'CCL_ERROR_MF',
    lib.CCL_ERROR_HMF_INTERP: 'CCL_ERROR_HMF_INTERP',
    lib.CCL_ERROR_PARAMETERS: 'CCL_ERROR_PARAMETERS',
    lib.CCL_ERROR_NU_INT: 'CCL_ERROR_NU_INT',
    lib.CCL_ERROR_EMULATOR_BOUND: 'CCL_ERROR_EMULATOR_BOUND',
    lib.CCL_ERROR_MISSING_CONFIG_FILE: 'CCL_ERROR_MISSING_CONFIG_FILE',
}


def check(status: int, cosmo: Cosmology = None) -> None:
    """Check the status returned by a :mod:`~pyccl.ccllib` function.

    Arguments
    ---------
    status
        Error type flag. The dictionary mapping is in
        :py:data:`~pyccl.pyutils.CLevelErrors`.
    cosmo
        :class:`~Cosmology` object that carries the error message.
    """
    if status == 0:
        return  # normal status (no errors raised)

    # Get status message from Cosmology object, if there is one
    msg = cosmo.cosmo.status_message if cosmo is not None else ""

    if status in CLevelErrors:
        raise CCLError(f"Error {CLevelErrors[status]}: {msg}")
    raise CCLError(f"Error {status}: {msg}")  # raise unknown error


def debug_mode(debug: bool) -> None:
    """Toggle debug mode.

    If debug mode is True (on), C-level errors are printed as soon as they are
    raised, ignoring normal program flow. Makes it easier to locate errors.
    If it is False (off), only the final C-level error is raised by the Python
    wrapper, once the program flow ends. Traceback not available.

    Arguments
    ---------
    debug
        Debug mode toggle switch.
    """
    lib.set_debug_policy(int(debug))


def check_openmp_version() -> str:
    """Get the OpenMP specification release date. 0 if not working."""
    return lib.openmp_version()


def check_openmp_threads() -> int:
    """Get the number of available threads. 0 if OpenMP is not working."""
    return lib.openmp_threads()


def cast_to_float(x: Union[Number, Sequence]) -> Union[Number, Sequence]:
    """Cast a scalar or a sequence to float."""
    if isinstance(x, Number):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    return [float(item) for item in x]


def _vectorize_fn(
        fn: Callable,
        fn_vec: Callable,
        cosmo: Cosmology,
        *varargs,
        x: Union[Real, NDArray[Real]],
        x2: Union[Real, NDArray[Real]] = None,
        pairwise: bool = False
) -> Union[float, NDArray[float]]:
    """Generic wrapper to allow vectorized access to CCL functions.

    Calling unvectorized functions is faster; this wrapper allocates the values
    as needed to increase efficiency. Vectorized functions are declared in an
    interface (``*.i``) module.

    fn
        Unvectorized function.
    fn_vec
        Vectorized function.
    cosmo
        Cosmological parameters.
    *varargs
        Other arguments to pass to the C function.
    x, x2
        Main axes.
    pairwise
        If True, call pairwise on `(x, x2)`. If False, orthogonalize `x2`.
    """
    xarrs = [arr for arr in [x, x2] if arr is not None]
    xarrs = list(map(cast_to_float, xarrs))

    if isinstance(x, Number):
        func = fn
        args = xarrs + list(varargs)
    else:
        func = fn_vec
        size = len(x) if pairwise else np.product([len(arr) for arr in xarrs])
        args = list(varargs) + xarrs + [int(size)]

    ret, status = func(cosmo.cosmo, *args, 0)
    cosmo.check(status)
    return ret


def loglin_spacing(logstart: Real, xmin: Real, xmax: Real,
                   num_log: int, num_lin: int) -> NDArray[float]:
    """Create an array spaced first logarithmically, then linearly.

    .. note::

        The number of logarithmically spaced points used is ``num_log - 1``
        because the first point of the linearly spaced points is the same as
        the end point of the logarithmically spaced points.

    .. code-block:: text

        | num_log - 1  |   |============== num_lin ================|
      --*-*--*---*-----*---*---*---*---*---*---*---*---*---*---*---*--> (axis)
        ^                  ^                                       ^
     logstart             xmin                                    xmax
    """
    log = np.geomspace(logstart, xmin, num_log-1, endpoint=False)
    lin = np.linspace(xmin, xmax, num_lin)
    return np.concatenate((log, lin))


def get_pk_spline_a(
        cosmo: Optional[Cosmology] = None,
        spline_params: Optional[Union[SplineParams, lib.spline_params]] = None
) -> NDArray[float]:
    r"""Get a sampling a-array. Used for P(k) splines.

    Arguments
    ---------
    cosmo
        Get the sampling rate from a :class:`~Cosmology` object.
    spline_params
        Get the sampling rate from a :class:`~SplineParams` instance, or from
        the C-level :class:`~lib.spline_params` struct. Ignored if `cosmo` is
        provided.

    Returns
    -------

        Samples in :math:`a`.
    """
    if cosmo is not None:
        spline_params = cosmo._spline_params
    if spline_params is None:
        from . import spline_params
    s = spline_params
    return loglin_spacing(s.A_SPLINE_MINLOG_PK, s.A_SPLINE_MIN_PK,
                          s.A_SPLINE_MAX, s.A_SPLINE_NLOG_PK, s.A_SPLINE_NA_PK)


def get_pk_spline_lk(
        cosmo: Optional[Cosmology] = None,
        spline_params: Optional[Union[SplineParams, lib.spline_params]] = None
) -> NDArray[float]:
    r"""Get a sampling log(k)-array. Used for P(k) splines.

    Arguments
    ---------
    cosmo
        Get the sampling rate from a :class:`~Cosmology` object.
    spline_params
        Get the sampling rate from a :class:`~SplineParams` instance, or from
        the C-level :class:`~lib.spline_params` struct. Ignored if `cosmo` is
        provided.

    Returns
    -------

        Samples in :math:`\log(k)`.
    """
    if cosmo is not None:
        spline_params = cosmo._spline_params
    if spline_params is None:
        from . import spline_params
    s = spline_params
    nk = int(np.ceil(np.log10(s.K_MAX/s.K_MIN)*s.N_K))
    return np.linspace(np.log(s.K_MIN), np.log(s.K_MAX), nk)


def resample_array(
        x_in: NDArray[Real],
        y_in: NDArray[Real],
        x_out: Union[Real, NDArray[Real]],
        extrap_lo: str = 'none',
        extrap_hi: str = 'none',
        fill_value_lo: Real = 0,
        fill_value_hi: Real = 0
) -> NDArray[float]:
    """Interpolate an array on a (new) set of values

    Arguments
    ---------
    x_in, y_in
        Input coordinates.
    x_out
        Interpolated `x` coordinates.
    extrap_lo, extrap_hi
        Extrapolation type, if `x_out` is outside of the range of `x_in`.
        Available options in
        :class:`~pyccl.base.parameters.fftlog_params.ExtrapolationMethods`.
    fill_value_lo, fill_value_hi
        Constant value to fill out-of-bounds if `extrap_xx` is ``'constant'``.

    Returns
    -------

        Interpolated values.

    Raises
    ------
    ValueError
        If the requested value is outside of the interpolation range.
    """
    types = extrap_types.keys()
    if not (extrap_lo in types and extrap_hi in types):
        raise ValueError("Invalid extrapolation type.")

    status = 0
    y_out, status = lib.array_1d_resample(x_in, y_in, x_out,
                                          fill_value_lo, fill_value_hi,
                                          extrap_types[extrap_lo],
                                          extrap_types[extrap_hi],
                                          x_out.size, status)

    if status == lib.CCL_ERROR_SPLINE_EV:
        raise ValueError("Value outside of interpolation range. "
                         "To extrapolate, pass `extrap_xx`.")
    check(status)
    return y_out


def _fftlog_transform(rs, frs, dim, mu, power_law_index):
    if np.ndim(rs) != 1:
        raise ValueError("rs should be a 1D array")
    if np.ndim(frs) < 1 or np.ndim(frs) > 2:
        raise ValueError("frs should be a 1D or 2D array")
    if np.ndim(frs) == 1:
        n_transforms = 1
        n_r = len(frs)
    else:
        n_transforms, n_r = frs.shape

    if len(rs) != n_r:
        raise ValueError(f"rs should have {n_r} elements")

    status = 0
    result, status = lib.fftlog_transform(n_transforms,
                                          rs, frs.flatten(),
                                          dim, mu, power_law_index,
                                          (n_transforms + 1) * n_r,
                                          status)
    check(status)
    result = result.reshape([n_transforms + 1, n_r])
    ks = result[0]
    fks = result[1:]
    if np.ndim(frs) == 1:
        fks = fks.squeeze()
    return ks, fks


def _spline_integrate(x, ys, a, b):
    if np.ndim(x) != 1:
        raise ValueError("x should be a 1D array")
    if np.ndim(ys) not in [1, 2]:
        raise ValueError("ys should be 1D or a 2D array")

    if np.ndim(ys) == 1:
        n_integ = 1
        n_x = len(ys)
    else:
        n_integ, n_x = ys.shape

    if len(x) != n_x:
        raise ValueError(f"x should have {n_x} elements")
    if np.ndim(a) or np.ndim(b):
        raise TypeError("Integration limits should be scalar")

    status = 0
    result, status = lib.spline_integrate(
        n_integ, x, ys.flatten(), a, b, n_integ, status)
    check(status)

    if np.ndim(ys) == 1:
        result = result[0]
    return result


def _check_array_params(f_arg, name=None, arr3=False):
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
        if ((not isinstance(f_arg, Iterable))
            or (len(f_arg) != (3 if arr3 else 2))
            or (not (isinstance(f_arg[0], Iterable)
                     and isinstance(f_arg[1], Iterable)))):
            raise ValueError(f"{name} must be a tuple of two arrays.")

        f1 = np.atleast_1d(np.array(f_arg[0], dtype=float))
        f2 = np.atleast_1d(np.array(f_arg[1], dtype=float))
        if arr3:
            f3 = np.atleast_1d(np.array(f_arg[2], dtype=float))
    if arr3:
        return f1, f2, f3
    return f1, f2


def _get_spline1d_arrays(gsl_spline) -> Tuple(NDArray[float]):
    """Get array data from a 1-D GSL spline.

    Arguments
    ---------
    gsl_spline : SwigPyObject of type gsl_spline *
        The SWIG object of the GSL spline.

    Returns
    -------
    x, y : ndarray
        Arrays of the spline.
    """
    status = 0
    size, status = lib.get_spline1d_array_size(gsl_spline, status)
    check(status)
    *arrs, status = lib.get_spline1d_arrays(gsl_spline, size, size, status)
    check(status)
    return arrs


def _get_spline2d_arrays(gsl_spline) -> Tuple(NDArray[float]):
    """Get array data from a 2-D GSL spline.

    Arguments
    ---------
    gsl_spline: SwigPyObject of type gsl_spline2d *
        The SWIG object of the GSL spline.

    Returns
    -------
    y, x, z : ndarray
        Arrays of the spline.
    """
    status = 0
    x_size, y_size, status = lib.get_spline2d_array_sizes(gsl_spline, status)
    check(status)
    z_size = x_size*y_size
    xarr, yarr, zarr, status = lib.get_spline2d_arrays(
        gsl_spline, x_size, y_size, z_size, status)
    check(status)
    return yarr, xarr, zarr.reshape(y_size, x_size)


def _get_spline3d_arrays(gsl_spline, length: int) -> Tuple(NDArray[float]):
    """Get array data from an array of 2-D GSL splines.

    Arguments
    ---------
    gsl_spline: SwigPyObject of type gsl_spline2d **
        The SWIG object of the GSL spline.
    length
        Length of the 3rd dimension.

    Returns
    -------
    x, y, z : ndarray
        Arrays of the spline.
    """
    status = 0
    x_size, y_size, status = lib.get_spline3d_array_sizes(gsl_spline, status)
    check(status)
    z_size = x_size*y_size*length
    xarr, yarr, zarr, status = lib.get_spline3d_arrays(
        gsl_spline, x_size, y_size, z_size, length, status)
    check(status)

    return xarr, yarr, zarr.reshape((length, x_size, y_size))

@deprecated
def assert_warns(wtype, f, *args, **kwargs):
    """Check that a function call `f(*args, **kwargs)` raises a warning of
    type wtype.
    Returns the output of `f(*args, **kwargs)` unless there was no warning,
    in which case an AssertionError is raised.
    """
    import warnings
    # Check that f() raises a warning, but not an error.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = f(*args, **kwargs)
    assert len(w) >= 1, "Expected warning was not raised."
    assert issubclass(w[0].category, wtype), \
        "Warning raised was the wrong type (got %s, expected %s)" % (
            w[0].category, wtype)
    return res
