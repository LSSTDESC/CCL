"""Utility functions to analyze status and error messages passed from CCL, as
well as wrappers to automatically vectorize functions.
"""
__all__ = (
    "CLevelErrors", "ExtrapolationMethods", "IntegrationMethods", "check",
    "debug_mode", "get_pk_spline_lk", "get_pk_spline_a", "resample_array")

from enum import Enum
from typing import Iterable

import numpy as np

from . import CCLError, lib, spline_params


NoneArr = np.array([])


class IntegrationMethods(Enum):
    QAG_QUAD = "qag_quad"
    SPLINE = "spline"


class ExtrapolationMethods(Enum):
    NONE = "none"
    CONSTANT = "constant"
    LINX_LINY = "linx_liny"
    LINX_LOGY = "linx_logy"
    LOGX_LINY = "logx_liny"
    LOGX_LOGY = "logx_logy"


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


def check(status, cosmo=None):
    """Check the status returned by a ccllib function.

    Args:
        status (:obj:`int` or :obj:`~pyccl.core.error_types`):
            Flag or error describing the success of a function.
        cosmo (:class:`~pyccl.cosmology.Cosmology`):
            A Cosmology object.

    :meta private:
    """
    # Check for normal status (no action required)
    if status == 0:
        return

    # Get status message from Cosmology object, if there is one
    if cosmo is not None:
        msg = cosmo.cosmo.status_message
    else:
        msg = ""

    # Check for known error status
    if status in CLevelErrors.keys():
        raise CCLError(f"Error {CLevelErrors[status]}: {msg}")

    # Check for unknown error
    if status != 0:
        raise CCLError(f"Error {status}: {msg}")


def debug_mode(debug):
    """Toggle debug mode on or off. If debug mode is on, the C backend is
    forced to print error messages as soon as they are raised, even if the
    flow of the program continues. This makes it easier to track down errors.

    If debug mode is off, the C code will not print errors, and the Python
    wrapper will raise the last error that was detected. If multiple errors
    were raised, all but the last will be overwritten within the C code, so the
    user will not necessarily be informed of the root cause of the error.

    Args:
        debug (:obj:`bool`): Switch debug mode on (True) or off (False).

    """
    if debug:
        lib.set_debug_policy(lib.CCL_DEBUG_MODE_ON)
    else:
        lib.set_debug_policy(lib.CCL_DEBUG_MODE_OFF)


# This function is not used anymore so we don't want Coveralls to
# include it, but we keep it in case it is needed at some point.
# def _vectorize_fn_simple(fn, fn_vec, x,
#                          returns_status=True):  # pragma: no cover
#     """Generic wrapper to allow vectorized (1D array) access to CCL
#     functions with one vector argument (but no dependence on cosmology).
#
#     Args:
#         fn (callable): Function with a single argument.
#         fn_vec (callable): Function that has a vectorized implementation in
#                            a .i file.
#         x (:obj:`float` or `array`): Argument to fn.
#         returns_stats (bool): Indicates whether fn returns a status.
#
#     """
#     status = 0
#     if isinstance(x, int):
#         x = float(x)
#     if isinstance(x, float):
#         # Use single-value function
#         if returns_status:
#             f, status = fn(x, status)
#         else:
#             f = fn(x)
#     elif isinstance(x, np.ndarray):
#         # Use vectorised function
#         if returns_status:
#             f, status = fn_vec(x, x.size, status)
#         else:
#             f = fn_vec(x, x.size)
#     else:
#         # Use vectorised function
#         if returns_status:
#             f, status = fn_vec(x, len(x), status)
#         else:
#             f = fn_vec(x, len(x))
#
#     # Check result and return
#     check(status)
#     return f


def _vectorize_fn(fn, fn_vec, cosmo, x, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL
    functions with one vector argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (:obj:`float` or `array`): Argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    status = 0

    if isinstance(x, int):
        x = float(x)
    if isinstance(x, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x, status)
        else:
            f = fn(cosmo, x)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x, x.size, status)
        else:
            f = fn_vec(cosmo, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x, len(x), status)
        else:
            f = fn_vec(cosmo, x, len(x))

    # Check result and return
    check(status, cosmo_in)
    return f


def _vectorize_fn3(fn, fn_vec, cosmo, x, n, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL
    functions with one vector argument and one integer argument,
    with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (:obj:`float` or `array`): Argument to fn.
        n (:obj:`int`): Integer argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0

    if isinstance(x, int):
        x = float(x)
    if isinstance(x, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x, n, status)
        else:
            f = fn(cosmo, x, n)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, n, x, x.size, status)
        else:
            f = fn_vec(cosmo, n, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, n, x, len(x), status)
        else:
            f = fn_vec(cosmo, n, x, len(x))

    # Check result and return
    check(status, cosmo_in)
    return f


def _vectorize_fn4(fn, fn_vec, cosmo, x, a, d, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL
    functions with one vector argument and two float arguments, with
    a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (:obj:`float` or `array`): Argument to fn.
        a (float): Float argument to fn.
        d (float): Float argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0

    if isinstance(x, int):
        x = float(x)
    if isinstance(x, float):
        if returns_status:
            f, status = fn(cosmo, x, a, d, status)
        else:
            f = fn(cosmo, x, a, d)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, a, d, x, x.size, status)
        else:
            f = fn_vec(cosmo, a, d, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, a, d, x, len(x), status)
        else:
            f = fn_vec(cosmo, a, d, x, len(x))

    # Check result and return
    check(status, cosmo_in)
    return f


def _vectorize_fn5(fn, fn_vec, cosmo, x1, x2, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL
    functions with two vector arguments of the same length,
    with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x1 (:obj:`float` or `array`): Argument to fn.
        x2 (:obj:`float` or `array`): Argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0

    # If a scalar was passed, convert to an array
    if isinstance(x1, int):
        x1 = float(x1)
        x2 = float(x2)
    if isinstance(x1, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x1, x2, status)
        else:
            f = fn(cosmo, x1, x2)
    elif isinstance(x1, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x1, x2, x1.size, status)
        else:
            f = fn_vec(cosmo, x1, x2, x1.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x1, x2, len(x2), status)
        else:
            f = fn_vec(cosmo, x1, x2, len(x2))

    # Check result and return
    check(status, cosmo_in)
    return f


def _vectorize_fn6(fn, fn_vec, cosmo, x1, x2, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL
    functions with two vector arguments of the any length,
    with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x1 (:obj:`float` or `array`): Argument to fn.
        x2 (:obj:`float` or `array`): Argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0

    # If a scalar was passed, convert to an array
    if isinstance(x1, int):
        x1 = float(x1)
        x2 = float(x2)
    if isinstance(x1, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x1, x2, status)
        else:
            f = fn(cosmo, x1, x2)
    elif isinstance(x1, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x1, x2, int(x1.size*x2.size), status)
        else:
            f = fn_vec(cosmo, x1, x2, int(x1.size*x2.size))
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x1, x2, int(len(x1)*len(x2)), status)
        else:
            f = fn_vec(cosmo, x1, x2, int(len(x1)*len(x2)))

    # Check result and return
    check(status, cosmo_in)
    return f


def loglin_spacing(logstart, xmin, xmax, num_log, num_lin):
    """Create an array spaced first logarithmically, then linearly.

    .. note::

        The number of logarithmically spaced points used is ``num_log - 1``
        because the first point of the linearly spaced points is the same as
        the end point of the logarithmically spaced points.

    .. code-block:: text

        |=== num_log ==|   |============== num_lin ================|
      --*-*--*---*-----*---*---*---*---*---*---*---*---*---*---*---*--> (axis)
        ^                  ^                                       ^
     logstart             xmin                                    xmax

    """
    log = np.geomspace(logstart, xmin, num_log-1, endpoint=False)
    lin = np.linspace(xmin, xmax, num_lin)
    return np.concatenate((log, lin))


def get_pk_spline_nk(cosmo=None, spline_params=spline_params):
    """Get the number of sampling points in the wavenumber dimension.

    Arguments:
        cosmo (:obj:`~pyccl.ccllib.cosmology` via SWIG):
            Input cosmology.

    :meta private:
    """
    if cosmo is not None:
        return lib.get_pk_spline_nk(cosmo.cosmo)
    ndecades = np.log10(spline_params.K_MAX / spline_params.K_MIN)
    return int(np.ceil(ndecades*spline_params.N_K))


def get_pk_spline_na(cosmo=None, spline_params=spline_params):
    """Get the number of sampling points in the scale factor dimension.

    Arguments:
        cosmo (:obj:`~pyccl.ccllib.cosmology` via SWIG):
            Input cosmology.

    :meta private:
    """
    if cosmo is not None:
        return lib.get_pk_spline_na(cosmo.cosmo)
    return spline_params.A_SPLINE_NA_PK + spline_params.A_SPLINE_NLOG_PK - 1


def get_pk_spline_lk(cosmo=None, spline_params=spline_params):
    """Get a log(k)-array with sampling rate defined by ``ccl.spline_params``
    or by the spline parameters of the input ``cosmo``.

    Arguments:
        cosmo (:obj:`~pyccl.ccllib.cosmology` via SWIG):
            Input cosmology.

    :meta private:
    """
    nk = get_pk_spline_nk(cosmo=cosmo, spline_params=spline_params)
    if cosmo is not None:
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, 0)
        check(status, cosmo)
        return lk_arr
    lk_arr, status = lib.get_pk_spline_lk_from_params(spline_params, nk, 0)
    check(status)
    return lk_arr


def get_pk_spline_a(cosmo=None, spline_params=spline_params):
    """Get an a-array with sampling rate defined by ``ccl.spline_params``
    or by the spline parameters of the input ``cosmo``.

    Arguments:
        cosmo (:obj:`~pyccl.ccllib.cosmology` via SWIG):
            Input cosmology.

    :meta private:
    """
    na = get_pk_spline_na(cosmo=cosmo, spline_params=spline_params)
    if cosmo is not None:
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, 0)
        check(status, cosmo)
        return a_arr
    a_arr, status = lib.get_pk_spline_a_from_params(spline_params, na, 0)
    check(status)
    return a_arr


def resample_array(x_in, y_in, x_out,
                   extrap_lo='none', extrap_hi='none',
                   fill_value_lo=0, fill_value_hi=0):
    """ Interpolates an input y array onto a set of x values.

    Args:
        x_in (`array`): input x-values.
        y_in (`array`): input y-values.
        x_out (`array`): x-values for output array.
        extrap_lo (:obj:`str`): type of extrapolation for x-values below the
            range of `x_in`. 'none' (for no interpolation), 'constant',
            'linx_liny' (linear in x and y), 'linx_logy', 'logx_liny' and
            'logx_logy'.
        extrap_hi (:obj:`str`): type of extrapolation for x-values above the
            range of `x_in`.
        fill_value_lo (:obj:`float`): constant value if `extrap_lo` is
            'constant'.
        fill_value_hi (:obj:`float`): constant value if `extrap_hi` is
            'constant'.
    Returns:
        `array`: output array.
    """
    # TODO: point to the enum in CCLv3 docs.
    if extrap_lo not in extrap_types.keys():
        raise ValueError("Invalid extrapolation type.")
    if extrap_hi not in extrap_types.keys():
        raise ValueError("Invalid extrapolation type.")

    status = 0
    y_out, status = lib.array_1d_resample(x_in, y_in, x_out,
                                          fill_value_lo, fill_value_hi,
                                          extrap_types[extrap_lo],
                                          extrap_types[extrap_hi],
                                          x_out.size, status)
    check(status)
    return y_out


def _fftlog_transform(rs, frs,
                      dim, mu, power_law_index):
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


# Compute the discrete Hankel transform of the function frs
# evaluated at values rs.
# Weighted by a power law and the bessel_deriv-th derivative of the
# (spherical) bessel function of order \mu.
# The computed transform will be centered about the peak of the given \mu.
# double bessel_deriv: the nth derivative of the (spherical) bessel function
# int spherical_bessel: 1 spherical bessel functions, 0 bessel functions
# double q: the biasing index
# NOTE: we ignore factors of (2*pi) found in typical fht algorithm,
# these factors should be calculated and applied a-posteriori if necessary
def _fftlog_transform_general(
    rs, frs, mu, q, spherical_bessel, bessel_deriv, window_frac
):
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
        raise ValueError("rs should have %d elements" % n_r)

    status = 0
    result, status = lib.fftlog_transform_general(
        n_transforms,
        rs,
        frs.flatten(),
        mu,
        q,
        spherical_bessel,
        bessel_deriv,
        window_frac,
        (n_transforms + 1) * n_r,
        status,
    )
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
    if np.ndim(ys) < 1 or np.ndim(ys) > 2:
        raise ValueError("ys should be 1D or a 2D array")
    if np.ndim(ys) == 1:
        n_integ = 1
        n_x = len(ys)
    else:
        n_integ, n_x = ys.shape

    if len(x) != n_x:
        raise ValueError(f"x should have {n_x} elements")

    if np.ndim(a) > 0 or np.ndim(b) > 0:
        raise TypeError("Integration limits should be scalar")

    status = 0
    result, status = lib.spline_integrate(n_integ,
                                          x, ys.flatten(),
                                          a, b, n_integ,
                                          status)
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
    else:
        return f1, f2


def _get_spline1d_arrays(gsl_spline):
    """Get array data from a 1D GSL spline.

    Args:
        gsl_spline: `SWIGObject` of gsl_spline
            The SWIG object of the GSL spline.

    Returns:
        xarr: `array`
            The x array of the spline.
        yarr: `array`
            The y array of the spline.
    """
    status = 0
    size, status = lib.get_spline1d_array_size(gsl_spline, status)
    check(status)

    xarr, yarr, status = lib.get_spline1d_arrays(gsl_spline, size, size,
                                                 status)
    check(status)

    return xarr, yarr


def _get_spline2d_arrays(gsl_spline):
    """Get array data from a 2D GSL spline.

    Args:
        gsl_spline: `SWIGObject` of gsl_spline2d *
            The SWIG object of the 2D GSL spline.

    Returns:
        yarr: `array`
            The y array of the spline.
        xarr: `array`
            The x array of the spline.
        zarr: `array`
            The z array of the spline. The shape is (yarr.size, xarr.size).
    """
    status = 0
    x_size, y_size, status = lib.get_spline2d_array_sizes(gsl_spline, status)
    check(status)

    z_size = x_size*y_size
    xarr, yarr, zarr, status = lib.get_spline2d_arrays(gsl_spline,
                                                       x_size, y_size, z_size,
                                                       status)
    check(status)

    return yarr, xarr, zarr.reshape(y_size, x_size)


def _get_spline3d_arrays(gsl_spline, length):
    """Get array data from an array of 2D GSL splines.

    Args:
        gsl_spline (`SWIGObject` of gsl_spline2d **):
            The SWIG object of the 2D GSL spline.
        length (:obj:`int`):
            The length of the 3rd dimension.

    Returns:
        xarr: `array`
            The x array of the spline.
        yarr: `array`
            The y array of the spline.
        zarr: `array`
            The z array of the spline. The shape is (yarr.size, xarr.size).
    """
    status = 0
    x_size, y_size, status = lib.get_spline3d_array_sizes(gsl_spline, status)
    check(status)

    z_size = x_size*y_size*length
    xarr, yarr, zarr, status = lib.get_spline3d_arrays(gsl_spline,
                                                       x_size, y_size, z_size,
                                                       length, status)
    check(status)

    return xarr, yarr, zarr.reshape((length, x_size, y_size))


def check_openmp_version():
    """Return the OpenMP specification release date.
    Return 0 if OpenMP is not working.
    """

    return lib.openmp_version()


def check_openmp_threads():
    """Returns the number of processors available to the device.
    Return 0 if OpenMP is not working.
    """

    return lib.openmp_threads()
