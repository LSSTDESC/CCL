"""Utility functions to analyze status and error messages passed from CCL, as
well as wrappers to automatically vectorize functions."""
from . import ccllib as lib
from ._types import error_types
from .errors import CCLError, CCLWarning, CCLDeprecationWarning  # noqa
import functools
import warnings
import numpy as np
from inspect import signature, Parameter
try:
    from collections.abc import Iterable
except ImportError:  # pragma: no cover  (for py2.7)
    from collections import Iterable

NoneArr = np.array([])

integ_types = {'qag_quad': lib.integration_qag_quad,
               'spline': lib.integration_spline}

extrap_types = {'none': lib.f1d_extrap_0,
                'constant': lib.f1d_extrap_const,
                'linx_liny': lib.f1d_extrap_linx_liny,
                'linx_logy': lib.f1d_extrap_linx_logy,
                'logx_liny': lib.f1d_extrap_logx_liny,
                'logx_logy': lib.f1d_extrap_logx_logy}


def check(status, cosmo=None):
    """Check the status returned by a ccllib function.

    Args:
        status (int or :obj:`~pyccl.core.error_types`):
            Flag or error describing the success of a function.
        cosmo (:class:`~pyccl.core.Cosmology`, optional):
            A Cosmology object.
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
    if status in error_types.keys():
        raise CCLError("Error %s: %s" % (error_types[status], msg))

    # Check for unknown error
    if status != 0:
        raise CCLError("Error %d: %s" % (status, msg))


def debug_mode(debug):
    """Toggle debug mode on or off. If debug mode is on, the C backend is
    forced to print error messages as soon as they are raised, even if the
    flow of the program continues. This makes it easier to track down errors.

    If debug mode is off, the C code will not print errors, and the Python
    wrapper will raise the last error that was detected. If multiple errors
    were raised, all but the last will be overwritten within the C code, so the
    user will not necessarily be informed of the root cause of the error.

    Args:
        debug (bool): Switch debug mode on (True) or off (False).

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
#         x (float or array_like): Argument to fn.
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
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
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
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument and one integer argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
        n (int): Integer argument to fn.
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
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument and two float arguments, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in
                           a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets
                                            converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
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
        x1 (float or array_like): Argument to fn.
        x2 (float or array_like): Argument to fn.
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
        x1 (float or array_like): Argument to fn.
        x2 (float or array_like): Argument to fn.
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


def resample_array(x_in, y_in, x_out,
                   extrap_lo='none', extrap_hi='none',
                   fill_value_lo=0, fill_value_hi=0):
    """ Interpolates an input y array onto a set of x values.

    Args:
        x_in (array_like): input x-values.
        y_in (array_like): input y-values.
        x_out (array_like): x-values for output array.
        extrap_lo (string): type of extrapolation for x-values below the
            range of `x_in`. 'none' (for no interpolation), 'constant',
            'linx_liny' (linear in x and y), 'linx_logy', 'logx_liny' and
            'logx_logy'.
        extrap_hi (string): type of extrapolation for x-values above the
            range of `x_in`.
        fill_value_lo (float): constant value if `extrap_lo` is
            'constant'.
        fill_value_hi (float): constant value if `extrap_hi` is
            'constant'.
    Returns:
        array_like: output array.
    """

    if extrap_lo not in extrap_types.keys():
        raise ValueError("'%s' is not a valid extrapolation type. "
                         "Available options are: %s"
                         % (extrap_lo, extrap_types.keys()))
    if extrap_hi not in extrap_types.keys():
        raise ValueError("'%s' is not a valid extrapolation type. "
                         "Available options are: %s"
                         % (extrap_hi, extrap_types.keys()))

    status = 0
    y_out, status = lib.array_1d_resample(x_in, y_in, x_out,
                                          fill_value_lo, fill_value_hi,
                                          extrap_types[extrap_lo],
                                          extrap_types[extrap_hi],
                                          x_out.size, status)
    check(status)
    return y_out


def deprecated(new_function=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. If there is a replacement function,
    pass it as `new_function`.
    """
    def _depr_decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            s = "The function {} is deprecated.".format(func.__qualname__)
            if new_function:
                s += " Use {} instead.".format(new_function.__qualname__)
            warnings.warn(s, CCLDeprecationWarning)
            return func(*args, **kwargs)
        return new_func
    return _depr_decorator


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
        raise ValueError("rs should have %d elements" % n_r)

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
    if np.ndim(ys) < 1 or np.ndim(ys) > 2:
        raise ValueError("ys should be 1D or a 2D array")
    if np.ndim(ys) == 1:
        n_integ = 1
        n_x = len(ys)
    else:
        n_integ, n_x = ys.shape

    if len(x) != n_x:
        raise ValueError("x should have %d elements" % n_x)

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
            raise ValueError("%s needs to be a tuple of two arrays." % name)

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
        xarr: array_like
            The x array of the spline.
        yarr: array_like
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
        yarr: array_like
            The y array of the spline.
        xarr: array_like
            The x array of the spline.
        zarr: array_like
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
    """Get array data from a 3D GSL spline.

    Args:
        gsl_spline (`SWIGObject` of gsl_spline2d **):
            The SWIG object of the 2D GSL spline.
        length (int):
            The length of the 3rd dimension.

    Returns:
        xarr: array_like
            The x array of the spline.
        yarr: array_like
            The y array of the spline.
        zarr: array_like
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


def warn_api(func=None, /, *, pairs=None, order=None):
    """ This decorator translates old API to new API for:
      - functions/methods with changed argument order;
      - functions/methods whose arguments have been renamed.

    Parameters
    ----------
    pairs : list of pairs, optional
        List of renaming pairs ``('new', 'old')``. The default is None.
    order : list, optional
        List of the **old** order of the arguments whose order
        has been changed, under their **new** name. The default is None.

    .. note::

        This wrapper assumes that
            1. there are no positional-only arguments;
            2. the order of the positional-or-keyword arguments is unchanged.

    Example
    -------
    We have the legacy function

    >>> def func(a, b, c, d):
            # do something
            return a, b, c, d

    and we want to change the API to

    >>> def func(a, *, see, bee, d):
            # do the same thing
            return a, bee, see, d

    Then, adding this decorator to our new function would preserve API

    >>> @warn_api(pairs=[('bee', 'b'), ('see', 'c')],
                  order=['bee', 'see'])

    Notes
    -----
    Similar implementations for more specific usecases:
      1. ``TreeCorr``: https://github.com/rmjarvis/TreeCorr/blob/main/treecorr/util.py#L932
      2. ``Matplotlib``: https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_api/deprecation.py#L454
      3. ``legacy-api-wrap``: https://github.com/flying-sheep/legacy-api-wrap/blob/master/legacy_api_wrap.py#L27

    """  # noqa

    def _decorator(func, pairs, order):
        POK = Parameter.POSITIONAL_OR_KEYWORD
        KWO = Parameter.KEYWORD_ONLY

        # extract new parameters
        params = signature(func).parameters
        pos_names = [n for n, p in params.items() if p.kind == POK]
        kwo_names = [n for n, p in params.items() if p.kind == KWO]
        names = pos_names + kwo_names
        npos = len(pos_names)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # check for normprof - we'll need that later (1 of 3)
            def normprof_warning():
                # all the variables we need will already be in locals
                if ("normprof" in names) and ("normprof" not in kwargs):
                    # backwards-compatibility
                    kwargs["normprof"] = False
                    warnings.warn(
                        "Halo profile normalization `normprof` "
                        "has to be explicitly specified to prevent "
                        "unwanted behavior. Not specifying it "
                        "will trigger an exception in the future",
                        CCLDeprecationWarning)

            def cosmo_warning(args):
                warn_cosmo = False
                from .core import Cosmology
                from .halos import HMCalculator, MassFunc, HaloBias
                depr_cosmo_funcs = [HMCalculator, MassFunc, HaloBias]
                depr_cosmo_funcs += [Class.__qualname__
                                     for Class in MassFunc.__subclasses__()]
                depr_cosmo_funcs += [Class.__qualname__
                                     for Class in HaloBias.__subclasses__()]
                this_name = func.__qualname__
                if this_name.endswith(".__init__"):
                    # TODO: py39 introduced `str.removesuffix` method
                    this_name = this_name[:-len(".__init__")]
                if this_name in depr_cosmo_funcs:
                    # first arg is `self`, then it would be `cosmo`
                    if len(args) > 1 and isinstance(args[1], Cosmology):
                        new_args = list(args)  # we do this because
                        del new_args[1]        # `args` is a tuple and
                        args = new_args        # tuples are immutable
                        warn_cosmo = True
                    elif "cosmo" in kwargs:
                        kwargs.pop("cosmo")
                        warn_cosmo = True
                if warn_cosmo:
                    warnings.warn(
                        "`cosmo` has been deprecated as the first "
                        f"argument in {this_name}. This will return an "
                        "exception in the future.", CCLDeprecationWarning)
                return args

            args = cosmo_warning(args)

            # transform decorator input
            rename = np.atleast_2d(pairs) if pairs is not None else None
            swap = order.copy() if order is not None else None

            # rename any keyword-arguments?
            if rename is not None:
                these_kwargs = list(kwargs.keys())
                do_rename = not all([n in names for n in these_kwargs])
                if do_rename:
                    olds = []
                    news = []
                    for new, old in rename:
                        if old in kwargs:
                            olds.append(old)
                            news.append(new)
                            kwargs[new] = kwargs.pop(old)
                            if (swap is not None) and (new in swap):
                                swap.remove(new)

                    # warn about API change
                    s = "" if len(olds) == 1 else "s"
                    if len(olds) == 1:
                        news = news[0]
                        olds = olds[0]
                    warnings.warn(
                        f"Use of argument{s} {olds} is deprecated "
                        f"in {func.__qualname__}. Pass the new name{s} "
                        f"of the argument{s} {news}, respectively.",
                        CCLDeprecationWarning)

            # return if we have everything we need
            if len(args) <= npos:
                # 1. wrapper assumes positional arguments are not swapped
                # 2. not checking for new function
                # 3. raise normprof warning? (2 of 3)
                normprof_warning()
                return func(*args, **kwargs)

            # remove what was already added from todo list
            if swap is not None:
                for key in kwargs:
                    if key in swap:
                        swap.remove(key)

            # include new positionals to kwarg dict
            for name, value in zip(pos_names, args):
                kwargs[name] = value
            args = args[npos:]

            # deal with the remaining arguments
            extra_names = [n for n in names if n not in kwargs]
            if len(extra_names) > 0:
                if (swap is not None) and len(swap) > 0:
                    indices = [extra_names.index(n) for n in swap]
                    idx1, idx2 = min(indices), max(indices)
                    extra_names_api = extra_names[:idx1] + swap + \
                                      extra_names[idx2+1:]  # noqa
                else:
                    extra_names_api = extra_names

                for name, value in zip(extra_names_api, args):
                    kwargs[name] = value

            # warn about API change
            no_kw = names[npos : npos + len(args)]  # noqa
            s = "" if len(no_kw) == 1 else "s"
            no_kw = f"`{no_kw[0]}`" if len(no_kw) == 1 else no_kw
            warnings.warn(
                f"Use of argument{s} {no_kw} as positional is deprecated "
                f"in {func.__qualname__}. Pass the name{s} of the "
                f"keyword-only argument{s} explicitly.",
                CCLDeprecationWarning)

            # raise normprof warning? (3 of 3)
            normprof_warning()

            return func(**kwargs)
        return wrapper

    def decorator(func):
        return _decorator(func, pairs=pairs, order=order)

    # Check if usage is with @warn_api or @warn_api()
    if func is None:
        # warn_api() with patentheses
        return decorator
    # warn_api without parentheses
    return decorator(func)


def deprecate_attr(pairs=None):
    """ This decorator can be used to deprecate any attributes of a class,
    warning the users about it and pointing them to the new attribute.

    Parameters
    ----------
    pairs : list of pairs, optional
        List of renaming pairs ``('new', 'old')``. The default is None.

    Example
    -------
    We have the legacy class attribute ``mdef`` which we want to rename
    to ``mass_def``. To achieve this we override the ``__getattr__`` method
    of the attribute class using this decorator:

    >>> @deprecate_attr([('mass_def', 'mdef')])
        def __getattr__(self, name):
            return

    Now, every time the attribute is called via its old name, the user will
    be warned about the renaming, and the attribute value will be returned.

    """

    def wrapper(getter):

        new_names, old_names = np.asarray(pairs).T.tolist()

        @functools.wraps(getter)
        def new_getter(cls, this_name):

            # an attribute may not exist because
            # it has been renamed...
            if this_name in old_names:
                idx = old_names.index(this_name)
                new_name = new_names[idx]
                class_name = cls.__class__.__name__

                warnings.warn(
                    f"Attribute {this_name} is deprecated in {class_name}. "
                    f"Pass the new name {new_name}.", CCLDeprecationWarning)

                this_name = new_name

            # ...or because it simply does not exist
            try:
                attr = cls.__getattribute__(this_name)
                return attr
            except AttributeError as err:
                raise err

        return new_getter
    return wrapper
