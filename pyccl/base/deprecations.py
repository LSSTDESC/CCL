from ..errors import CCLDeprecationWarning
from inspect import signature, Parameter
import functools
import warnings


__all__ = ("deprecated", "warn_api",)


def deprecated(new_function=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. If there is a replacement function,
    pass it as `new_function`.
    """
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            s = "The function {} is deprecated.".format(func.__name__)
            if new_function:
                s += " Use {} instead.".format(new_function.__name__)
            warnings.warn(s, CCLDeprecationWarning)
            return func(*args, **kwargs)
        return new_func
    return decorator


def warn_api(func=None, *, pairs=[], reorder=[]):
    # To add below after halos modifications have been implemented.
    #  - constructors in the ``halos`` sub-package where ``cosmo`` is removed,
    #  - functions/methods where ``normprof`` is now a required argument.
    """ This decorator translates old API to new API for:
      - functions/methods whose arguments have been ranamed,
      - functions/methods with changed argument order,
    Parameters:
        pairs : list of pairs, optional
            List of renaming pairs ``('old', 'new')``.
        reorder : list, optional
            List of the **previous** order of the arguments whose order
            has been changed, under their **new** name.
    Example:
        We have the legacy constructor:
        >>> def __init__(self, cosmo, a, b, c=0, d=1, normprof=False):
                # do something
                return a, b, c, d, normprof
        and we want to change the API to
        >>> def __init__(self, a, *, see=0, bee, d=1, normprof=None):
                # do the same thing
                return a, bee, see, d, normprof
        Then, adding this decorator to our new function would preserve API
        >>> @warn_api(pairs=[('b', 'bee'), ('c', 'see')],
                      reorder=['bee', 'see'])
    """
    # To add above after halos modifications have been implemented
    # - ``cosmo`` is automatically detected for all constructors in ``halos``
    # - ``normprof`` is automatically detected for all decorated functions.
    if func is None:
        # called with parentheses
        return functools.partial(warn_api, pairs=pairs, reorder=reorder)

    name = func.__qualname__
    plural = lambda expr: "" if not len(expr)-1 else "s"  # noqa: final 's'
    params = signature(func).parameters
    POK = Parameter.POSITIONAL_OR_KEYWORD
    KWO = Parameter.KEYWORD_ONLY
    pos_names = [k for k, v in params.items() if v.kind == POK]
    kwo_names = [k for k, v in params.items() if v.kind == KWO]
    npos = len(pos_names)
    rename = dict(pairs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Custom definition of `isinstance` to avoic cyclic imports.
        is_instance = lambda obj, cl: cl in obj.__class__.__name__  # noqa

        # To add after halos modifications have been implemented
        # API compatibility with `cosmo` as a first argument in `halos`.
        # catch_cosmo = args[1] if len(args) > 1 else kwargs.get("cosmo")
        # if ("pyccl.halos" in func.__module__
        #         and func.__name__ == "__init__"
        #         and is_instance(catch_cosmo, "Cosmology")):
        #     warnings.warn(
        #         f"Use of argument `cosmo` has been deprecated in {name}. "
        #         "This will trigger an exception in the future.",
        #         CCLDeprecationWarning)
        #     # `cosmo` may be in `args` or in `kwargs`, so we check both.
        #     args = tuple(
        #         item for item in args if not is_instance(item, "Cosmology"))
        #     kwargs.pop("cosmo", None)
        #
        # API compatibility for reordered positionals in `fourier_2pt`.
        # first_arg = args[1] if len(args) > 1 else None
        # if (func.__name__ == "fourier_2pt"
        #         and is_instance(first_arg, "HaloProfile")):
        #     api = dict(zip(["prof", "cosmo", "k", "M", "a"], args[1: 6]))
        #     args = (args[0],) + args[6:]  # discard args [1-5]
        #     kwargs.update(api)            # they are now kwargs
        #     warnings.warn(
        #         "API for Profile2pt.fourier_2pt has changed. "
        #         "Argument order (prof, cosmo, k, M, a) has been replaced by "
        #         "(cosmo, k, M, a, prof).", CCLDeprecationWarning)

        # API compatibility for renamed arguments.
        warn_names = set(kwargs) - set(params)
        if warn_names:
            s = plural(warn_names)
            warnings.warn(
                f"Use of argument{s} {list(warn_names)} is deprecated "
                f"in {name}. Pass the new name{s} of the argument{s} "
                f"{[rename[k] for k in warn_names]}, respectively.",
                CCLDeprecationWarning)
            for param in warn_names:
                kwargs[rename[param]] = kwargs.pop(param)

        # API compatibility for star operator.
        if len(args) > npos:
            # API compatibility for shuffled order.
            if reorder:
                # Pick up the positions of the common elements.
                mask = [param in reorder for param in kwo_names]
                start = mask.index(True)
                stop = start + len(reorder)
                # Sort the reordered part of `kwo_names` by `reorder` indexing.
                kwo_names[start: stop] = sorted(kwo_names[start: stop],
                                                key=reorder.index)
            extras = dict(zip(kwo_names, args[npos:]))
            kwargs.update(extras)
            s = plural(extras)
            warnings.warn(
                f"Use of argument{s} {list(extras)} as positional is "
                f"deprecated in {func.__qualname__}.", CCLDeprecationWarning)

        # To add after halos modifications have been implemented
        # # API compatibility for `normprof` as a required argument.
        # if "normprof" in set(params) - set(kwargs):
        #     kwargs["normprof"] = False
        #     warnings.warn(
        #        "Halo profile normalization `normprof` has become a required "
        #        f"argument in {name}. Not specifying it will trigger an "
        #        "exception in the future", CCLDeprecationWarning)

        # Collect what's remaining and sort to preserve signature order.
        pos = dict(zip(pos_names, args))
        kwargs.update(pos)
        kwargs = {param: kwargs[param]
                  for param in sorted(kwargs, key=list(params).index)}

        return func(**kwargs)
    return wrapper
