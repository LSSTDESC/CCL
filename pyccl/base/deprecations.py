from ..errors import CCLDeprecationWarning
from inspect import signature, Parameter
import functools
import warnings


__all__ = ("deprecated", "warn_api", "deprecate_attr",)


def deprecated(new_function=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. If there is a replacement function,
    pass it as `new_function`.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            s = f"The function {func.__qualname__} is deprecated."
            if new_function:
                s += f" Use {new_function.__qualname__} instead."
            warnings.warn(s, CCLDeprecationWarning)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def warn_api(func=None, *, pairs=[], reorder=[]):
    """This decorator translates old API to new API for:
      - functions/methods whose arguments have been ranamed,
      - functions/methods with changed argument order,
      - constructors in the ``halos`` sub-package where ``cosmo`` is removed,
      - constructors in ``halos`` where the default ``MassDef`` is not None,
      - functions/methods where ``normprof`` is deprecated.

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

        - ``cosmo`` is automatically detected for all constructors in ``halos``
        - ``normprof`` is automatically detected for all decorated functions.
    """
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

        # API compatibility with `cosmo` as a first argument in `halos`.
        catch_cosmo = args[1] if len(args) > 1 else kwargs.get("cosmo")
        if ("pyccl.halos" in func.__module__
                and func.__name__ == "__init__"
                and is_instance(catch_cosmo, "Cosmology")):
            warnings.warn(
                f"Use of argument `cosmo` has been deprecated in {name}. "
                "This will trigger an exception in the future.",
                CCLDeprecationWarning)
            # `cosmo` may be in `args` or in `kwargs`, so we check both.
            args = tuple(
                item for item in args if not is_instance(item, "Cosmology"))
            kwargs.pop("cosmo", None)

        # API compatibility for reordered positionals in `fourier_2pt`.
        first_arg = args[1] if len(args) > 1 else None
        if (func.__name__ == "fourier_2pt"
                and is_instance(first_arg, "HaloProfile")):
            api = dict(zip(["prof", "cosmo", "k", "M", "a"], args[1: 6]))
            args = (args[0],) + args[6:]  # discard args [1-5]
            kwargs.update(api)            # they are now kwargs
            warnings.warn(
                "API for Profile2pt.fourier_2pt has changed. "
                "Argument order (prof, cosmo, k, M, a) has been replaced by "
                "(cosmo, k, M, a, prof).", CCLDeprecationWarning)

        # API compatibility for renamed arguments.
        warn_names = set(kwargs) - set(params)
        unexpected = set([k for k in warn_names if k not in rename])
        if unexpected:
            # check whether it is an FFTLog parameter (passed as kwarg)
            if "HaloProfile" in func.__qualname__:
                from ..parameters import FFTLogParams as FFTL
                for param in list(unexpected):
                    if hasattr(FFTL, param):
                        # these aren't unexpected - they will be piped to FFTL
                        unexpected.remove(param)
                        warn_names.remove(param)
        if unexpected:
            # emulate Python default behavior for arguments that don't exist
            raise TypeError(
                f"{func.__name__}() got an unexpected keyword argument "
                f"'{list(unexpected[0])}'")
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

        # API compatibility for `normprof` as a required argument.
        if any(["normprof" in par for par in kwargs.items()]):
            warnings.warn(
                "Argument `normprof` has been deprecated. Change the default "
                "value only by subclassing. More comprehensive profile "
                "normalization options will be provided with CCLv3.0.0.",
                CCLDeprecationWarning)

        # API compatibility for deprecated HMCalculator argument k_min.
        if func.__qualname__ == "HMCalculator.__init__" and "k_min" in kwargs:
            warnings.warn(
                "Argument `k_min` has been deprecated in `HMCalculator. "
                "This is now specified in each profile's `_normalization()` "
                "method.", CCLDeprecationWarning)

        # API compatibility for non-None default `MassDef` in `halos`.
        if (params.get("mass_def") is not None
                and "mass_def" in kwargs
                and kwargs["mass_def"] is None):
            kwargs["mass_def"] = params["mass_def"].default
            warnings.warn(
                "`None` has been deprecated as a value for mass_def. "
                "To use the default, leave the parameter empty.",
                CCLDeprecationWarning)

        # Collect what's remaining and sort to preserve signature order.
        # Set aside the ones passed as kwargs (e.g. FFTL in HaloProfile).
        others = {param: kwargs[param]
                  for param in kwargs
                  if param not in list(params)}
        # Only these will be bound.
        kwargs = {param: kwargs[param]
                  for param in kwargs
                  if param not in others}
        pos = dict(zip(pos_names, args))
        kwargs.update(pos)
        kwargs = {param: kwargs[param]
                  for param in sorted(kwargs, key=list(params).index)}
        # Now, insert the other kwargs back in.
        kwargs.update(others)

        return func(**kwargs)
    return wrapper


def deprecate_attr(getter=None, *, pairs=[]):
    """This decorator can be used to deprecate attributes,
    warning users about it and pointing them to the new attribute.

    Parameters
    ----------
    getter : slot wrapper ``__getattribute__``
        This is the getter method to be decorated.
    pairs : list of pairs
        List of renaming pairs ``('old', 'new')``.

    Example
    -------
    We have the legacy attribute ``old_name`` which we want to rename
    to ``new_name``. To achieve this we decorate the ``__getattribute__``
    method of the parent class in the main class body to retrieve the
    ``__getattr__`` method for the main class, like so:

    >>>  __getattr__ = deprecate_attr([('old_name', 'new_name')])(
             super.__getattribute__)

    Now, every time the attribute is called via its old name, the user will
    be warned about the renaming, and the attribute value will be returned.

    .. note:: Make sure that you bind ``__getattr__`` to the decorator,
              rather than ``__getattribute__``, because ``__getattr__``
              provides the fallback mechanism we want to use. Otherwise,
              an infinite recursion will initiate.

    """
    if getter is None:
        return functools.partial(deprecate_attr, pairs=pairs)

    rename = dict(pairs)

    @functools.wraps(getter)
    def wrapper(cls, name):
        if name in rename:
            new_name = rename[name]
            class_name = cls.__class__.__name__
            warnings.warn(
                f"Attribute {name} is deprecated in {class_name}. "
                f"Pass the new name {new_name}.", CCLDeprecationWarning)
            name = new_name

        return cls.__getattribute__(name)
    return wrapper
