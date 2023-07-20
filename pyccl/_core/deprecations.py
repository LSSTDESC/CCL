__all__ = ("deprecated", "deprecate_attr",)

import functools
import warnings

from .. import CCLDeprecationWarning


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
