from collections import OrderedDict
from .ccllib import (
    CCL_ERROR_CLASS, CCL_ERROR_INCONSISTENT, CCL_ERROR_INTEG,
    CCL_ERROR_LINSPACE, CCL_ERROR_MEMORY, CCL_ERROR_ROOT, CCL_ERROR_SPLINE,
    CCL_ERROR_SPLINE_EV, CCL_CORR_FFTLOG, CCL_CORR_BESSEL, CCL_CORR_LGNDRE,
    CCL_CORR_GG, CCL_CORR_GL, CCL_CORR_LP, CCL_CORR_LM)


__all__ = ('CCL_ERROR_CLASS', 'CCL_ERROR_INCONSISTENT', 'CCL_ERROR_INTEG',
           'CCL_ERROR_LINSPACE', 'CCL_ERROR_MEMORY', 'CCL_ERROR_ROOT',
           'CCL_ERROR_SPLINE', 'CCL_ERROR_SPLINE_EV', 'CCL_CORR_FFTLOG',
           'CCL_CORR_BESSEL', 'CCL_CORR_LGNDRE', 'CCL_CORR_GG', 'CCL_CORR_GL',
           'CCL_CORR_LP', 'CCL_CORR_LM')


class ClassProperty(type):
    """Metaclass that implements ``property`` to ``classmethod``."""
    # TODO: in py39 decorators `classmethod` and `property can be combined

    def __init__(cls, *args, **kwargs):
        cls._maxsize = 64

    @property
    def maxsize(cls):
        return cls._maxsize

    @maxsize.setter
    def maxsize(cls, value):
        cls._maxsize = value
        while len(cls._caches) > cls.maxsize:
            # iteratively reduce size to current maxsize
            first_item = cls.first(cls._caches)
            cls._caches.pop(first_item)


class Caching(metaclass=ClassProperty):
    """Infrastructure to hold cached objects.

    Caching is used for pre-computed objects that are expensive to compute.

    Attributes:
        maxsize (``int``):
            Maximum number of caches. If the dictionary is full, new caches
            are assigned in a rolling fashion (i.e. delete the oldest).
    """
    _caches = OrderedDict()
    _enabled = True

    @classmethod
    def first(cls, dic):
        """Get first element of ``OrderedDict``."""
        return next(iter(dic))

    @classmethod
    def cache(cls, func):
        """Decorator used to cache slow parts of the code."""

        def wrapper(instance):
            if not cls._enabled:
                return func(instance)

            key = str(hash(instance))

            if key in cls._caches:
                # output object has been cached
                return cls._caches[key]
            elif len(cls._caches) < cls.maxsize:
                # output object is not cached and there is space
                pass
            elif len(cls._caches) >= cls.maxsize:
                # output object is not cached and there is no space
                first_item = cls.first(cls._caches)
                cls._caches.pop(first_item)

            out = func(instance)
            cls._caches[key] = out
            return out
        return wrapper

    @classmethod
    def enable(cls, maxsize=64):
        cls._enabled = True
        cls.maxsize = maxsize

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def toggle(cls):
        cls._enabled = not cls._enabled
