"""
====================================
Caching (:mod:`pyccl._core.caching`)
====================================

Framework that enables caching.
"""

__all__ = ("hash_", "Caching", "cache", "CacheInfo", "CachedObject",)

import sys
import functools
from collections import OrderedDict
from inspect import signature
from numbers import Number
from _thread import RLock
from typing import Any, Callable, Hashable, Iterable, List, Literal, final

import numpy as np


def _to_string(obj: Any) -> Hashable:
    """Make unhashable objects hashable in a consistent manner.

    Arguments
    ---------
    obj
        Any object to be hashed.

    See Also
    --------
    :func:`~hash_`
    """
    if isinstance(obj, str):                     # strings returned directly
        return obj
    if isinstance(obj, Number):                  # numbers converted to strings
        return str(obj)
    if isinstance(obj, np.ndarray):              # arrays converted to bytes
        return str(obj.tobytes())
    if isinstance(obj, dict):                    # recurse dicts
        out = {key: _to_string(value) for key, value in obj.items()}
        if isinstance(obj, OrderedDict):
            return repr(tuple(out.items()))      # ordered dicts unsorted
        return repr(tuple(sorted(out.items())))  # dicts sorted
    if isinstance(obj, Iterable):                # recurse iterables
        return repr(tuple([_to_string(item) for item in obj]))
    return repr(obj)                             # rely on unique repr


def hash_(obj: Any) -> str:
    """Generic hash method, which changes between processes. It is designed to
    hash every type, even those that are by default unhashable.

    The steps of the algorithm are (in order):

    * Strings are returned directly.
    * Numerical types are converted to strings because some numbers share hash.
    * For numpy arrays, return the bytes data buffer.
    * Dictionaries are sorted and iterated recursively with the above rules.
    * Ordered dictionaries follow dictionaries, but their order is preserved.
    * Other iterables are iterated recursively.
    * If none of the above holds, the representation string is returned.
    """
    return hash(_to_string(obj)) + sys.maxsize + 1


class _CachingMeta(type):
    """Implement `property` to a `classmethod` for `Caching`."""
    # NOTE: Only in 3.8 < py < 3.11 can `classmethod` wrap `property`.
    # https://docs.python.org/3.11/library/functions.html#classmethod
    @property
    def maxsize(cls) -> int:
        """Maximum number of caches to store. If the register is full, new
        caches are assigned according to the cache retention policy.
        """
        return cls._maxsize

    @maxsize.setter
    def maxsize(cls, value):
        if value < 0:
            raise ValueError(
                "`maxsize` should be larger than zero. "
                "To disable caching, use `Caching.disable()`.")
        cls._maxsize = value
        for func in cls._cached_functions:
            func.cache_info.maxsize = value

    @property
    def policy(cls) -> Literal["fifo", "lru", "lfu"]:
        """`Cache retention policy
        <https://en.wikipedia.org/wiki/Cache_replacement_policies#Policies>`_.
        """
        return cls._policy

    @policy.setter
    def policy(cls, value):
        if value not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")
        if value == "lfu" != cls._policy:
            # Reset counter if we change policy to lfu
            # otherwise new objects are prone to being discarded immediately.
            # Now, the counter is not just used for stats,
            # it is part of the retention policy.
            for func in cls._cached_functions:
                for item in func.cache_info._caches.values():
                    item.reset()
        cls._policy = value
        for func in cls._cached_functions:
            func.cache_info.policy = value


@final
class Caching(metaclass=_CachingMeta):
    """Utility class that implements the infrastructure for caching.

    Attributes
    ----------
    maxsize : int
        Maximum number of caches to store. If the register is full, new caches
        are assigned according to the cache retention policy.
    policy : Literal["fifo", "lru", "lfu"]
        `Cache retention policy
        <https://en.wikipedia.org/wiki/Cache_replacement_policies#Policies>`_.
    """
    _enabled = False

    _policies = ['fifo', 'lru', 'lfu']
    _default_maxsize = 128   # class default maxsize
    _default_policy = 'lru'  # class default policy

    _maxsize: int = _default_maxsize                          # user maxsize
    _policy: Literal["fifo", "lru", "lfu"] = _default_policy  # user policy
    _cached_functions: List[Callable] = []

    @classmethod
    def _get(cls, dic, key, policy):
        """Get the cached object under the implemented caching policy.
        Used on a cache hit to retrieve the object.
        """
        obj = dic[key]
        if policy == "lru":
            dic.move_to_end(key)
        obj.increment()  # update stats
        return obj

    @classmethod
    def _pop(cls, dic, policy):
        """Remove one cached item as per the implemented caching policy.
        Used on a cache miss to store a new object.
        """
        if policy == "lfu":
            keys = list(dic)
            idx = np.argmin([item.counter for item in dic.values()])
            dic.move_to_end(keys[idx], last=False)
        dic.popitem(last=False)

    @classmethod
    def cache(
            cls,
            func: Callable = None,
            *,
            maxsize: int = _maxsize,
            policy: Literal["fifo", "lru", "lfu"] = _policy
    ) -> Callable:
        """Wrapper to cache the output of a function.

        Binds the arguments to the function signature and hashes the created
        dictionary with :func:`~hash_`.

        Arguments
        ---------
        func
            Function to be wrapped.
        maxsize
            Maximum cache size for the wrapped function. The default is
            :attr:`~Caching.maxsize`.
        policy
            Cache retention policy. The default is :attr:`~Caching.policy`.

        Returns
        -------

            Wrapped function.
        """
        if maxsize < 0:
            raise ValueError(
                "`maxsize` should be larger than zero. "
                "To disable caching, use `Caching.disable()`.")
        if policy not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")

        if func is None:
            # `@cache()` with parentheses
            return functools.partial(
                cls.cache, maxsize=maxsize, policy=policy)

        # Store the signature: it is accessed to make a key.
        func.__signature__ = signature(func)
        # assign caching attributes to decorated function
        func.cache_info = CacheInfo(maxsize=maxsize, policy=policy)
        func.clear_cache = func.cache_info._clear_cache
        cls._cached_functions.append(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cls._enabled:
                # caching is disabled
                return func(*args, **kwargs)

            key = hash_(func.__signature__.bind(*args, **kwargs).arguments)
            # shorthand access
            caches = func.cache_info._caches
            maxsize = func.cache_info.maxsize
            policy = func.cache_info.policy

            with RLock():
                if key in caches:
                    # output has been cached; update stats and return it
                    out = cls._get(caches, key, policy)
                    func.cache_info.hits += 1
                    return out.item

            with RLock():
                while len(caches) >= maxsize:
                    # output not cached and no space available, so remove
                    # items as per the caching policy until there is space
                    cls._pop(caches, policy)

            # cache new entry and update stats
            out = CachedObject(func(*args, **kwargs))
            caches[key] = out
            func.cache_info.misses += 1
            return out.item

        return wrapper

    @classmethod
    def enable(cls):
        """Enable caching throughout the library."""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable caching throughout the library."""
        cls._enabled = False

    @classmethod
    def reset(cls):
        """Reset all caching settings to defaults and clear all caches."""
        cls.maxsize = cls._default_maxsize
        cls.policy = cls._default_policy

    @classmethod
    def clear_cache(cls):
        """Clear all caches throughout the library."""
        [func.clear_cache() for func in cls._cached_functions]


cache = Caching.cache


class CacheInfo:
    """Container that holds stats for caching.
    Assigned as an attribute to every cached function as
    :attr:`func.cache_info`.

    Parameters
    ----------
    maxsize
        Maximum number of caches to store.
    policy
        Cache retention policy.
    """
    maxsize: int
    policy: Literal["fifo", "lru", "lfu"]
    hits: int
    """Number of times the function has been bypassed (cache hit)."""
    misses: int
    """Number of times the function has computed something (cache miss)."""

    def __init__(
            self,
            *,
            maxsize: int = Caching.maxsize,
            policy: Literal["fifo", "lru", "lfu"] = Caching.policy
    ):
        self._caches = OrderedDict()
        self.maxsize = maxsize
        self.policy = policy
        self.hits = self.misses = 0

    @property
    def current_size(self) -> int:
        """Current size of the cache dictionary."""
        return len(self._caches)

    def __repr__(self):
        s = f"<{self.__class__.__name__}>"
        for par, val in self.__dict__.items():
            if not par.startswith("_"):
                s += f"\n\t{par} = {val!r}"
        s += f"\n\tcurrent_size = {self.current_size!r}"
        return s

    def _clear_cache(self) -> None:
        """Reset cache for this function only.

        :meta public:
        """
        self._caches = OrderedDict()
        self.hits = self.misses = 0


class CachedObject:
    """Container for the cached object.

    This is what is actually stored in the cache register, rather than the bare
    object that is cached. In this way, the caching framework is fully agnostic
    to the internal CCL types.

    Parameters
    ----------
    obj
        Any object to be cached.

    Attributes
    ----------
    item : Any
        The cached object.
    counter : int
        Number of times the cached item has been retrieved (cache hits).
    """

    def __init__(self, obj: Any):
        self.item = obj
        self.counter = 0

    def __repr__(self):
        s = f"CachedObject(counter={self.counter})"
        return s

    def increment(self) -> None:
        """Increment the cache hit counter."""
        self.counter += 1

    def reset(self) -> None:
        """Reset the cache hit counter."""
        self.counter = 0
