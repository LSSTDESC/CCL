__all__ = ("hash_", "Caching", "cache", "CacheInfo", "CachedObject",)

import sys
import functools
from collections import OrderedDict
from inspect import signature
from _thread import RLock

import numpy as np


def _to_hashable(obj):
    """Make unhashable objects hashable in a consistent manner."""

    if isinstance(obj, (int, float, str)):
        # Strings and Numbers are hashed directly.
        return obj

    elif hasattr(obj, "__iter__"):
        # Encapsulate all the iterables to quickly discard as needed.

        if isinstance(obj, np.ndarray):
            # Numpy arrays: Convert the data buffer to a byte string.
            return obj.tobytes()

        elif isinstance(obj, dict):
            # Dictionaries: Build a tuple from key-value pairs,
            # where all values are converted to hashables.
            out = {key: _to_hashable(value) for key, value in obj.items()}
            # Sort unordered dictionaries for hash consistency.
            if isinstance(obj, OrderedDict):
                return tuple(out.items())
            return tuple(sorted(out.items()))

        else:
            # Iterables: Build a tuple from values converted to hashables.
            out = [_to_hashable(item) for item in obj]
            return tuple(out)

    elif hasattr(obj, "__hash__"):
        # Hashables: Just return the object.
        return obj

    # NotImplemented: Can't hash safely, so raise TypeError.
    # Note: This will never be triggered since `type` has a repr slot wrapper.
    raise TypeError(f"Hashing for {type(obj)} not implemented.")


def hash_(obj):
    """Generic hash method, which changes between processes.
    It is designed to hash every type, even those that are by default
    unhashable, through their representation string.
    """
    digest = hash(repr(_to_hashable(obj))) + sys.maxsize + 1
    return digest


class _CachingMeta(type):
    """Implement ``property`` to a ``classmethod`` for ``Caching``."""
    # NOTE: Only in 3.8 < py < 3.11 can `classmethod` wrap `property`.
    # https://docs.python.org/3.11/library/functions.html#classmethod
    @property
    def maxsize(cls):
        """Capacity of the cache registry."""
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
    def policy(cls):
        """Cache retention policy."""
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


class Caching(metaclass=_CachingMeta):
    """Utility class that implements the infrastructure for caching.

    Attributes
    ----------
    maxsize : int
        Maximum number of caches to store. If the registry is full, new
        caches are assigned according to the set cache retention policy.
    policy : {'fifo', 'lru', 'lfu'}
        Cache retention policy.
    """
    _enabled: bool = False
    _policies: list = ['fifo', 'lru', 'lfu']
    _default_maxsize: int = 128   # class default maxsize
    _default_policy: str = 'lru'  # class default policy
    _maxsize = _default_maxsize   # user-defined maxsize
    _policy = _default_policy     # user-defined policy
    _cached_functions: list = []

    @classmethod
    def _get_key(cls, func, *args, **kwargs):
        """Calculate the hex hash from arguments and keyword arguments."""
        # get a dictionary of default parameters
        params = func.cache_info._signature.parameters
        # get a dictionary of the passed parameters
        passed = {**dict(zip(params, args)), **kwargs}
        # discard the values equal to the default
        defaults = {param: value.default for param, value in params.items()}
        return hex(hash_({**defaults, **passed}))

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
    def _decorator(cls, func, maxsize, policy):
        # assign caching attributes to decorated function
        func.cache_info = CacheInfo(func, maxsize=maxsize, policy=policy)
        func.clear_cache = func.cache_info._clear_cache
        cls._cached_functions.append(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cls._enabled:
                return func(*args, **kwargs)

            key = cls._get_key(func, *args, **kwargs)
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
    def cache(cls, func=None, *, maxsize=_maxsize, policy=_policy):
        """Cache the output of the decorated function, using the input
        arguments as a proxy to build a hash key.

        Arguments
        ---------
        func : function
            Function to be wrapped.
        maxsize : int, optional
            Maximum cache size for the wrapped function. The default is 128.
        policy : {'fifo', 'lru', 'lfu'}, optional
            Cache retention policy. When the registry reaches ``maxsize`` and
            a new object needs to be stored, decide which cached object will
            be discarded. The default is ``'lru'``.

            * ``'fifo'``: first-in-first-out,
            * ``'lru'``: least-recently-used,
            * ``'lfu'``: least-frequently-used.
        """
        if maxsize < 0:
            raise ValueError(
                "`maxsize` should be larger than zero. "
                "To disable caching, use `Caching.disable()`.")
        if policy not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")

        if func is None:
            # `@cache` with parentheses
            return functools.partial(
                cls._decorator, maxsize=maxsize, policy=policy)
        # `@cache()` without parentheses
        return cls._decorator(func, maxsize=maxsize, policy=policy)

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
    Assigned to every cached function as ``function.cache_info``.

    Parameters
    ----------
    func : function
        Function in which an instance of this class will be assigned.
        Used to access the function signature.
    maxsize : int
        Maximum number of caches to store.
    policy :  {'fifo', 'lru', 'lfu'}
        Cache retention policy.

    Attributes
    ----------
    maxsize

    policy

    hits : int
        Number of times the function has been bypassed (cache hit).
    misses : int
        Number of times the function has computed something (cache miss).
    current_size : int
        Current size of cache dictionary.
    """

    def __init__(self, func, maxsize=Caching.maxsize, policy=Caching.policy):
        # we store the signature of the function on import
        # as it is the most expensive operation (~30x slower)
        self._signature = signature(func)
        self._caches = OrderedDict()
        self.maxsize = maxsize
        self.policy = policy
        self.hits = self.misses = 0

    @property
    def current_size(self):
        """Current size of the cache dictionary."""
        return len(self._caches)

    def __repr__(self):
        s = f"<{self.__class__.__name__}>"
        for par, val in self.__dict__.items():
            if not par.startswith("_"):
                s += f"\n\t {par} = {val!r}"
        s += f"\n\t current_size = {self.current_size!r}"
        return s

    def _clear_cache(self):
        # Reset cache for this function only.
        self._caches = OrderedDict()
        self.hits = self.misses = 0


class CachedObject:
    """Container for the cached object.

    This is what is actually stored in the cache registry, rather than the bare
    object that is cached. In this way, the caching framework is fully agnostic
    to the internal CCL types.

    Parameters
    ----------
    obj : object
        Any object to be cached.

    Attributes
    ----------
    item : object
        The cached object.
    counter : int
        Number of times the cached item has been retrieved (cache hits).
    """
    counter: int = 0

    def __init__(self, obj):
        self.item = obj

    def __repr__(self):
        s = f"CachedObject(counter={self.counter})"
        return s

    def increment(self):
        """Increment the cache hit counter."""
        self.counter += 1

    def reset(self):
        """Reset the cache hit counter."""
        self.counter = 0
