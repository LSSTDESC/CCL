import sys
import functools
from collections import OrderedDict
import hashlib
import numpy as np
from inspect import isclass


class Hashing:
    """Container class which implements hashing consistently.

    Attributes:
        consistent (``bool``):
            If False, hashes of different processes are randomly salted.
            Defaults to True for consistent hash values across processes.
    """
    consistent: bool = True

    @classmethod
    def _finalize(cls, obj, /):
        """Alphabetically sort all dictionaries except ordered dictionaries.
        """
        if isinstance(obj, OrderedDict):
            return tuple(obj)
        return tuple(sorted(obj))

    @classmethod
    def to_hashable(cls, obj, /):
        """Make unhashable objects hashable in a consistent manner."""
        if isclass(obj):
            return obj.__qualname__
        elif isinstance(obj, (tuple, list, set)):
            return tuple([cls.to_hashable(item) for item in obj])
        elif isinstance(obj, np.ndarray):
            return obj.tobytes()
        elif isinstance(obj, dict):
            dic = dict.fromkeys(obj.keys())
            for key, value in obj.items():
                dic[key] = cls.to_hashable(value)
            return cls._finalize(dic.items())
        # nothing left to do; just return the object
        return obj

    @classmethod
    def _hash_consistent(cls, obj, /):
        """Calculate consistent hash value for an input object."""
        hasher = hashlib.md5()
        hasher.update(repr(cls.to_hashable(obj)).encode())
        return int(hasher.digest().hex(), 16)

    @classmethod
    def _hash_generic(cls, obj, /):
        """Generic hash method, which changes between processes."""
        digest = hash(repr(cls.to_hashable(obj))) + sys.maxsize + 1
        return digest

    @classmethod
    def hash_(cls, obj, /):
        if not cls.consistent:
            return cls._hash_generic(obj)
        return cls._hash_consistent(obj)


hash_ = Hashing.hash_


class CachedObject:
    """A cached object container."""
    counter: int = 0

    def __init__(self, obj):
        self.item = obj

    def __repr__(self):
        s = f"CachedObject(counter={self.counter})"
        return s

    def increment(self):
        self.counter += 1

    def reset(self):
        self.counter = 0


class _ClassPropertyMeta(type):
    """Implement `property` to a `classmethod`."""
    # TODO: in py39+ decorators `classmethod` and `property` can be combined
    @property
    def maxsize(cls):
        return cls._maxsize

    @maxsize.setter
    def maxsize(cls, value):
        cls._maxsize = value
        for func in cls._caches.keys():
            func._cache["maxsize"] = value

    @property
    def policy(cls):
        return cls._policy

    @policy.setter
    def policy(cls, value):
        if value not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")
        if (cls._policy == "lfu") and (not value == "lfu"):
            # reset counter if current policy is lfu
            # otherwise new objects are prone to being discarded
            for dic in cls._caches.values():
                for item in dic.values():
                    item.reset()
        cls._policy = value
        for func in cls._caches.keys():
            func._cache["policy"] = value


class Caching(metaclass=_ClassPropertyMeta):
    """Infrastructure to hold cached objects.

    Caching is used for pre-computed objects that are expensive to compute.

    Attributes:
        maxsize (``int``):
            Maximum number of caches. If the dictionary is full, new caches
            are assigned according to the set cache retention policy.
        policy (``'fifo'``, ``'lru'``, ``'lfu'``):
            Cache retention policy.
    """
    _caches = {}
    _enabled = True
    _policies = ['fifo', 'lru', 'lfu']
    _maxsize = 64
    _policy = 'lru'

    @classmethod
    def _get_hash(cls, *args, **kwargs):
        """Calculate the hex hash from the combination the passed arguments
        and keyword arguments.
        """
        total_hash = sum([hash_(obj) for obj in [args, kwargs]])
        return hex(hash_(total_hash))

    @classmethod
    def _get(cls, dic, key, policy):
        """Get the cached object container
        under the implemented caching policy.
        """
        obj = dic[key]
        if policy == "lru":
            dic.move_to_end(key)
        elif policy == "lfu":
            obj.increment()
        return obj

    @classmethod
    def _pop(cls, dic, policy):
        """Remove one cached item as per the implemented caching policy."""
        if policy == "lfu":
            keys = list(dic)
            idx = np.argmin([item.counter for item in dic.values()])
            dic.move_to_end(keys[idx], last=False)
        dic.pop(next(iter(dic)))

    @classmethod
    def _decorator(cls, func, maxsize, policy):
        settings = {'maxsize': maxsize, 'policy': policy}
        func._cache = settings
        cls._caches[func] = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cls._enabled:
                return func(*args, **kwargs)

            # initialize entry
            key = cls._get_hash(*args, **kwargs)
            caches = cls._caches[func]
            maxsize = func._cache['maxsize']
            policy = func._cache['policy']

            if key in caches:
                # output has been cached
                out = cls._get(caches, key, policy)
                return out.item

            while len(caches) >= maxsize:
                # output not cached and no space available, so remove
                # items as per the caching policy until there is space
                cls._pop(caches, policy)

            # cache new entry
            out = CachedObject(func(*args, **kwargs))
            caches[key] = out
            return out.item

        return wrapper

    @classmethod
    def cache(cls, func=None, /, *, maxsize=None, policy=None):
        """Cache the output of the decorated function.

        Arguments:
            func (``function``):
                Function to be decorated.
            maxsize (``int`` or ``None``):
                Maximum cache size for the decorated function.
                If None, defaults to ``pyccl.Caching.maxsize``.
            policy (``'fifo'``, ``'lru'``, ``'lfu'``):
                Cache retention policy. When the storage reaches maxsize
                decide which cached object will be deleted. Default is 'lru'.\n
                'fifo': first-in-first-out,\n
                'lru': least-recently-used,\n
                'lfu': least-frequently-used.
        """
        if maxsize is None:
            maxsize = cls.maxsize
        if policy is None:
            policy = cls.policy
        elif policy not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")

        def decorator(func):
            return cls._decorator(func, maxsize, policy)

        # Check if usage is with @cache or @cache()
        if func is None:
            # @cache() with parentheses
            return decorator
        # @cache without parentheses
        return decorator(func)

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def toggle(cls):
        cls._enabled = not cls._enabled

    @classmethod
    def reset(cls):
        cls.maxsize = 64
        cls.policy = 'lru'


cache = Caching.cache
