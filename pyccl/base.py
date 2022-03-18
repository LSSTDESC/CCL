import sys
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
