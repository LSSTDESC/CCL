import sys
from collections import OrderedDict
import hashlib
import numpy as np


class Hashing:
    """Container class which implements hashing consistently.

    Attributes:
        consistent (``bool``):
            If False, hashes of different processes are randomly salted.
            Defaults to False for speed, but hashes differ across processes.

    .. note::

        Consistent (unsalted) hashing between different processes comes at
        the expense of extra computation time (~200x slower).
        Buitin ``hash`` computes in O(100 ns) while using hashlib with md5
        computes in O(20 μs).
    """
    consistent: bool = False

    @classmethod
    def _to_hashable(cls, obj):
        """Make unhashable objects hashable in a consistent manner."""

        if hasattr(obj, "__iter__"):
            # Encapsulate all the iterables to quickly discard
            # and go to numbers hashing in the second clause.

            if isinstance(obj, np.ndarray):
                # Numpy arrays: Convert the data buffer to a byte string.
                return obj.tobytes()

            elif isinstance(obj, dict):
                # Dictionaries: Build a tuple from key-value pairs,
                # where all values are converted to hashables.
                out = dict.fromkeys(obj)
                for key, value in obj.items():
                    out[key] = cls._to_hashable(value)
                # Sort unordered dictionaries for hash consistency.
                if isinstance(obj, OrderedDict):
                    return tuple(obj)
                return tuple(sorted(obj))

            else:
                # Iterables: Build a tuple from values converted to hashables.
                out = [cls._to_hashable(item) for item in obj]
                return tuple(out)

        elif hasattr(obj, "__hash__"):
            # Hashables: Just return the object.
            return obj

        # NotImplemented: Can't hash safely, so raise TypeError.
        raise TypeError(f"Hashing for {type(obj)} not implemented.")

    @classmethod
    def _hash_consistent(cls, obj):
        """Calculate consistent hash value for an input object."""
        hasher = hashlib.md5()
        hasher.update(repr(cls.t_o_hashable(obj)).encode())
        return int(hasher.digest().hex(), 16)

    @classmethod
    def _hash_generic(cls, obj):
        """Generic hash method, which changes between processes."""
        digest = hash(repr(cls._to_hashable(obj))) + sys.maxsize + 1
        return digest

    @classmethod
    def hash_(cls, obj):
        if not cls.consistent:
            return cls._hash_generic(obj)
        return cls._hash_consistent(obj)


hash_ = Hashing.hash_
