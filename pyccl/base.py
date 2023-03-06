import sys
from collections import OrderedDict
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
    raise TypeError(f"Hashing for {type(obj)} not implemented.")


def hash_(obj):
    """Generic hash method, which changes between processes."""
    digest = hash(repr(_to_hashable(obj))) + sys.maxsize + 1
    return digest
