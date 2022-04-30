"""Test the hashing function of CCL."""
import pytest
import pyccl as ccl
import numpy as np
from collections import OrderedDict

OBJECTS = [ccl.Cosmology,  # class
           (0, 1, 2),      # tuple
           [0, 1, 2],      # list
           set([0, 1, 2]),   # set
           np.arange(3),   # array
           {0: None, 1: None, 2: None},                    # dict
           {0: None, 1: None, 2: {2.1: None, 2.2: None}},  # nested dict
           OrderedDict({0: None, 1: None, 2: None}),       # OrderedDict
           ccl.CosmologyVanillaLCDM(),  # something else
           None,                        # something else
           ]


@pytest.mark.parametrize("obj", OBJECTS)
def test_hashing_smoke(obj):
    assert isinstance(ccl.hash_(obj), int)


def test_hashing_large_array():
    # Hashing ultimately uses the representation of the object.
    # The representation of large numpy arrays only contains the start
    # and the end. We check that the entire array is considered.
    array = np.random.random(64**3).reshape(64, 64, 64)
    array2 = array.copy()
    array2[31, 31, 31] += 1.  # this is now the max value
    vmax = str(array2.max())[:6]
    assert vmax not in repr(array2)  # make sure it doesn't show
    assert ccl.hash_(array) != ccl.hash_(array2)
