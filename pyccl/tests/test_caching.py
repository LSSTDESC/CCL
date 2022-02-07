import pyccl as ccl
import numpy as np
from time import time


s8_arr = np.linspace(0.75, 0.95, 3)


def get_cosmo(sigma8):
    return ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96,
                         sigma8=sigma8)


def cosmo_create_and_compute_linpow(sigma8):
    cosmo = get_cosmo(sigma8)
    cosmo.compute_linear_power()
    return cosmo


def timeit_(sigma8):
    t0 = time()
    cosmo_create_and_compute_linpow(sigma8)
    return time() - t0


def test_times():
    """Verify that caching is happening.
    (cached calls are ~5000 faster than calls to Boltzmann solvers)
    """
    t1 = np.array([timeit_(s8) for s8 in s8_arr])
    t2 = np.array([timeit_(s8) for s8 in s8_arr])
    assert np.all(t1/t2 > 1000)


def test_caching_switches():
    """Test that the Caching switches work as intended."""
    assert ccl.Caching._enabled
    assert ccl.Caching.maxsize == 64
    ccl.Caching.disable()
    assert not ccl.Caching._enabled
    ccl.Caching.enable()
    assert ccl.Caching._enabled
    ccl.Caching.toggle()
    assert not ccl.Caching._enabled
    ccl.Caching.enable(maxsize=3)
    assert ccl.Caching._enabled
    assert ccl.Caching.maxsize == 3


def test_caching_overfull():
    """Test that Caching rolls as intended when overfull."""
    # To save time, we test caching by limiting the maximum cache size
    # from 64 (default) to 3. We cache Comologies with different sigma8.
    assert len(ccl.Caching._caches) == ccl.Caching.maxsize

    t1 = timeit_(sigma8=0.75)  # this is the oldest cached pk
    cosmo_create_and_compute_linpow(0.81)  # newly cached; oldest one discarded
    t2 = timeit_(sigma8=0.75)  # newly cached
    assert t2/t1 > 1000

    # now, reset the switch
    ccl.Caching.maxsize = 64
