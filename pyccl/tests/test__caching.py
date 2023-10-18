# We use double underscore to make it the first test alphabetically.
import pytest
import pyccl as ccl
import numpy as np
from time import time

# Enable caching for this test.
DEFAULT_CACHING_STATUS = ccl.Caching._enabled

NUM = 3  # number of different Cosmologies we will check
# some unusual numbers that have not occurred before
s8_arr = np.linspace(0.753141592, 0.953141592, NUM)
# a modest speed increase - we are modest in the test to accommodate for slow
# runs; normally this is is expected to be another order of magnitude faster
SPEEDUP = 50


def get_cosmo(sigma8):
    return ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96,
                         sigma8=sigma8)


@ccl.cache(maxsize=3)
def cosmo_create_and_compute_linpow(sigma8):
    cosmo = get_cosmo(sigma8)
    cosmo.compute_linear_power()
    return cosmo


def timeit_(sigma8):
    t0 = time()
    cosmo_create_and_compute_linpow(sigma8)
    return time() - t0


def test_caching_switches():
    """Test that the Caching switches work as intended."""
    assert ccl.Caching._maxsize == ccl.Caching._default_maxsize
    ccl.Caching.maxsize = 128
    assert ccl.Caching._maxsize == 128
    ccl.Caching.disable()
    assert not ccl.Caching._enabled
    ccl.Caching.enable()
    assert ccl.Caching._enabled
    ccl.Caching.disable()


def test_times():
    """Verify that caching is happening.
    Return time for querying the Boltzmann code goes from O(1s) to O(5ms).
    """
    # If we disable caching, t1 and t2 will be of the same order of magnitude.
    # No need to run through the entire s8_arr.
    ccl.Caching.disable()
    t1 = np.array([timeit_(s8) for s8 in s8_arr[:1]])
    t2 = np.array([timeit_(s8) for s8 in s8_arr[:1]])
    assert np.abs(np.log10(t2/t1)) < 1.0
    # But if caching is enabled, the second call will be much faster.
    ccl.Caching.enable()
    t1 = np.array([timeit_(s8) for s8 in s8_arr])
    t2 = np.array([timeit_(s8) for s8 in s8_arr])
    assert np.all(t1/t2 > SPEEDUP)
    ccl.Caching.disable()


def test_caching_fifo():
    """Test First-In-First-Out retention policy."""
    ccl.Caching.enable()
    # To save time, we test caching by limiting the maximum cache size
    # from 64 (default) to 3. We cache Comologies with different sigma8.
    # By now, the caching repo will be full.
    ccl.Caching.maxsize = NUM
    func = cosmo_create_and_compute_linpow
    assert len(func.cache_info._caches) >= ccl.Caching.maxsize

    ccl.Caching.policy = "fifo"

    t1 = timeit_(sigma8=s8_arr[0])  # this is the oldest cached pk
    # create new and discard oldest
    cosmo_create_and_compute_linpow(0.42)
    t2 = timeit_(sigma8=s8_arr[0])  # cached again
    assert t2/t1 > SPEEDUP
    ccl.Caching.disable()


def test_caching_lru():
    """Test Least-Recently-Used retention policy."""
    # By now the stored Cosmologies are { s8_arr[2], 0.42, s8_arr[0]] }
    # from oldest to newest. Here, we show that we can retain s8_arr[2]
    # simply by using it and moving it to the end of the stack.
    ccl.Caching.enable()
    ccl.Caching.policy = "lru"

    t1 = timeit_(sigma8=s8_arr[2])  # moves to the end of the stack
    # create new and discard the least recently used
    cosmo_create_and_compute_linpow(0.43)
    t2 = timeit_(sigma8=s8_arr[2])  # retrieved
    assert np.abs(np.log10(t2/t1)) < 1.0
    ccl.Caching.disable()


def test_caching_lfu():
    """Test Least-Frequently-Used retention policy."""
    # Now, the stored Cosmologies are { s8_arr[0], 0.43, s8_arr[2] }
    # from oldest to newest. Here, we call each a different number of times
    # and we check that the one used the least (0.43) is discarded.
    ccl.Caching.enable()
    ccl.Caching.policy = "lfu"

    t1 = timeit_(sigma8=0.43)  # increments counter by 1
    _ = [timeit_(sigma8=s8_arr[0]) for _ in range(5)]
    _ = [timeit_(sigma8=s8_arr[2]) for _ in range(3)]
    # create new and discard the least frequently used
    cosmo_create_and_compute_linpow(0.44)
    t2 = timeit_(sigma8=0.43)  # cached again
    assert t2/t1 > SPEEDUP
    ccl.Caching.disable()


def test_cache_info():
    """Test that the CacheInfo repr gives us the expected information."""
    info = cosmo_create_and_compute_linpow.cache_info
    for text in ["maxsize", "policy", "hits", "misses", "current_size"]:
        assert text in repr(info)

    obj = list(info._caches.values())[0]
    assert "counter" in repr(obj)


def test_caching_reset():
    """Test the reset switches."""
    ccl.Caching.reset()
    assert ccl.Caching.maxsize == ccl.Caching._default_maxsize
    assert ccl.Caching.policy == ccl.Caching._default_policy
    ccl.Caching.clear_cache()
    func = cosmo_create_and_compute_linpow
    assert len(func.cache_info._caches) == 0


def test_caching_policy_raises():
    """Test that if the set policy is not correct, it raises an exception."""
    with pytest.raises(ValueError):
        @ccl.Caching.cache(maxsize=-1)
        def func1():
            return

    with pytest.raises(ValueError):
        @ccl.Caching.cache(policy="my_policy")
        def func2():
            return

    with pytest.raises(ValueError):
        ccl.Caching.maxsize = -1

    with pytest.raises(ValueError):
        ccl.Caching.policy = "my_policy"


def test_caching_parentheses():
    """Verify that ``cache`` can be used with or without parentheses."""
    func1, func2 = [lambda: None for _ in range(2)]
    ccl.cache(func1)
    ccl.cache(maxsize=10, policy="lru")(func2)
    assert all([hasattr(func, "cache_info") for func in [func1, func2]])


# Revert to defaults.
ccl.Caching._enabled = DEFAULT_CACHING_STATUS
