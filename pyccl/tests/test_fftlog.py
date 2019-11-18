import numpy as np
import pytest
import pyccl as ccl


@pytest.mark.parametrize('dim', [2, 3])
def test_fftlog_exact(dim):
    # The d-D Hankel transform of k^{-d/2} is
    # (2 * \pi * r)^{-d/2}
    def f(k):
        return k**(-dim / 2.)

    def fr(r):
        return (2 * np.pi * r)**(-dim / 2.)

    nk = 1024
    k_arr = np.logspace(-4, 4, nk)
    fk_arr = f(k_arr)

    status = 0
    result, status = ccl.ccllib.fftlog_transform(k_arr, fk_arr,
                                                 dim, 2 * k_arr.size,
                                                 status)
    r_arr, fr_arr = result.reshape([2, k_arr.size])
    fr_arr_pred = fr(r_arr)
    res = np.fabs(fr_arr / fr_arr_pred -1)
    assert np.all(res < 1E-10)
