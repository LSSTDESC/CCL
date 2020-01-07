import numpy as np
import pytest
import pyccl as ccl


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mu', [0, 2])
@pytest.mark.parametrize('alpha', [1.2, 1.5, 1.8])
def test_fftlog_plaw(dim, mu, alpha):
    # The d-D Hankel transform of k^{-alpha} is
    # \Gamma[(d - \alpha + \mu) / 2] /
    # \Gamma[(\alpha + \mu) / 2] /
    # (\pi^{d/2} * 2^\alpha * r^{d-\alpha})
    from scipy.special import gamma

    def f(k):
        return k**(-alpha)

    def fr(r):
        g1 = gamma(0.5 * (dim - alpha + mu))
        g2 = gamma(0.5 * (alpha + mu))
        den = np.pi**(dim/2.) * 2**alpha
        return g1 / (g2 * den * r**(dim - alpha))

    nk = 1024
    k_arr = np.logspace(-4, 4, nk)
    fk_arr = f(k_arr)

    status = 0
    result, status = ccl.ccllib.fftlog_transform(1, k_arr, fk_arr,
                                                 dim, mu, -alpha,
                                                 2 * k_arr.size, status)
    assert status == 0
    r_arr, fr_arr = result.reshape([2, k_arr.size])
    fr_arr_pred = fr(r_arr)
    res = np.fabs(fr_arr / fr_arr_pred - 1)
    assert np.all(res < 1E-10)
