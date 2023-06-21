import numpy as np
import pytest
from pyccl.pyutils import _fftlog_transform, _fftlog_transform_general


def fk(k, alpha, mu, dim):
    return k ** (-alpha)


def fr(r, alpha, mu, dim):
    from scipy.special import gamma

    g1 = gamma(0.5 * (dim - alpha + mu))
    g2 = gamma(0.5 * (alpha + mu))
    den = np.pi ** (dim / 2.0) * 2**(alpha)
    return g1 / (g2 * den * r ** (dim - alpha))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mu", [0, 2])
@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.8])
def test_fftlog_plaw(dim, mu, alpha):
    # The d-D Hankel transform of k^{-alpha} is
    # \Gamma[(d - \alpha + \mu) / 2] /
    # \Gamma[(\alpha + \mu) / 2] /
    # (\pi^{d/2} * 2^\alpha * r^{d-\alpha})

    nk = 1024
    k_arr = np.logspace(-4, 4, nk)
    fk_arr = fk(k_arr, alpha, mu, dim)

    r_arr, fr_arr = _fftlog_transform(k_arr, fk_arr, dim, mu, -alpha)

    if dim == 2:
        ccl_r, ccl_Fr = _fftlog_transform_general(k_arr,
                                                  fk_arr * (k_arr**(dim)),
                                                  mu + 1,
                                                  -alpha + dim,
                                                  0, 0.0, 0.0)
        ccl_Fr /= 2 * np.pi * (dim + mu - alpha)
    elif dim == 3:
        ccl_r, ccl_Fr = _fftlog_transform_general(k_arr,
                                                  fk_arr * (k_arr**(dim)),
                                                  mu,
                                                  -alpha + dim,
                                                  1, 0.0, 0.0)
        ccl_Fr /= 2 * np.pi**2
    else:
        print('unsupported _fftlog_transform test!')
        assert (0)
    fr_arr_pred = fr(ccl_r, alpha, mu, dim)
    res = np.fabs(ccl_Fr / fr_arr_pred - 1)
    assert np.all(res < 1e-10)
    fr_arr_pred = fr(r_arr, alpha, mu, dim)
    res = np.fabs(fr_arr / fr_arr_pred - 1)
    assert np.all(res < 1e-10)


def test_fftlog_shapes():
    nk = 1024
    nt = 4
    k_arr = np.logspace(-4, 4, nk)
    fk_arr = []
    for i in range(nt):
        fk_arr.append(fk(k_arr, 1.5, 0, 2))
    fk_arr = np.array(fk_arr)

    # Scalar rs
    with pytest.raises(ValueError):
        _fftlog_transform(k_arr[0], fk_arr, 2, 0, 1.5)
    with pytest.raises(ValueError):
        _fftlog_transform_general(k_arr[0], fk_arr, 0, 1.5, 0, 0.0, 0.0)

    # Scalar frs
    with pytest.raises(ValueError):
        _fftlog_transform(k_arr, fk_arr[0][0], 2, 0, 1.5)
    with pytest.raises(ValueError):
        _fftlog_transform_general(k_arr, fk_arr[0][0], 0, 1.5, 0, 0.0, 0.0)

    # Wrong rs
    with pytest.raises(ValueError):
        _fftlog_transform(k_arr[1:], fk_arr, 2, 0, 1.5)
    with pytest.raises(ValueError):
        _fftlog_transform_general(k_arr[1:], fk_arr, 0, 1.5, 0, 0.0, 0.0)

    # Single transform
    r, fr = _fftlog_transform(k_arr, fk_arr[0], 2, 0, 1.5)
    assert fr.shape == (nk,)
    r, fr = _fftlog_transform_general(k_arr, fk_arr[0], 0, 1.5, 0, 0.0, 0.0)
    assert fr.shape == (nk,)

    # Multiple transforms
    r, fr = _fftlog_transform(k_arr, fk_arr, 2, 0, 1.5)
    assert fr.shape == (nt, nk)
    r, fr = _fftlog_transform_general(k_arr, fk_arr, 0, 1.5, 0, 0.0, 0.0)
    assert fr.shape == (nt, nk)
