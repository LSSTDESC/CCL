import numpy as np
import pytest
import pyccl as ccl


def kf(k):
    return (k/0.1)**(-1)


def af(a):
    return np.exp(a-1)


def fka1f(k, a):
    return kf(k)*af(a)**2


def fka2f(k, a):
    return (kf(k))**2.1*af(a)**2


def tkkaf(k1, k2, a):
    return fka1f(k1, a)*fka2f(k2, a)


def get_arrays(islog=True):
    a_arr = np.linspace(0.05, 1., 10)
    k_arr = np.geomspace(1E-4, 1E2, 10)
    lk_arr = np.log(k_arr)
    fka1_arr = np.array([fka1f(k_arr, a) for a in a_arr])
    fka2_arr = np.array([fka2f(k_arr, a) for a in a_arr])
    tkka_arr = np.array([tkkaf(k_arr[None, :],
                               k_arr[:, None], a)
                         for a in a_arr])
    if islog:
        fka1_arr = np.log(fka1_arr)
        fka2_arr = np.log(fka2_arr)
        tkka_arr = np.log(tkka_arr)
    return (a_arr, lk_arr, fka1_arr,
            fka2_arr, tkka_arr)


def test_tk3d_errors():
    """
    Test initialization of Pk2D objects
    """

    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()

    # Decreasing a
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr[::-1], lk_arr=lk_arr, tkk_arr=tkka_arr)
    # Decreasing lk
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr[::-1], tkk_arr=tkka_arr)
    # Non monotonic
    a2 = a_arr.copy()
    a2[1] = a2[0]
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a2, lk_arr=lk_arr, tkk_arr=tkka_arr)

    # If no input
    with pytest.raises(TypeError):
        ccl.Tk3D()

    # No input tkk or fkas
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr)

    # Missing one fka factor
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk2_arr=fka2_arr)

    # fka has wrong shape
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=tkka_arr)

    # tkka has wrong shape
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=fka1_arr)

    # Wrong extrapolation orders
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr,
                 extrap_order_hik=-1)
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr,
                 extrap_order_lok=2)


def test_tk3d_smoke():
    """Make sure it works once."""
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp1 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=fka1_arr,
                    pk2_arr=fka2_arr)
    tsp2 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)
    assert not np.isnan(tsp1.eval(1E-2, 0.5))
    assert not np.isnan(tsp2.eval(1E-2, 0.5))


def test_tk3d_eval_errors():
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=fka1_arr,
                   pk2_arr=fka2_arr)
    with pytest.raises(TypeError):
        tsp.eval(1E-2, np.array([0.1]))


def test_tk3d_delete():
    """Check that ccl.Tk3D.__del__ works."""
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=fka1_arr,
                   pk2_arr=fka2_arr)
    # This should not cause an ignored exception
    del tsp


@pytest.mark.parametrize('is_product', [True, False])
def test_tk3d_eval(is_product):
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    if is_product:
        tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=fka1_arr,
                       pk2_arr=fka2_arr)
    else:
        tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)

    # Test at single point
    ktest = 0.7
    atest = 0.5
    ptrue = tkkaf(ktest, ktest, atest)
    phere = tsp.eval(ktest, atest)
    assert np.allclose(phere, ptrue, rtol=1e-6)

    ktest = 5E-5
    atest = 0.5
    ptrue = tkkaf(ktest, ktest, atest)
    phere = tsp.eval(ktest, atest)
    assert np.allclose(phere, ptrue, rtol=1e-6)

    # Test at array of points
    ktest = np.logspace(-3, 1, 10)
    ptrue = tkkaf(ktest[None, :], ktest[:, None], atest)
    phere = tsp.eval(ktest, atest)
    assert np.allclose(phere.flatten(), ptrue.flatten(), rtol=1E-6)
