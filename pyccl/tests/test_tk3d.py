import numpy as np
import pytest
import pyccl as ccl
from .test_cclobject import check_eq_repr_hash


def test_Tk3D_eq_repr_hash():
    # Test eq, repr, hash for Tk3D.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    cosmo.compute_linear_power()
    PK1 = cosmo.get_linear_power()

    # 1. Using a factorizable Tk3D object.
    a_arr, lk_arr, pk_arr = PK1.get_spline_arrays()
    TK1 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                   pk1_arr=pk_arr, pk2_arr=pk_arr, is_logt=False)
    TK2 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                   pk1_arr=pk_arr, pk2_arr=pk_arr, is_logt=False)
    assert check_eq_repr_hash(TK1, TK2)

    TK3 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                   pk1_arr=2*pk_arr, pk2_arr=2*pk_arr, is_logt=False)
    assert check_eq_repr_hash(TK1, TK3, equal=False)

    # 2. Using a non-factorizable Tk3D object.
    a_arr_2 = np.arange(0.5, 0.9, 0.1)
    lk_arr_2 = np.linspace(-2, 1, 8)
    TK4 = ccl.Tk3D(
        a_arr=a_arr_2, lk_arr=lk_arr_2,
        tkk_arr=np.ones((a_arr_2.size, lk_arr_2.size, lk_arr_2.size)))
    TK5 = ccl.Tk3D(
        a_arr=a_arr_2, lk_arr=lk_arr_2,
        tkk_arr=np.ones((a_arr_2.size, lk_arr_2.size, lk_arr_2.size)))
    assert check_eq_repr_hash(TK4, TK5)

    TK6 = ccl.Tk3D(
        a_arr=a_arr_2, lk_arr=lk_arr_2,
        tkk_arr=2*np.ones((a_arr_2.size, lk_arr_2.size, lk_arr_2.size)))
    assert check_eq_repr_hash(TK4, TK6, equal=False)

    # edge-case: comparing different types
    assert check_eq_repr_hash(TK1, 1, equal=False)

    # edge-case: empty objects
    tka1, tka2 = [ccl.Tk3D.__new__(ccl.Tk3D) for _ in range(2)]
    assert check_eq_repr_hash(tka1, tka2)

    # edge-case: only one Tk is factorizable (exits early)
    assert check_eq_repr_hash(TK1, TK4, equal=False)

    # edge-case: different extrapolation orders
    a_arr, lk_arr, pk_arr = PK1.get_spline_arrays()
    t1 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=pk_arr, pk2_arr=pk_arr,
                  is_logt=False, extrap_order_lok=0)
    t2 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=pk_arr, pk2_arr=pk_arr,
                  is_logt=False, extrap_order_lok=1)
    assert check_eq_repr_hash(t1, t2, equal=False)


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
        ccl.Tk3D(a_arr=a_arr[::-1], lk_arr=lk_arr,
                 tkk_arr=tkka_arr)
    # Decreasing lk
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr[::-1],
                 tkk_arr=tkka_arr)
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
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                 tkk_arr=tkka_arr, extrap_order_hik=-1)
    with pytest.raises(ValueError):
        ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                 tkk_arr=tkka_arr, extrap_order_lok=2)


def test_tk3d_smoke():
    """Make sure it works once."""
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp1 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, pk1_arr=fka1_arr,
                    pk2_arr=fka2_arr)
    tsp2 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)
    assert not np.isnan(tsp1(1E-2, 0.5))
    assert not np.isnan(tsp2(1E-2, 0.5))


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
    phere = tsp(ktest, atest)
    assert np.allclose(phere, ptrue, atol=0, rtol=1e-6)

    ktest = 5E-5
    atest = 0.5
    ptrue = tkkaf(ktest, ktest, atest)
    phere = tsp(ktest, atest)
    assert np.allclose(phere, ptrue, atol=0, rtol=1e-6)

    # Test at array of points
    ktest = np.logspace(-3, 1, 10)
    ptrue = tkkaf(ktest[None, :], ktest[:, None], atest)
    phere = tsp(ktest, atest)
    assert np.allclose(phere, ptrue, atol=0, rtol=1e-6)


def test_tk3d_call():
    # Test `__call__` and `__bool__`
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)
    assert bool(tsp) is tsp.has_tsp
    assert np.allclose(np.array([tsp(np.exp(lk_arr), a) for a in a_arr]),
                       tsp(np.exp(lk_arr), a_arr), rtol=1e-15)


@pytest.mark.parametrize('is_product', [True, False])
def test_tk3d_spline_arrays(is_product):
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    if is_product:
        tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                       pk1_arr=fka1_arr, pk2_arr=fka2_arr)
    else:
        tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)

    a_get, lk_get1, lk_get2, out = tsp.get_spline_arrays()
    assert np.allclose(a_get, a_arr, rtol=1e-15)
    assert np.allclose(lk_get1, lk_arr, rtol=1e-15)
    assert np.allclose(lk_get2, lk_arr, rtol=1e-15)

    if is_product:
        assert np.allclose(np.log(out[0]), fka1_arr, rtol=1e-15)
        assert np.allclose(np.log(out[1]), fka2_arr, rtol=1e-15)
    else:
        assert np.allclose(np.log(out[0]), tkka_arr, rtol=1e-15)


def test_tk3d_spline_arrays_raises():
    (a_arr, lk_arr, fka1_arr, fka2_arr, tkka_arr) = get_arrays()
    tsp = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkka_arr)

    ccl.lib.f3d_t_free(tsp.tsp)
    delattr(tsp, "tsp")

    with pytest.raises(ValueError):
        tsp.get_spline_arrays()
