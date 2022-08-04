import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_raises, assert_almost_equal, assert_allclose)
import pyccl as ccl
from pyccl import CCLWarning


def pk1d(k):
    return (k/0.1)**(-1)


def grw(a):
    return a


def pk2d(k, a):
    return pk1d(k)*grw(a)


def lpk2d(k, a):
    return np.log(pk2d(k, a))


def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.all(np.isfinite(vals))


def test_pk2d_init():
    """
    Test initialization of Pk2D objects
    """

    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

    # If no input
    assert_raises(ValueError, ccl.Pk2D)

    # Input function has incorrect signature
    assert_raises(ValueError, ccl.Pk2D, pkfunc=pk1d)
    ccl.Pk2D(pkfunc=lpk2d, cosmo=cosmo)

    # Input arrays have incorrect sizes
    lkarr = -4.+6*np.arange(100)/99.
    aarr = 0.05+0.95*np.arange(100)/99.
    pkarr = np.zeros([len(aarr), len(lkarr)])
    assert_raises(
        ValueError, ccl.Pk2D, a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr[1:])

    # Scale factor is not monotonically increasing
    assert_raises(
        ValueError, ccl.Pk2D, a_arr=aarr[::-1], lk_arr=lkarr, pk_arr=pkarr)


def test_pk2d_smoke():
    """Make sure it works once."""
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    lkarr = -4.+6*np.arange(100)/99.
    aarr = 0.05+0.95*np.arange(100)/99.
    pkarr = np.zeros([len(aarr), len(lkarr)])
    psp = ccl.Pk2D(a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr)
    assert_(not np.isnan(psp.eval(1E-2, 0.5, cosmo)))


@pytest.mark.parametrize('model', ['bbks', 'eisenstein_hu',
                                   'eisenstein_hu_nowiggles'])
def test_pk2d_from_model(model):
    cosmo_fixed = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=model)
    pk = ccl.Pk2D.from_model(cosmo_fixed, model=model)
    ks = np.geomspace(1E-3, 1E1, 128)
    for z in [0., 0.5, 2.]:
        a = 1./(1+z)
        pk1 = pk.eval(ks, a, cosmo)
        pk2 = ccl.linear_matter_power(cosmo, ks, a)
        maxdiff = np.amax(np.fabs(pk1/pk2-1))
        assert maxdiff < 1E-10


def test_pk2d_from_model_emu():
    pars = [0.3643, 0.071075, 0.55, 0.8333, 0.9167, -0.7667, 0.1944]
    cosmo_fixed = ccl.Cosmology(Omega_c=pars[0],
                                Omega_b=pars[1],
                                h=pars[2],
                                sigma8=pars[3],
                                n_s=pars[4],
                                w0=pars[5],
                                wa=pars[6],
                                Neff=3.04,
                                Omega_g=0,
                                Omega_k=0,
                                transfer_function='bbks')
    cosmo = ccl.Cosmology(Omega_c=pars[0],
                          Omega_b=pars[1],
                          h=pars[2],
                          sigma8=pars[3],
                          n_s=pars[4],
                          w0=pars[5],
                          wa=pars[6],
                          Neff=3.04,
                          Omega_g=0,
                          Omega_k=0,
                          transfer_function='bbks',
                          matter_power_spectrum='emu')
    pk = ccl.Pk2D.from_model(cosmo_fixed, model='emu')
    ks = np.geomspace(1E-3, 1E1, 128)
    for z in [0., 0.5, 2.]:
        a = 1./(1+z)
        pk1 = pk.eval(ks, a, cosmo)
        pk2 = ccl.nonlin_matter_power(cosmo, ks, a)
        maxdiff = np.amax(np.fabs(pk1/pk2-1))
        assert maxdiff < 1E-10


@pytest.mark.parametrize('model', ['bbks', 'eisenstein_hu'])
def test_pk2d_from_model_fails(model):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1E-10, n_s=0.96,
        transfer_function='boltzmann_class')
    assert_raises(ccl.CCLError, ccl.Pk2D.from_model,
                  cosmo, model=model)


def test_pk2d_from_model_raises():
    cosmo = ccl.CosmologyVanillaLCDM()
    assert_raises(ValueError, ccl.Pk2D.from_model,
                  cosmo, model='bbkss')


def test_pk2d_function():
    """
    Test evaluation of Pk2D objects
    """

    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

    psp = ccl.Pk2D(pkfunc=lpk2d, cosmo=cosmo)

    # Test at single point
    ktest = 1E-2
    atest = 0.5
    ptrue = pk2d(ktest, atest)
    phere = psp.eval(ktest, atest, cosmo)
    assert_almost_equal(np.fabs(phere/ptrue), 1., 6)
    dphere = psp.eval_dlogpk_dlogk(ktest, atest, cosmo)
    assert_almost_equal(dphere, -1., 6)

    ktest = 1
    atest = 0.5
    ptrue = pk2d(ktest, atest)
    phere = psp.eval(ktest, atest, cosmo)
    assert_almost_equal(np.fabs(phere/ptrue), 1., 6)
    dphere = psp.eval_dlogpk_dlogk(ktest, atest, cosmo)
    assert_almost_equal(dphere, -1., 6)

    # Test at array of points
    ktest = np.logspace(-3, 1, 10)
    ptrue = pk2d(ktest, atest)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)
    dphere = psp.eval_dlogpk_dlogk(ktest, atest, cosmo)
    assert_allclose(dphere, -1.*np.ones_like(dphere), 6)

    # Test input is not logarithmic
    psp = ccl.Pk2D(pkfunc=pk2d, is_logp=False, cosmo=cosmo)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)
    dphere = psp.eval_dlogpk_dlogk(ktest, atest, cosmo)
    assert_allclose(dphere, -1.*np.ones_like(dphere), 6)

    # Test input is arrays
    karr = np.logspace(-4, 2, 1000)
    aarr = np.linspace(0.01, 1., 100)
    parr = np.array([pk2d(karr, a) for a in aarr])
    psp = ccl.Pk2D(
        a_arr=aarr, lk_arr=np.log(karr), pk_arr=parr, is_logp=False)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)
    dphere = psp.eval_dlogpk_dlogk(ktest, atest, cosmo)
    assert_allclose(dphere, -1.*np.ones_like(dphere), 6)


def test_pk2d_cls():
    """
    Test interplay between Pk2D and the Limber integrator
    """

    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)
    lens1 = ccl.WeakLensingTracer(cosmo, (z, n))
    ells = np.arange(2, 10)

    # Check that passing no power spectrum is fine
    cells = ccl.angular_cl(cosmo, lens1, lens1, ells)
    assert all_finite(cells)

    # Check that passing a bogus power spectrum fails as expected
    assert_raises(
        ValueError, ccl.angular_cl, cosmo, lens1, lens1, ells, p_of_k_a=1)

    # Check that passing a correct power spectrum runs as expected
    psp = ccl.Pk2D(pkfunc=lpk2d, cosmo=cosmo)
    cells = ccl.angular_cl(cosmo, lens1, lens1, ells, p_of_k_a=psp)
    assert all_finite(cells)


def test_pk2d_parsing():
    a_arr = np.linspace(0.1, 1, 100)
    k_arr = np.geomspace(1E-4, 1E3, 1000)
    pk_arr = a_arr[:, None] * ((k_arr/0.01)/(1+(k_arr/0.01)**3))[None, :]

    psp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr),
                   pk_arr=np.log(pk_arr))

    cosmo = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr,
                   'a:b': pk_arr})
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)
    lens1 = ccl.WeakLensingTracer(cosmo, (z, n))
    ells = np.linspace(2, 100, 10)

    cls1 = ccl.angular_cl(cosmo, lens1, lens1, ells,
                          p_of_k_a=None)
    cls2 = ccl.angular_cl(cosmo, lens1, lens1, ells,
                          p_of_k_a='delta_matter:delta_matter')
    cls3 = ccl.angular_cl(cosmo, lens1, lens1, ells,
                          p_of_k_a='a:b')
    cls4 = ccl.angular_cl(cosmo, lens1, lens1, ells,
                          p_of_k_a=psp)
    assert all_finite(cls1)
    assert all_finite(cls2)
    assert all_finite(cls3)
    assert all_finite(cls4)
    assert np.all(np.fabs(cls2/cls1-1) < 1E-10)
    assert np.all(np.fabs(cls3/cls1-1) < 1E-10)
    assert np.all(np.fabs(cls4/cls1-1) < 1E-10)

    # Wrong name
    with pytest.raises(KeyError):
        ccl.angular_cl(cosmo, lens1, lens1, ells,
                       p_of_k_a='a:c')

    # Wrong type
    with pytest.raises(ValueError):
        ccl.angular_cl(cosmo, lens1, lens1, ells,
                       p_of_k_a=3)


def test_pk2d_get_spline_arrays():
    empty_pk2d = ccl.Pk2D(empty=True)

    # Pk2D needs splines defined to get splines out
    with pytest.raises(ValueError):
        empty_pk2d.get_spline_arrays()


def test_pk2d_add():
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_a = np.outer(x, np.exp(log_y))
    zarr_b = np.outer(-1*x, 4*np.exp(log_y))

    empty_pk2d = ccl.Pk2D(empty=True)
    pk2d_a = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=np.log(zarr_a),
                      is_logp=True)
    pk2d_b = ccl.Pk2D(a_arr=2*x, lk_arr=log_y, pk_arr=zarr_b,
                      is_logp=False)
    pk2d_b2 = ccl.Pk2D(a_arr=x, lk_arr=log_y+0.5, pk_arr=zarr_b,
                       is_logp=False)

    # This raises an error because the a ranges don't match
    with pytest.raises(ValueError):
        pk2d_a + pk2d_b
    # This raises an error because the k ranges don't match
    with pytest.raises(ValueError):
        pk2d_a + pk2d_b2
    # This raises an error because addition with an empty Pk2D should not work
    with pytest.raises(ValueError):
        pk2d_a + empty_pk2d

    pk2d_c = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=zarr_b,
                      is_logp=False)

    pk2d_d = pk2d_a + pk2d_c
    pk2d_d2 = pk2d_a + 1.0
    xarr_d, yarr_d, zarr_d = pk2d_d.get_spline_arrays()
    _, _, zarr_d2 = pk2d_d2.get_spline_arrays()

    assert np.allclose(x, xarr_d)
    assert np.allclose(log_y, yarr_d)
    assert np.allclose(zarr_a + zarr_b, zarr_d)
    assert np.allclose(zarr_a + 1.0, zarr_d2)

    pk2d_e = ccl.Pk2D(a_arr=x[1:-1], lk_arr=log_y[1:-1],
                      pk_arr=zarr_b[1:-1, 1:-1],
                      is_logp=False)

    # This raises a warning because the power spectra are not defined on the
    # same support
    with pytest.warns(CCLWarning):
        pk2d_f = pk2d_e + pk2d_a

    xarr_f, yarr_f, zarr_f = pk2d_f.get_spline_arrays()

    assert np.allclose((zarr_a + zarr_b)[1:-1, 1:-1], zarr_f)


def test_pk2d_mul_pow():
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_a = np.outer(x, np.exp(log_y))
    zarr_b = np.outer(-1*x, 4*np.exp(log_y))

    pk2d_a = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=np.log(zarr_a),
                      is_logp=True)
    pk2d_b = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=zarr_b,
                      is_logp=False)

    # This raises an error because multiplication is only defined for
    # float, int, and Pk2D
    with pytest.raises(TypeError):
        pk2d_a*np.array([0.1, 0.2])

    # This raises an error because exponention is only defined for
    # float and int
    with pytest.raises(TypeError):
        pk2d_a**pk2d_b

    # This raises a warning because the power spectrum is non-negative and the
    # power is non-integer
    with pytest.warns(CCLWarning):
        pk2d_b**0.5

    pk2d_g = pk2d_a * pk2d_b
    pk2d_h = 2*pk2d_a
    pk2d_i = pk2d_a**1.8

    _, _, zarr_g = pk2d_g.get_spline_arrays()
    _, _, zarr_h = pk2d_h.get_spline_arrays()
    _, _, zarr_i = pk2d_i.get_spline_arrays()

    assert np.allclose(zarr_a * zarr_b, zarr_g)
    assert np.allclose(2 * zarr_a, zarr_h)
    assert np.allclose(zarr_a**1.8, zarr_i)

    pk2d_j = (pk2d_a + 0.5*pk2d_i)**1.5
    _, _, zarr_j = pk2d_j.get_spline_arrays()
    assert np.allclose((zarr_a + 0.5*zarr_i)**1.5, zarr_j)


def test_pk2d_pkfunc_init_without_cosmo():
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    arr1 = ccl.Pk2D(pkfunc=lpk2d, cosmo=cosmo).get_spline_arrays()[-1]
    arr2 = ccl.Pk2D(pkfunc=lpk2d).get_spline_arrays()[-1]
    assert np.allclose(arr1, arr2, rtol=0)


def test_pk2d_extrap_orders():
    # Check that setting extrap orders propagates down to the `psp`.
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_a = np.outer(x, np.exp(log_y))
    pk = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=np.log(zarr_a), is_logp=True)

    assert pk.extrap_order_hik == pk.psp.extrap_order_hik
    assert pk.extrap_order_lok == pk.psp.extrap_order_lok


def test_pk2d_descriptor():
    # Check that `apply_halofit` can be called as a class method or
    # as an instance method.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    cosmo.compute_linear_power()
    pkl = cosmo.get_linear_power()
    pk1 = ccl.Pk2D.apply_halofit(cosmo, pk_linear=pkl)
    pk2 = pkl.apply_halofit(cosmo)
    assert np.all(pk1.get_spline_arrays()[-1] == pk2.get_spline_arrays()[-1])


def test_pk2d_eval_cosmo():
    # Check that `eval` can be called without `cosmo` and that an error
    # is raised when scale factor is out of interpolation range.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    cosmo.compute_linear_power()
    pk = cosmo.get_linear_power()
    assert pk.eval(1., 1.) == pk.eval(1., 1., cosmo)

    amin = pk.psp.amin
    pk.eval(1., amin*0.99, cosmo)  # doesn't fail because cosmo is provided
    with pytest.raises(TypeError):
        pk.eval(1., amin*0.99)


def test_pk2d_copy():
    # Check that copying works as intended (also check `bool`).
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_a = np.outer(x, np.exp(log_y))
    pk = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=np.log(zarr_a), is_logp=True)

    pkc = pk.copy()
    assert np.allclose(pk.get_spline_arrays()[-1],
                       pkc.get_spline_arrays()[-1],
                       rtol=1e-15)
    assert bool(pk) is bool(pkc) is True  # they both have `psp`

    pk = ccl.Pk2D(empty=True)
    pkc = pk.copy()
    assert bool(pk) is bool(pkc) is False


def test_pk2d_operations():
    # Everything is based on the already tested `add`, `mul`, and `pow`,
    # so we don't need to test every accepted type separately.
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_a = np.outer(x, np.exp(log_y))
    pk0 = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=np.log(zarr_a), is_logp=True)
    pk1, pk2 = pk0.copy(), pk0.copy()

    # sub, truediv
    assert np.allclose((pk1 - pk2).get_spline_arrays()[-1], 0, rtol=1e-15)
    assert np.allclose((pk1 / pk2).get_spline_arrays()[-1], 1, rtol=1e-15)

    # rsub, rtruediv
    assert np.allclose((1 - pk1).get_spline_arrays()[-1],
                       1 - pk1.get_spline_arrays()[-1])
    assert np.allclose((1 / pk1).get_spline_arrays()[-1],
                       1 / pk1.get_spline_arrays()[-1])

    # iadd, isub, imul, itruediv, ipow
    pk1 += pk1
    assert np.allclose((pk1 / pk2).get_spline_arrays()[-1], 2, rtol=1e-15)
    pk1 -= pk2
    assert np.allclose((pk1 / pk2).get_spline_arrays()[-1], 1, rtol=1e-15)
    pk1 *= pk1
    assert np.allclose(pk1.get_spline_arrays()[-1],
                       pk2.get_spline_arrays()[-1]**2,
                       rtol=1e-15)
    pk1 /= pk2
    assert np.allclose((pk1 / pk2).get_spline_arrays()[-1], 1, rtol=1e-15)
    pk1 **= 2
    assert np.allclose(pk1.get_spline_arrays()[-1],
                       pk2.get_spline_arrays()[-1]**2,
                       rtol=1e-15)


def test_pk2d_from_model_smoke():
    # Verify that both `from_model` methods are equivalent.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    pk1 = ccl.Pk2D.from_model(cosmo, "bbks")
    pk2 = ccl.Pk2D.pk_from_model(cosmo, "bbks")
    assert np.all(pk1.get_spline_arrays()[-1] == pk2.get_spline_arrays()[-1])
