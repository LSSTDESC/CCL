import numpy as np
from numpy.testing import (
    assert_,
    assert_raises, assert_almost_equal, assert_allclose)
import pyccl as ccl


def pk1d(k):
    return ((k+0.001)/0.1)**(-1)


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

    # Input function but no cosmo
    assert_raises(ValueError, ccl.Pk2D, pkfunc=lpk2d)

    # Input arrays have incorrect sizes
    lkarr = -4.+6*np.arange(100)/99.
    aarr = 0.05+0.95*np.arange(100)/99.
    pkarr = np.zeros([len(aarr), len(lkarr)])
    assert_raises(
        ValueError, ccl.Pk2D, a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr[1:])


def test_pk2d_smoke():
    """Make sure it works once."""
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    lkarr = -4.+6*np.arange(100)/99.
    aarr = 0.05+0.95*np.arange(100)/99.
    pkarr = np.zeros([len(aarr), len(lkarr)])
    psp = ccl.Pk2D(a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr)
    assert_(not np.isnan(psp.eval(1E-2, 0.5, cosmo)))


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

    ktest = 1
    atest = 0.5
    ptrue = pk2d(ktest, atest)
    phere = psp.eval(ktest, atest, cosmo)
    assert_almost_equal(np.fabs(phere/ptrue), 1., 6)

    # Test at array of points
    ktest = np.logspace(-3, 1, 10)
    ptrue = pk2d(ktest, atest)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)

    # Test input is not logarithmic
    psp = ccl.Pk2D(pkfunc=pk2d, is_logp=False, cosmo=cosmo)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)

    # Test input is arrays
    karr = np.logspace(-4, 2, 1000)
    aarr = np.linspace(0.01, 1., 100)
    parr = np.array([pk2d(karr, a) for a in aarr])
    psp = ccl.Pk2D(
        a_arr=aarr, lk_arr=np.log(karr), pk_arr=parr, is_logp=False)
    phere = psp.eval(ktest, atest, cosmo)
    assert_allclose(phere, ptrue, rtol=1E-6)


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
