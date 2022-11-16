import numpy as np
import pytest

import pyccl as ccl

from numpy.testing import assert_raises

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
PKA = ccl.Pk2D(lambda k, a: np.log(a/k), cosmo=COSMO)
ZZ = np.linspace(0., 1., 200)
NN = np.exp(-((ZZ-0.5)/0.1)**2)
LENS = ccl.WeakLensingTracer(COSMO, (ZZ, NN))


@pytest.mark.parametrize('p_of_k_a', [None, PKA])
def test_cls_smoke(p_of_k_a):
    # make a set of tracers to test with
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)
    b = np.sqrt(1. + z)
    lens1 = ccl.WeakLensingTracer(COSMO, (z, n))
    lens2 = ccl.WeakLensingTracer(COSMO, dndz=(z, n), ia_bias=(z, n))
    nc1 = ccl.NumberCountsTracer(COSMO, False, dndz=(z, n), bias=(z, b))
    nc2 = ccl.NumberCountsTracer(COSMO, True, dndz=(z, n), bias=(z, b))
    nc3 = ccl.NumberCountsTracer(
        COSMO, True, dndz=(z, n), bias=(z, b), mag_bias=(z, b))
    cmbl = ccl.CMBLensingTracer(COSMO, 1100.)
    tracers = [lens1, lens2, nc1, nc2, nc3, cmbl]

    ell_scl = 4.
    ell_int = 4
    ell_lst = [2, 3, 4, 5]
    ell_arr = np.arange(2, 5)
    ells = [ell_int, ell_scl, ell_lst, ell_arr]

    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            for ell in ells:
                corr = ccl.angular_cl(
                    COSMO, tracers[i], tracers[j], ell, p_of_k_a=p_of_k_a)
                assert np.all(np.isfinite(corr))
                assert np.shape(corr) == np.shape(ell)

                # reversing should be fine
                corr_rev = ccl.angular_cl(
                    COSMO, tracers[j], tracers[i], ell, p_of_k_a=p_of_k_a)
                assert np.allclose(corr, corr_rev)

    # Check invalid dndz
    with assert_raises(ValueError):
        ccl.NumberCountsTracer(COSMO, False, dndz=z, bias=(z, b))
    with assert_raises(ValueError):
        ccl.NumberCountsTracer(COSMO, False, dndz=(z, n, n), bias=(z, b))
    with assert_raises(ValueError):
        ccl.NumberCountsTracer(COSMO, False, dndz=(z,), bias=(z, b))
    with assert_raises(ValueError):
        ccl.NumberCountsTracer(COSMO, False, dndz=(1, 2), bias=(z, b))
    with assert_raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=z)
    with assert_raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z, n, n))
    with assert_raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z,))
    with assert_raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(1, 2))


@pytest.mark.parametrize('ells', [[3, 2, 1], [1, 3, 2], [2, 3, 1]])
def test_cls_raise_ell_reversed(ells):
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells)


def test_cls_raise_integ_method():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells,
                       limber_integration_method='guad')


def test_cls_raise_weird_pk():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells, p_of_k_a=lambda k, a: 10)


def test_cls_mg():
    # Check that if we feed the non-linear matter power spectrum from a MG
    # cosmology into a Calculator and get Cells using MG tracers, we get the
    # same results.

    # set up a MG cosmology
    cosmo_MG = ccl.CosmologyVanillaLCDM(mu_0=0.5, sigma_0=0.5,
                                        transfer_function="bbks",
                                        matter_power_spectrum="linear")
    cosmo_MG.compute_nonlin_power()
    pk2d = cosmo_MG.get_nonlin_power()

    # copy it into a calculator
    a, lk, pk = pk2d.get_spline_arrays()
    pk_nonlin = {"a": a, "k": np.exp(lk), "delta_matter:delta_matter": pk}
    cosmo_calc = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        mu_0=0.5, sigma_0=0.5, pk_nonlin=pk_nonlin)

    # get the Cells
    ell = np.geomspace(2, 2000, 128)
    tr_MG = ccl.CMBLensingTracer(cosmo_MG, 1100.)
    tr_calc = ccl.CMBLensingTracer(cosmo_calc, 1100.)

    cl0 = ccl.angular_cl(cosmo_MG, tr_MG, tr_MG, ell)
    cosmo_calc.compute_growth()
    cl1 = ccl.angular_cl(cosmo_calc, tr_calc, tr_calc, ell)
    assert np.all(np.fabs(1 - cl1 / cl0) < 1E-10)
