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
