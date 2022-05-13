import numpy as np
import pytest
from . import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
PKA = ccl.Pk2D(pkfunc=lambda k, a: np.log(a/k), cosmo=COSMO)
ZZ = np.linspace(0., 1., 256)
NN = np.exp(-((ZZ-0.5)/0.1)**2)
LENS = ccl.WeakLensingTracer(COSMO, dndz=(ZZ, NN))


@pytest.mark.parametrize('p_of_k_a', [None, PKA])
def test_cls_smoke(p_of_k_a):
    # make a set of tracers to test with
    z = np.linspace(0., 1., 256)
    n = np.exp(-((z-0.5)/0.1)**2)
    b = np.sqrt(1. + z)
    lens1 = ccl.WeakLensingTracer(COSMO, dndz=(z, n))
    lens2 = ccl.WeakLensingTracer(COSMO, dndz=(z, n), ia_bias=(z, n))
    nc1 = ccl.NumberCountsTracer(COSMO, dndz=(z, n), bias=(z, b),
                                 has_rsd=False)
    nc2 = ccl.NumberCountsTracer(COSMO, dndz=(z, n), bias=(z, b),
                                 has_rsd=True)
    nc3 = ccl.NumberCountsTracer(COSMO, dndz=(z, n), bias=(z, b),
                                 mag_bias=(z, b), has_rsd=True)
    cmbl = ccl.CMBLensingTracer(COSMO, z_source=1100.)
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
                    COSMO, tracers[i], tracers[j], ell=ell, p_of_k_a=p_of_k_a)
                assert np.all(np.isfinite(corr))
                assert np.shape(corr) == np.shape(ell)

                # reversing should be fine
                corr_rev = ccl.angular_cl(
                    COSMO, tracers[j], tracers[i], ell=ell, p_of_k_a=p_of_k_a)
                assert np.allclose(corr, corr_rev)

    # Check invalid dndz
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=z, bias=(z, b))
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, dndz=(z, n, n), bias=(z, b),
                               has_rsd=False)
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, dndz=(z,), bias=(z, b), has_rsd=False)
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, dndz=(1, 2), bias=(z, b), has_rsd=False)
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=z)
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z, n, n))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z,))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(1, 2))


@pytest.mark.parametrize('ells', [[3, 2, 1], [1, 3, 2], [2, 3, 1]])
def test_cls_raise_ell_reversed(ells):
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ell=ells)


def test_cls_raise_integ_method():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ell=ells,
                       limber_integration_method='guad')


def test_cls_raise_weird_pk():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ell=ells, p_of_k_a=lambda k, a: 10)
