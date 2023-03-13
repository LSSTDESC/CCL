import pytest
import numpy as np
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_bcm_smoke(k):
    a = 0.8
    fka = ccl.bcm_model_fka(COSMO, k, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k)


def test_bcm_correct_smoke():
    k_arr = np.geomspace(1E-2, 1, 10)
    fka = ccl.bcm_model_fka(COSMO, k_arr, 0.5)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    ccl.bcm_correct_pk2d(COSMO,
                         COSMO._pk_nl['delta_matter:delta_matter'])
    pk_wbar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


def test_bcm_correct_raises():
    with pytest.raises(TypeError):
        ccl.bcm_correct_pk2d(COSMO, None)
