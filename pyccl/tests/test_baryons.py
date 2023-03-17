import pytest
import numpy as np
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(
    transfer_function='bbks',
    matter_power_spectrum='halofit')
COSMO.compute_nonlin_power()


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
    pk2d_nobar = COSMO.get_nonlin_power()
    pk2d_wbar = COSMO.baryon_correct("bcm", pk2d_nobar)
    pk_wbar = pk2d_wbar.eval(k_arr, 0.5, COSMO)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


def test_bcm_correct_raises():
    with pytest.raises(TypeError):
        COSMO.baryon_correct("bcm", None)


def test_baryon_correct_raises():
    pknl = COSMO.get_nonlin_power()
    with pytest.raises(NotImplementedError):
        ccl.baryon_correct(COSMO, model="hello_world", pk2d=pknl)


def test_func_deprecated():
    pknl1 = COSMO.get_nonlin_power().copy()
    pknl2 = COSMO.get_nonlin_power().copy()

    with pytest.warns(ccl.CCLDeprecationWarning):
        ccl.baryons.bcm_correct_pk2d(COSMO, pknl1)
    # new function is private and raises no warnings
    ccl.baryons._bcm_correct_pk2d(COSMO, pknl2)

    k_arr, a = np.logspace(-1, 1, 32), 1.
    assert np.allclose(pknl1.eval(k_arr, a), pknl2.eval(k_arr, a), rtol=0)


def test_arg_deprecated():
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo1 = ccl.CosmologyVanillaLCDM(
            bcm_log10Mc=14., bcm_etab=0.5, bcm_ks=55)
        cosmo1.compute_nonlin_power()
    cosmo2 = ccl.CosmologyVanillaLCDM(
        extra_parameters={"bcm": {"log10Mc": 14., "etab": 0.5, "ks": 55}})
    cosmo2.compute_nonlin_power()

    k_arr, a = np.logspace(-1, 1, 32), 1.
    assert np.allclose(cosmo1.nonlin_matter_power(k_arr, a),
                       cosmo2.nonlin_matter_power(k_arr, a), rtol=0)
