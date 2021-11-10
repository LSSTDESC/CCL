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
    ccl.bcm_correct_pk2d(COSMO,
                         COSMO._pk_nl['delta_matter:delta_matter'])
    pk_wbar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


@pytest.mark.parametrize('model', ['arico21'])
def test_baryon_correct_smoke(model):
    # we compare each model with BCM
    if model == "arico21":
        extras = {"arico21":
                  {'M_c'           :  14,
                   'eta'           : -0.3,
                   'beta'          : -0.22,
                   'M1_z0_cen'     : 10.5,
                   'theta_out'     : 0.25,
                   'theta_inn'     : -0.86,
                   'M_inn'         : 13.4}
                  }
        cosmo = ccl.CosmologyVanillaLCDM(
            transfer_function="bbks",
            matter_power_spectrum="halofit",
            extra_parameters=extras)

    k_arr = np.geomspace(1e-2, 1, 16)
    pknl = COSMO.get_nonlin_power()
    COSMO.baryon_correct("bcm", pknl)
    for z in [0., 0.5, 2.]:
        a = 1./(1+z)
        pk_bar = cosmo.baryon_correct(model, pknl)
        pk1 = pknl.eval(k_arr, a, COSMO)
        pk2 = pk_bar.eval(k_arr, a, COSMO)
        maxdiff = np.amax(np.fabs(1-pk1/pk2))
        assert maxdiff < 0.5  # be lenient for different baryon models!


def test_bcm_correct_raises():
    with pytest.raises(ValueError):
        ccl.bcm_correct_pk2d(COSMO, None)
