import pytest
import numpy as np
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM()
bar = ccl.baryons.BaryonsSchneider15()


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_bcm_smoke(k):
    a = 0.8
    fka = bar.boost_factor(COSMO, k, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k)


def test_bcm_correct_smoke():
    k_arr = np.geomspace(1E-2, 1, 10)
    fka = bar.boost_factor(COSMO, k_arr, 0.5)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    pkb = bar.include_baryonic_effects(
        COSMO, COSMO.get_nonlin_power())
    pk_wbar = pkb.eval(k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


def test_bcm_update_params():
    bar2 = ccl.baryons.BaryonsSchneider15(log10Mc=14.1, eta_b=0.7, k_s=40.)
    bar2.update_parameters(log10Mc=bar.log10Mc,
                           eta_b=bar.eta_b,
                           k_s=bar.k_s)
    assert bar == bar2


def test_baryons_from_name():
    bar2 = ccl.Baryons.from_name('Schneider15')
    assert bar.name == bar2.name
    assert bar2.name == 'Schneider15'
