import pytest
import numpy as np
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM()
bar = ccl.BaryonsSchneider15()


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
    bar.include_baryonic_effects(
        COSMO, COSMO._pk_nl['delta_matter:delta_matter'],
        in_place=True)
    pk_wbar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


def test_baryons_from_name():
    bar2 = ccl.Baryons.from_name('Schneider15')
    assert bar.name == bar2.name
    assert bar2.name == 'Schneider15'
