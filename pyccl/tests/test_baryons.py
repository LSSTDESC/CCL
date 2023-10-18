import numpy as np
import pyccl as ccl
import pytest


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks')
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
    pkb = bar.include_baryonic_effects(
        COSMO, COSMO.get_nonlin_power())
    pk_wbar = pkb(k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar/(pk_nobar*fka)-1) < 1E-5)


def test_bcm_update_params():
    bar2 = ccl.BaryonsSchneider15(log10Mc=14.1, eta_b=0.7, k_s=40.)
    bar2.update_parameters(log10Mc=bar.log10Mc,
                           eta_b=bar.eta_b,
                           k_s=bar.k_s)
    assert bar == bar2


def test_baryons_from_name():
    bar2 = ccl.Baryons.from_name('Schneider15')
    assert bar.name == bar2.name
    assert bar2.name == 'Schneider15'


def test_baryons_in_cosmology():
    # Test that applying baryons during cosmology creation works.
    # 1. Outside of cosmo
    cosmo_nb = ccl.CosmologyVanillaLCDM(
        transfer_function='bbks', baryonic_effects=None)
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)
    # 2. In cosmo - from object.
    cosmo_wb2 = ccl.CosmologyVanillaLCDM(
        transfer_function='bbks', baryonic_effects=bar)
    pk_wb2 = cosmo_wb2.get_nonlin_power()

    ks = np.geomspace(1E-2, 10, 128)
    pk_wb = pk_wb(ks, 1.0)
    pk_wb2 = pk_wb2(ks, 1.0)

    assert np.allclose(pk_wb, pk_wb2, atol=0, rtol=1E-6)


def test_baryons_in_cosmology_error():
    with pytest.raises(ValueError):
        ccl.CosmologyVanillaLCDM(baryonic_effects=3.1416)
