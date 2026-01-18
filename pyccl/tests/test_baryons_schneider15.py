"""Unit tests for the Schneider15 baryonic correction model in CCL."""

from __future__ import annotations

import pytest

import numpy as np

import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
bar = ccl.BaryonsSchneider15()


@pytest.mark.parametrize(
    "k",
    [
        1,
        1.0,
        [0.3, 0.5, 10],
        np.array([0.3, 0.5, 10]),
    ],
)
def test_bcm_smoke(k):
    """Tests that boost_factor runs and returns finite values."""
    a = 0.8
    fka = bar.boost_factor(COSMO, k, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k)


def test_bcm_correct_smoke():
    """Tests that include_baryonic_effects matches boost_factor application."""
    k_arr = np.geomspace(1e-2, 1.0, 10)
    a = 0.5
    fka = bar.boost_factor(COSMO, k_arr, a)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, a)

    pkb = bar.include_baryonic_effects(COSMO, COSMO.get_nonlin_power())
    pk_wbar = pkb(k_arr, a)

    np.testing.assert_allclose(pk_wbar, pk_nobar * fka, rtol=1e-6, atol=0.0)


def test_bcm_update_params():
    """Tests that updating parameters works correctly."""
    bar2 = ccl.BaryonsSchneider15(log10Mc=14.1, eta_b=0.7, k_s=40.0)
    bar2.update_parameters(
        log10Mc=bar.log10Mc,
        eta_b=bar.eta_b,
        k_s=bar.k_s,
    )
    assert bar == bar2


def test_baryons_from_name():
    """Tests that from_name('Schneider15') works correctly."""
    bar2 = ccl.Baryons.from_name("Schneider15")
    assert bar.name == bar2.name
    assert bar2.name == "Schneider15"


def test_baryons_in_cosmology():
    """Test that baryonic effects are consistent via two methods."""
    # 1. Apply baryons outside the cosmology
    cosmo_nb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks",
        baryonic_effects=None,
    )
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)

    # 2. Apply baryons via Cosmology(baryonic_effects=...)
    cosmo_wb2 = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks",
        baryonic_effects=bar,
    )
    pk_wb2 = cosmo_wb2.get_nonlin_power()

    ks = np.geomspace(1e-2, 10.0, 128)
    pk_wb = pk_wb(ks, 1.0)
    pk_wb2 = pk_wb2(ks, 1.0)

    np.testing.assert_allclose(pk_wb, pk_wb2, rtol=1e-6, atol=0.0)


def test_baryons_in_cosmology_error():
    """Test that passing invalid baryonic_effects raises ValueError."""
    with pytest.raises(ValueError):
        ccl.CosmologyVanillaLCDM(baryonic_effects=3.1416)


def test_bcm_include_baryonic_effects_returns_pk2d():
    """Tests that include_baryonic_effects returns a Pk2D object."""
    pk_nb = COSMO.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(COSMO, pk_nb)
    assert isinstance(pk_wb, ccl.Pk2D)


def test_baryons_from_name_type():
    """Test that from_name returns the correct class type."""
    bar_cls = ccl.Baryons.from_name("Schneider15")
    assert issubclass(bar_cls, ccl.Baryons)
    assert bar_cls.name == "Schneider15"
