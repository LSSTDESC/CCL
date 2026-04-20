"""Unit tests for the Van Daalen et al. 2019 baryonic effects model in CCL."""

from __future__ import annotations

import pytest

import numpy as np

import pyccl as ccl


# Set tolerances
BOOST_TOLERANCE = 1e-5

# Set up cosmology and k,a values
cosmo = ccl.CosmologyVanillaLCDM()
k = 0.5*cosmo['h']
a = 1.

# Set up power without baryons
pk2D_no_baryons = cosmo.get_nonlin_power()

# Set up fbarc and mass definition vectors
fbarcvec = np.linspace(0.25, 1, 20)
mdef_vec = ['500c', '200c']


def compare_boost():
    """Compares boost factor to expected values from Van Daalen+19."""
    vdboost = []
    vdboost_expect = []

    pk_nl = pk2D_no_baryons(k, a)

    for mdef in mdef_vec:
        for f in fbarcvec:
            # Takes ftilde as argument
            vd19 = ccl.BaryonsvanDaalen19(fbar=f, mass_def=mdef)
            pk2D_with_baryons = vd19.include_baryonic_effects(
                cosmo, pk2D_no_baryons)
            # Takes k in units of 1/Mpc as argument
            pk_nl_bar = pk2D_with_baryons(k, a)
            vdboost.append(pk_nl_bar/pk_nl-1)
            if mdef == '500c':
                vdboost_expect.append(-np.exp(-5.99*f-0.5107))
            else:
                vdboost_expect.append(-np.exp(-5.816*f-0.4005))

    assert np.allclose(
        vdboost, vdboost_expect, atol=1e-5, rtol=BOOST_TOLERANCE)


def test_boost_model():
    """Tests that the boost factor matches Van Daalen+19 results."""
    compare_boost()


def test_baryons_from_name():
    """Tests that from_name('vanDaalen19') works correctly."""
    baryons = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')
    bar2 = ccl.Baryons.from_name('vanDaalen19')

    assert baryons.name == bar2.name
    assert baryons.name == 'vanDaalen19'


def test_baryons_vd19_raises():
    """Tests that invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='blah')

    b = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')

    with pytest.raises(ValueError):
        b.update_parameters(mass_def='blah')


def test_update_params():
    """Tests that updating parameters works correctly."""
    b = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')
    b.update_parameters(fbar=0.6, mass_def='200c')

    assert b.mass_def == '200c'
    assert b.fbar == 0.6


@pytest.mark.parametrize(
    "k_in",
    [
        0.5 * cosmo["h"],  # scalar
        [0.1 * cosmo["h"], 0.5 * cosmo["h"], 1.0 * cosmo["h"]],
        np.array([0.1, 0.5, 1.0]) * cosmo["h"],
    ],
)
def test_vd19_boost_smoke(k_in):
    """Tests that boost_factor runs and returns finite values."""
    bar = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c")
    fka = bar.boost_factor(cosmo, k_in, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k_in)


def test_vd19_include_baryonic_effects_returns_pk2d():
    """Tests that include_baryonic_effects returns a Pk2D instance."""
    bar = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c")
    pk_nb = cosmo.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo, pk_nb)
    assert isinstance(pk_wb, ccl.Pk2D)


def test_vd19_baryons_in_cosmology():
    """Tests that baryonic effects are consistent via two methods."""
    bar = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c")

    cosmo_nb = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="eisenstein_hu",
        baryonic_effects=None,
    )
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)

    cosmo_wb = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="eisenstein_hu",
        baryonic_effects=bar,
    )
    pk_wb2 = cosmo_wb.get_nonlin_power()

    ks = np.geomspace(1e-2, 10.0, 64)
    np.testing.assert_allclose(
        pk_wb(ks, a),
        pk_wb2(ks, a),
        rtol=1e-6,
        atol=0.0,
    )
